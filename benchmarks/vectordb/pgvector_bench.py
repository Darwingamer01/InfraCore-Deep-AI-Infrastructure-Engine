#!/usr/bin/env python
"""
INFRACORE — pgvector Benchmark Harness

Purpose: Measure QPS, Recall@10, p99 latency, RAM usage on PostgreSQL + pgvector
Uses: PgVectorStore async interface (sync wrapped for latency measurement)
      psutil for RAM measurement
      dataset_generator for loading .npz benchmark data

Parallel to qdrant_bench.py for direct backend comparison.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import psutil

from benchmarks.vectordb.dataset_generator import load_dataset
from src.infracore.vectordb.pgvector_store import PgVectorConfig, PgVectorStore


@dataclass
class BenchConfig:
    """pgvector benchmark configuration."""

    dataset_path: str
    batch_size: int = 256
    collection_name: str = "bench"
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/infracore"
    n_warmup_queries: int = 10
    index_type: str = "hnsw"  # hnsw or ivfflat
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128


@dataclass
class BenchResult:
    """Benchmark results (matches Qdrant structure for comparability)."""

    config_name: str
    dataset_size: int
    vector_dims: int
    index_build_time_s: float
    upsert_time_s: float
    qps: float
    recall_at_10: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    ram_mb_before: float
    ram_mb_after: float
    ram_delta_mb: float


class PgVectorBench:
    """pgvector benchmark harness."""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_path)
        
        # Create PgVectorStore config
        self.store_config = PgVectorConfig(
            collection_name=config.collection_name,
            vector_size=self.dataset.vectors.shape[1],
            dsn=config.pg_dsn,
            table_name=config.collection_name,
            index_type=config.index_type,
            hnsw_m=config.hnsw_m,
            hnsw_ef_construct=config.hnsw_ef_construct,
        )
        self.store = PgVectorStore(self.store_config)

    async def run(self) -> BenchResult:
        """Run complete benchmark."""
        config_name = f"pgvector_{self.config.index_type}"
        if self.config.index_type == "hnsw":
            config_name += f"_m{self.config.hnsw_m}_ef{self.config.hnsw_ef_construct}"

        print(f"\n{'='*70}")
        print(f"pgvector Benchmark: {config_name}")
        print(f"  Dataset: {Path(self.config.dataset_path).name}")
        print(f"  Vectors: {len(self.dataset.vectors):,} × {self.dataset.vectors.shape[1]}D")
        print(f"  Index: {self.config.index_type}")
        if self.config.index_type == "hnsw":
            print(f"    m={self.config.hnsw_m} ef_construct={self.config.hnsw_ef_construct}")
        print(f"{'='*70}")

        # STEP 1 — RAM baseline
        print("\n📊 Recording baseline RAM...")
        ram_before = psutil.Process().memory_info().rss / (1024**2)
        print(f"   RAM before: {ram_before:.1f} MB")

        # STEP 2 — Create table with index
        print("\n🔨 Creating pgvector table...")
        try:
            await self.store.create_table()
            print(f"   Table created: {self.config.collection_name}")
        except Exception as e:
            # Table might already exist, try to clear it
            print(f"   Note: {e}")

        index_build_time_s = 0.0  # Index building happens during insert

        # STEP 3 — Upsert vectors in batches
        print(f"\n📝 Upserting {len(self.dataset.vectors):,} vectors...")
        upsert_start = time.perf_counter()

        vectors = self.dataset.vectors
        batches = (len(vectors) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min((batch_idx + 1) * self.config.batch_size, len(vectors))

            batch_vectors = vectors[start_idx:end_idx]
            batch_payloads = [{"idx": i} for i in range(start_idx, end_idx)]
            batch_ids = [str(i) for i in range(start_idx, end_idx)]

            await self.store.upsert(batch_vectors, batch_payloads, batch_ids)

            if (batch_idx + 1) % max(1, batches // 10) == 0:
                print(f"   Upserted {end_idx:,}/{len(vectors):,}")

        upsert_time_s = time.perf_counter() - upsert_start
        print(f"   Upsert time: {upsert_time_s:.2f}s")

        # STEP 4 — Warmup
        print(f"\n🔥 Warmup ({self.config.n_warmup_queries} queries)...")
        for i in range(min(self.config.n_warmup_queries, len(self.dataset.queries))):
            query = self.dataset.queries[i]
            await self.store.search(query, top_k=10)

        # STEP 5 — Timed search (all queries)
        print(f"\n⏱️  Running {len(self.dataset.queries)} timed queries...")
        latencies = []
        recalled_ids_per_query = []

        for i, query in enumerate(self.dataset.queries):
            t0 = time.perf_counter()
            results = await self.store.search(query, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            # Extract IDs from results
            recalled_ids = [int(float(r.payload.get("idx", 0))) for r in results]
            recalled_ids_per_query.append(recalled_ids)

            if (i + 1) % max(1, len(self.dataset.queries) // 10) == 0:
                print(f"   Queries run: {i + 1}/{len(self.dataset.queries)}")

        # STEP 6 — Recall@10
        recalls = []
        for i in range(len(self.dataset.queries)):
            ground_truth_set = set(self.dataset.ground_truth[i])
            returned_set = set(recalled_ids_per_query[i])
            recall_i = len(ground_truth_set & returned_set) / 10.0
            recalls.append(recall_i)

        recall_at_10 = np.mean(recalls)

        # STEP 7 — Percentiles
        latencies_array = np.array(latencies)
        p50_latency_ms = float(np.percentile(latencies_array, 50))
        p95_latency_ms = float(np.percentile(latencies_array, 95))
        p99_latency_ms = float(np.percentile(latencies_array, 99))

        total_query_time_s = sum(latencies) / 1000.0
        qps = len(self.dataset.queries) / total_query_time_s

        # STEP 8 — RAM after
        ram_after = psutil.Process().memory_info().rss / (1024**2)
        ram_delta = ram_after - ram_before

        # STEP 9 — Cleanup
        print("\n🧹 Cleaning up...")
        try:
            await self.store.close()
        except Exception as e:
            print(f"   Note during close: {e}")

        # Results
        result = BenchResult(
            config_name=config_name,
            dataset_size=len(self.dataset.vectors),
            vector_dims=self.dataset.vectors.shape[1],
            index_build_time_s=index_build_time_s,
            upsert_time_s=upsert_time_s,
            qps=qps,
            recall_at_10=recall_at_10,
            p50_latency_ms=p50_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            ram_mb_before=ram_before,
            ram_mb_after=ram_after,
            ram_delta_mb=ram_delta,
        )

        print(f"\n{'='*70}")
        print(f"Results: {config_name}")
        print(f"{'='*70}")
        print(f"  QPS: {result.qps:,.0f}")
        print(f"  Recall@10: {result.recall_at_10:.3f}")
        print(f"  Latency p50/p95/p99: {result.p50_latency_ms:.1f}/{result.p95_latency_ms:.1f}/{result.p99_latency_ms:.1f} ms")
        print(f"  RAM delta: {result.ram_delta_mb:+.0f} MB")
        print(f"  Upsert time: {result.upsert_time_s:.2f}s")

        return result


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="pgvector Benchmark Harness")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument(
        "--index-type",
        type=str,
        default="hnsw",
        choices=["hnsw", "ivfflat"],
        help="Index type",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="HNSW M parameter",
    )
    parser.add_argument(
        "--hnsw-ef-construct",
        type=int,
        default=128,
        help="HNSW ef_construct parameter",
    )
    parser.add_argument(
        "--pg-dsn",
        type=str,
        default="postgresql://postgres:postgres@localhost:5432/infracore",
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--output", type=str, default="eval_reports", help="Output directory for results"
    )

    args = parser.parse_args()

    config = BenchConfig(
        dataset_path=args.dataset,
        pg_dsn=args.pg_dsn,
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construct=args.hnsw_ef_construct,
    )

    bench = PgVectorBench(config)
    result = await bench.run()

    # Save results to JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    result_file = output_dir / f"pgvector_bench_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump({"config": asdict(config), "results": [asdict(result)]}, f, indent=2)

    print(f"\n✅ Results saved to: {result_file}")


if __name__ == "__main__":
    asyncio.run(main())
