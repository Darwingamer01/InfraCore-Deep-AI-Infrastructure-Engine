#!/usr/bin/env python
"""
INFRACORE — Qdrant Benchmark Harness

Purpose: Measure QPS, Recall@10, p99 latency, RAM usage across HNSW configs
Uses: qdrant-client (SYNC for benchmarking — async overhead skews latency)
      psutil for RAM measurement
      dataset_generator for loading .npz benchmark data
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import psutil
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, HnswConfigDiff, PointStruct, SearchParams, VectorParams
from tqdm import tqdm

from benchmarks.vectordb.dataset_generator import load_dataset


@dataclass
class BenchConfig:
    """Qdrant benchmark configuration."""

    dataset_path: str
    hnsw_m: int
    hnsw_ef_construct: int
    hnsw_ef: int
    batch_size: int = 256
    collection_name: str = "bench"
    qdrant_url: str = "http://localhost:6333"
    n_warmup_queries: int = 10


@dataclass
class BenchResult:
    """Benchmark results."""

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


class QdrantBench:
    """Qdrant benchmark harness."""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_path)
        self.client = QdrantClient(url=config.qdrant_url)

    def run(self) -> BenchResult:
        """Run complete benchmark."""
        config_name = f"m{self.config.hnsw_m}_ef{self.config.hnsw_ef}"
        print(f"\n{'='*70}")
        print(f"Qdrant Benchmark: {config_name}")
        print(f"  Dataset: {Path(self.config.dataset_path).name}")
        print(f"  Vectors: {len(self.dataset.vectors):,} × {self.dataset.vectors.shape[1]}D")
        print(f"  HNSW m={self.config.hnsw_m} ef_construct={self.config.hnsw_ef_construct} ef_search={self.config.hnsw_ef}")
        print(f"{'='*70}")

        # STEP 1 — RAM baseline
        print("\n📊 Recording baseline RAM...")
        ram_before = psutil.Process().memory_info().rss / (1024**2)
        print(f"   RAM before: {ram_before:.1f} MB")

        # STEP 2 — Create collection with HNSW config
        print("\n🔨 Creating Qdrant collection...")
        self.client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.dataset.vectors.shape[1], distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
            ),
        )
        index_build_time_s = 0.0  # Collection creation is instant, actual indexing happens during insert

        # STEP 3 — Upsert vectors in batches
        print(f"\n📝 Upserting {len(self.dataset.vectors):,} vectors...")
        upsert_start = time.perf_counter()

        vectors = self.dataset.vectors
        batches = (len(vectors) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min((batch_idx + 1) * self.config.batch_size, len(vectors))

            points = [
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"idx": i},
                )
                for i in range(start_idx, end_idx)
            ]

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )

            if (batch_idx + 1) % max(1, batches // 10) == 0:
                print(f"   Upserted {end_idx:,}/{len(vectors):,}")

        upsert_time_s = time.perf_counter() - upsert_start
        print(f"   Upsert time: {upsert_time_s:.2f}s")

        # STEP 4 — Warmup
        print(f"\n🔥 Warmup ({self.config.n_warmup_queries} queries)...")
        for i in range(min(self.config.n_warmup_queries, len(self.dataset.queries))):
            query = self.dataset.queries[i]
            self.client.query_points(
                collection_name=self.config.collection_name,
                query=query.tolist(),
                limit=10,
                search_params=SearchParams(hnsw_ef=self.config.hnsw_ef),
            )

        # STEP 5 — Timed search (all queries)
        print(f"\n⏱️  Running {len(self.dataset.queries)} timed queries...")
        latencies = []
        recalled_ids_per_query = []

        for i, query in enumerate(self.dataset.queries):
            t0 = time.perf_counter()
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                query=query.tolist(),
                limit=10,
                search_params=SearchParams(hnsw_ef=self.config.hnsw_ef),
            )
            latencies.append((time.perf_counter() - t0) * 1000)  # ms
            recalled_ids = [int(r.id) for r in results.points]
            recalled_ids_per_query.append(recalled_ids)

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
        self.client.delete_collection(collection_name=self.config.collection_name)

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


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Qdrant Benchmark Harness")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument(
        "--configs",
        type=str,
        default="m8_ef64,m8_ef128,m16_ef128,m16_ef256,m32_ef256",
        help="Comma-separated configs: m{M}_ef{EF}",
    )
    parser.add_argument(
        "--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant server URL"
    )
    parser.add_argument(
        "--output", type=str, default="eval_reports", help="Output directory for results"
    )

    args = parser.parse_args()

    # Parse configs
    config_strs = [c.strip() for c in args.configs.split(",")]
    configs = []

    for config_str in config_strs:
        # Parse "m8_ef64" → m=8, ef=64, ef_construct=128
        parts = config_str.lower().split("_")
        m = int(parts[0][1:])  # "m8" → 8
        ef = int(parts[1][2:])  # "ef64" → 64
        ef_construct = ef * 2  # Double for construction

        configs.append(
            BenchConfig(
                dataset_path=args.dataset,
                hnsw_m=m,
                hnsw_ef_construct=ef_construct,
                hnsw_ef=ef,
                qdrant_url=args.qdrant_url,
            )
        )

    # Run benchmarks
    print(f"\n🚀 Qdrant Benchmark Suite")
    print(f"   Dataset: {args.dataset}")
    print(f"   Configs: {args.configs}")

    results = []
    for config in configs:
        bench = QdrantBench(config)
        result = bench.run()
        results.append(result)

    # Print summary table
    print(f"\n\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(
        f"{'Config':<15} {'QPS':<12} {'Recall@10':<12} {'p50 (ms)':<12} {'p99 (ms)':<12} {'RAM (MB)':<12}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result.config_name:<15} {result.qps:>10,.0f} {result.recall_at_10:>10.3f}  {result.p50_latency_ms:>10.1f}  {result.p99_latency_ms:>10.1f}  {result.ram_delta_mb:>+10.0f}"
        )

    print(f"{'='*100}\n")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(args.dataset).stem
    timestamp = datetime.now().isoformat().replace(":", "-")
    output_file = output_dir / f"qdrant_bench_{dataset_name}_{timestamp}.json"

    results_dict = {
        "dataset": args.dataset,
        "configs": args.configs,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }

    output_file.write_text(json.dumps(results_dict, indent=2))
    print(f"✅ Results saved to: {output_file}\n")

    return results_dict


if __name__ == "__main__":
    main()
