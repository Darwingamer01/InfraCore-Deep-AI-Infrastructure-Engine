"""Chunking benchmark runner

Runs FixedChunker and SemanticChunker over sample documents and
emits JSON results and optional charts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.infracore.chunking.fixed import FixedChunker
from src.infracore.chunking.semantic import SemanticChunker
from src.infracore.chunking.recursive import RecursiveChunker
from src.infracore.chunking.base import ChunkConfig


RESULT_DIR = Path("benchmarks/chunking/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkerStats:
    name: str
    num_chunks: int
    avg_chunk_size: float
    latencies: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_chunks": self.num_chunks,
            "avg_chunk_size": self.avg_chunk_size,
            "p50_latency": percentile(self.latencies, 50),
            "p95_latency": percentile(self.latencies, 95),
            "latencies": self.latencies,
        }


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return data[int(k)]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1


class ChunkingBench:
    def __init__(self, configs: List[ChunkConfig]):
        self.configs = configs

    def _make_docs(self) -> List[str]:
        # Sample documents: short, medium, long
        short = "This is a short document. " * 5
        medium = "This is a medium document with several sentences. " * 50
        long = "Long document content. " * 500
        return [short, medium, long]

    def run(self) -> Dict[str, Any]:
        docs = self._make_docs()
        results: Dict[str, Any] = {"runs": [], "timestamp": time.time()}

        for cfg in self.configs:
            if cfg.strategy == "fixed":
                chunker = FixedChunker(cfg)
            elif cfg.strategy == "semantic":
                chunker = SemanticChunker(cfg)
            else:
                chunker = RecursiveChunker(cfg)

            latencies: List[float] = []
            total_chunks = 0
            total_chunk_size = 0

            for doc in docs:
                t0 = time.perf_counter()
                chunks = chunker.chunk(doc)
                # chunker.chunk is async in interface; but implementations are sync-compatible
                # Ensure compatibility by awaiting if returned is coroutine
                if hasattr(chunks, "__await__"):
                    import asyncio

                    chunks = asyncio.get_event_loop().run_until_complete(chunks)
                t1 = time.perf_counter()
                latencies.append(t1 - t0)

                total_chunks += len(chunks)
                total_chunk_size += mean([len(c.text.split()) for c in chunks]) if chunks else 0

            stats = ChunkerStats(
                name=cfg.strategy,
                num_chunks=total_chunks,
                avg_chunk_size=(total_chunk_size / 3.0),
                latencies=latencies,
            )

            results["runs"].append(stats.to_dict())

        # Persist results
        out = RESULT_DIR / f"results_{int(time.time())}.json"
        out.write_text(json.dumps(results, indent=2))
        return {"path": str(out), "results": results}


def main() -> None:
    cfg_fixed = ChunkConfig(strategy="fixed", max_tokens=50, overlap=5, min_chunk_size=10)
    cfg_sem = ChunkConfig(strategy="semantic", max_tokens=50, overlap=1, min_chunk_size=5)
    cfg_rec = ChunkConfig(strategy="recursive", max_tokens=50, overlap=2, min_chunk_size=5)
    bench = ChunkingBench([cfg_fixed, cfg_sem, cfg_rec])
    res = bench.run()
    print("Wrote results to:", res["path"])


if __name__ == "__main__":
    main()
