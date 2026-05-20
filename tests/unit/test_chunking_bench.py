"""Unit tests for chunking benchmark harness.

These tests are lightweight and validate runner behavior and serialization.
"""
import json
from pathlib import Path

from benchmarks.chunking.chunking_bench import ChunkingBench
from src.infracore.chunking.base import ChunkConfig


def test_chunking_bench_run_tmp(tmp_path: Path):
    cfg_fixed = ChunkConfig(strategy="fixed", max_tokens=20, overlap=2, min_chunk_size=5)
    cfg_sem = ChunkConfig(strategy="semantic", max_tokens=20, overlap=1, min_chunk_size=3)

    bench = ChunkingBench([cfg_fixed, cfg_sem])
    res = bench.run()

    assert "path" in res and "results" in res
    p = Path(res["path"])
    assert p.exists()

    data = res["results"]
    assert "runs" in data and len(data["runs"]) == 2

    # Check keys in run
    k = data["runs"][0]
    for key in ("name", "num_chunks", "avg_chunk_size", "p50_latency", "p95_latency"):
        assert key in k

    # Cleanup
    p.unlink()
