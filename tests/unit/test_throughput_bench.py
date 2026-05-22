import asyncio
import importlib
import sys
from pathlib import Path


# Robust import: add repo root to sys.path if needed so tests can run in CI and local
try:
    bench_mod = importlib.import_module("benchmarks.inference.throughput_bench")
except ModuleNotFoundError:
    # Fallback: load module directly from file path
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "benchmarks" / "inference" / "throughput_bench.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("throughput_bench", module_path)
    bench_mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(bench_mod)

BenchConfig = bench_mod.BenchConfig
run_benchmark = bench_mod.run_benchmark


def test_bench_module_loads_and_has_classes():
    # Basic smoke test: classes are importable and BenchConfig can be instantiated.
    cfg = BenchConfig(
        backend="ollama",
        model="llama3.2:1b",
        concurrency=1,
        batch_size=1,
        n_requests=1,
        prompt_length=50,
    )
    assert cfg.backend == "ollama"


def test_run_benchmark_skips_when_unavailable():
    # Use an event loop to check that run_benchmark raises RuntimeError when backend unavailable
    cfg = BenchConfig(
        backend="hf",
        model="nonexistent-model",
        concurrency=1,
        batch_size=1,
        n_requests=1,
        prompt_length=50,
    )

    async def _run():
        try:
            await run_benchmark(cfg, [{"id": 0, "text": "hello", "category": "short"}])
        except RuntimeError:
            return True
        return False

    loop = asyncio.get_event_loop()
    skipped = loop.run_until_complete(_run())
    assert skipped is True
