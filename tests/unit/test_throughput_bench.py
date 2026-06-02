import asyncio
import importlib
import sys
from pathlib import Path
import json

import pytest


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
p99_ms = bench_mod.p99_ms
ollama_tokens_per_second = bench_mod.ollama_tokens_per_second
write_results_json = bench_mod.write_results_json
write_markdown_summary = bench_mod.write_markdown_summary
BenchResult = bench_mod.BenchResult
OllamaBench = bench_mod.OllamaBench
HFBench = bench_mod.HFBench


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


def test_p99_ms_helper():
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert p99_ms(samples) == 50.0


def test_ollama_tps_helper():
    # 200 tokens in 2 seconds
    tps = ollama_tokens_per_second(200, 2_000_000_000)
    assert round(tps, 2) == 100.00


def test_result_writers(tmp_path):
    result = BenchResult(
        config_name="hf_fake_b1_p50",
        backend="hf",
        model="fake/model",
        concurrency=1,
        batch_size=1,
        n_requests=10,
        tokens_per_second=12.5,
        p99_latency_ms=250.0,
        memory_mb_delta=100.0,
        error_rate=0.0,
        requests_succeeded=10,
        total_time_s=4.0,
        ttft_ms_mean=0.0,
    )
    json_path = write_results_json([result], tmp_path)
    md_path = write_markdown_summary([result], tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text())
    assert "results" in payload
    assert payload["results"][0]["backend"] == "hf"

    md = md_path.read_text()
    assert "| engine | model |" in md
    assert "fake/model" in md


def test_ollama_run_warms_up_before_measurement(monkeypatch):
    cfg = BenchConfig(
        backend="ollama",
        model="llama3.2:1b",
        concurrency=1,
        batch_size=1,
        n_requests=2,
        prompt_length=50,
        base_url=None,
    )
    bench = OllamaBench(cfg, [{"text": "p1"}, {"text": "p2"}])
    call_log = []

    async def fake_total(prompt):
        call_log.append(("total", prompt))
        return 10.0, 20.0, True

    async def fake_ttft(prompt):
        call_log.append(("ttft", prompt))
        return 5.0, True

    monkeypatch.setattr(bench, "measure_total_latency_and_tps", fake_total)
    monkeypatch.setattr(bench, "measure_ttft_stream", fake_ttft)

    result = asyncio.run(bench.run())

    assert result.requests_succeeded == 10
    assert call_log[0] == ("total", "p1")
    assert call_log[1] == ("ttft", "p1")
    assert len([item for item in call_log if item[0] == "total"]) == 11
    assert len([item for item in call_log if item[0] == "ttft"]) == 11


def test_hf_run_warms_up_before_measurement(monkeypatch):
    cfg = BenchConfig(
        backend="hf",
        model="fake/model",
        concurrency=1,
        batch_size=4,
        n_requests=2,
        prompt_length=50,
    )
    monkeypatch.setattr(bench_mod, "pipeline", None)
    bench = HFBench(cfg, [{"text": "p1"}, {"text": "p2"}])
    call_log = []

    async def fake_generate_batch(prompts):
        call_log.append(list(prompts))
        return 0.1, 10, True

    monkeypatch.setattr(bench, "_generate_batch", fake_generate_batch)

    result = asyncio.run(bench.run())

    assert result.requests_succeeded == 10
    assert call_log[0] == ["p1", "p1", "p1", "p1"]
    assert len(call_log) == 11


def test_hf_pipeline_pad_token_is_configured(monkeypatch):
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 50256
            self.eos_token = "<eos>"
            self.pad_token_id = None
            self.pad_token = None

    class DummyModelConfig:
        def __init__(self):
            self.pad_token_id = None

    class DummyModel:
        def __init__(self):
            self.config = DummyModelConfig()

    class DummyPipeline:
        def __init__(self):
            self.tokenizer = DummyTokenizer()
            self.model = DummyModel()

        def __call__(self, *args, **kwargs):
            return [[{"generated_text": "dummy output"}]]

    monkeypatch.setattr(bench_mod, "pipeline", lambda *args, **kwargs: DummyPipeline())

    cfg = BenchConfig(
        backend="hf",
        model="sshleifer/tiny-gpt2",
        concurrency=1,
        batch_size=4,
        n_requests=2,
        prompt_length=50,
        device="cpu",
    )
    bench = HFBench(cfg, [{"text": "p1"}, {"text": "p2"}])
    assert bench.generator is not None
    assert bench.generator.tokenizer.pad_token_id == 50256
    assert bench.generator.tokenizer.pad_token == "<eos>"
    assert bench.generator.model.config.pad_token_id == 50256


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

    skipped = asyncio.run(_run())
    assert skipped is True
