#!/usr/bin/env python
"""
INFRACORE — Inference Throughput & Latency Benchmark (Scaffold)

Purpose: Measure tokens/sec, p99 latency and memory footprint across inference
backends. This scaffold implements the harness, engine adapters (Ollama/HuggingFace)
and I/O for results. It intentionally avoids heavy work at import time and
provides `available()` checks so tests can skip gracefully.

Usage (scaffold):
  python benchmarks/inference/throughput_bench.py --backend ollama --model llama3.2:1b

Notes:
- This scaffold follows the `BenchConfig` / `BenchResult` pattern used elsewhere.
- Implementations for real measurements are provided as placeholders and should
  be filled in when you want to run full benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil

# Optional imports (lazy)
try:
    import httpx
except Exception:
    httpx = None  # type: ignore

try:
    import transformers
    from transformers import pipeline
except Exception:
    transformers = None  # type: ignore


@dataclass
class BenchConfig:
    backend: str  # 'ollama' | 'hf'
    model: str
    concurrency: int
    batch_size: int
    n_requests: int
    prompt_length: int
    base_url: Optional[str] = None  # for Ollama
    device: str = "cpu"
    timeout_seconds: int = 120


@dataclass
class BenchResult:
    config_name: str
    backend: str
    model: str
    concurrency: int
    batch_size: int
    n_requests: int
    tokens_per_second: float
    p99_latency_ms: float
    memory_mb_delta: float
    error_rate: float
    requests_succeeded: int
    total_time_s: float


class BenchBase:
    def __init__(self, config: BenchConfig, prompts: List[dict]):
        self.config = config
        self.prompts = prompts

    async def available(self) -> bool:
        """Return whether the backend is available on this machine."""
        raise NotImplementedError()

    async def run(self) -> BenchResult:
        """Run the benchmark and return a BenchResult."""
        raise NotImplementedError()


class OllamaBench(BenchBase):
    def __init__(self, config: BenchConfig, prompts: List[dict]):
        super().__init__(config, prompts)
        # Lazy client creation
        self.client = None
        if httpx and config.base_url:
            self.client = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout_seconds)

    async def available(self) -> bool:
        return httpx is not None and self.client is not None

    async def generate(self, prompt: str) -> tuple[float, int, bool, str]:
        # Placeholder: implement HTTP call to Ollama and measure TTFT and tokens
        start = time.perf_counter()
        await asyncio.sleep(0.01)  # simulate network + generation
        total_ms = (time.perf_counter() - start) * 1000
        return total_ms * 0.3, 20, True, ""

    async def run(self) -> BenchResult:
        # Very lightweight scaffolded run that simulates timings for unit tests
        ram_before = psutil.Process().memory_info().rss / (1024**2)
        start = time.perf_counter()
        # Simulate work
        await asyncio.sleep(0.1)
        total_time = time.perf_counter() - start
        ram_after = psutil.Process().memory_info().rss / (1024**2)

        # Simulated numbers
        tokens_per_second = 150.0
        p99 = 250.0

        return BenchResult(
            config_name=f"ollama_{self.config.model}",
            backend="ollama",
            model=self.config.model,
            concurrency=self.config.concurrency,
            batch_size=self.config.batch_size,
            n_requests=self.config.n_requests,
            tokens_per_second=tokens_per_second,
            p99_latency_ms=p99,
            memory_mb_delta=ram_after - ram_before,
            error_rate=0.0,
            requests_succeeded=self.config.n_requests,
            total_time_s=total_time,
        )


class HFBench(BenchBase):
    def __init__(self, config: BenchConfig, prompts: List[dict]):
        super().__init__(config, prompts)
        self.pipeline = None
        if transformers is not None:
            # Lazy pipeline creation
            try:
                self.pipeline = pipeline("text-generation", model=config.model, device=0 if config.device != "cpu" else -1)
            except Exception:
                self.pipeline = None

    async def available(self) -> bool:
        return transformers is not None and self.pipeline is not None

    async def run(self) -> BenchResult:
        # Scaffolded run: do not actually invoke heavy model inference here
        ram_before = psutil.Process().memory_info().rss / (1024**2)
        start = time.perf_counter()
        await asyncio.sleep(0.1)
        total_time = time.perf_counter() - start
        ram_after = psutil.Process().memory_info().rss / (1024**2)

        # Simulated numbers; replace with measured values when implementing
        tokens_per_second = 200.0
        p99 = 180.0

        return BenchResult(
            config_name=f"hf_{self.config.model}",
            backend="hf",
            model=self.config.model,
            concurrency=self.config.concurrency,
            batch_size=self.config.batch_size,
            n_requests=self.config.n_requests,
            tokens_per_second=tokens_per_second,
            p99_latency_ms=p99,
            memory_mb_delta=ram_after - ram_before,
            error_rate=0.0,
            requests_succeeded=self.config.n_requests,
            total_time_s=total_time,
        )


async def run_benchmark(config: BenchConfig, prompts: List[dict]) -> BenchResult:
    """Factory-run convenience: pick engine and run benchmark if available."""
    if config.backend == "ollama":
        bench = OllamaBench(config, prompts)
    elif config.backend == "hf":
        bench = HFBench(config, prompts)
    else:
        raise ValueError("Unsupported backend")

    if not await bench.available():
        raise RuntimeError(f"Backend not available: {config.backend}")

    return await bench.run()


def save_result(result: BenchResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat().replace(":", "-")
    out_file = output_dir / f"throughput_{result.config_name}_{timestamp}.json"
    out = {"config": asdict(result)}
    out_file.write_text(json.dumps(out, indent=2))
    return out_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Throughput benchmark scaffold")
    parser.add_argument("--backend", type=str, default="ollama")
    parser.add_argument("--model", type=str, default="llama3.2:1b")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-requests", type=int, default=10)
    parser.add_argument("--prompt-length", type=int, default=50)
    parser.add_argument("--output", type=str, default="eval_reports")
    args = parser.parse_args()

    # Load small synthetic prompt set
    dataset = [
        {"id": 0, "text": "Hello world" * (args.prompt_length // 2), "category": "short"}
    ]

    cfg = BenchConfig(
        backend=args.backend,
        model=args.model,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        n_requests=args.n_requests,
        prompt_length=args.prompt_length,
        base_url="http://localhost:11434",
    )

    try:
        res = asyncio.run(run_benchmark(cfg, dataset))
        saved = save_result(res, Path(args.output))
        print("Saved:", saved)
    except RuntimeError as e:
        print("Skipping benchmark:", e)
