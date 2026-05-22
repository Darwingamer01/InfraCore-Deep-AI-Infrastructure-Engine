#!/usr/bin/env python
"""Inference throughput/latency benchmark harness for Sprint 3.

Implements real measurement functions for:
- Ollama (`/api/generate` with stream false for total latency and stream true for TTFT)
- Hugging Face batched generation (`transformers.pipeline`)

Outputs:
- Timestamped JSON results
- Markdown summary table
- Two charts (throughput vs latency, memory vs batch)

This file provides harness logic only. It does not auto-run heavy benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional

import psutil

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    from transformers import pipeline
except Exception:
    pipeline = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore[assignment]


SEED = 42
N_RUNS = 10


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def p99_ms(samples_ms: list[float]) -> float:
    if not samples_ms:
        return 0.0
    ordered = sorted(samples_ms)
    idx = max(0, min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1)))))
    return float(ordered[idx])


def ollama_tokens_per_second(eval_count: int, eval_duration_ns: int) -> float:
    if eval_duration_ns <= 0:
        return 0.0
    return float(eval_count) / (float(eval_duration_ns) / 1_000_000_000.0)


@dataclass
class BenchConfig:
    backend: str  # 'ollama' | 'hf'
    model: str
    concurrency: int
    batch_size: int
    n_requests: int
    prompt_length: int
    base_url: Optional[str] = None
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
    ttft_ms_mean: float = 0.0


class BenchBase:
    def __init__(self, config: BenchConfig, prompts: list[dict[str, Any]]):
        self.config = config
        self.prompts = prompts

    async def available(self) -> bool:
        raise NotImplementedError()

    async def run(self) -> BenchResult:
        raise NotImplementedError()


class OllamaBench(BenchBase):
    def __init__(self, config: BenchConfig, prompts: list[dict[str, Any]]):
        super().__init__(config, prompts)
        self.client = None
        if httpx is not None and config.base_url:
            self.client = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout_seconds)

    async def available(self) -> bool:
        if self.client is None:
            return False
        try:
            resp = await self.client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    async def measure_total_latency_and_tps(self, prompt: str) -> tuple[float, float, bool]:
        if self.client is None:
            return 0.0, 0.0, False
        start = time.perf_counter()
        try:
            resp = await self.client.post(
                "/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()
            tps = ollama_tokens_per_second(
                int(data.get("eval_count", 0)), int(data.get("eval_duration", 0))
            )
            return elapsed_ms, tps, True
        except Exception:
            return 0.0, 0.0, False

    async def measure_ttft_stream(self, prompt: str) -> tuple[float, bool]:
        if self.client is None:
            return 0.0, False
        start = time.perf_counter()
        try:
            async with self.client.stream(
                "POST",
                "/api/generate",
                json={"model": self.config.model, "prompt": prompt, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        return (time.perf_counter() - start) * 1000.0, True
            return 0.0, False
        except Exception:
            return 0.0, False

    async def run(self) -> BenchResult:
        set_deterministic_seed(SEED)
        prompts = [p["text"] for p in self.prompts[: self.config.n_requests]]
        if not prompts:
            prompts = ["Hello"]

        ram_before = psutil.Process().memory_info().rss / (1024**2)
        latencies: list[float] = []
        ttfts: list[float] = []
        tps_values: list[float] = []
        ok = 0

        start_all = time.perf_counter()
        for i in range(N_RUNS):
            prompt = prompts[i % len(prompts)]
            lat_ms, tps, success = await self.measure_total_latency_and_tps(prompt)
            ttft_ms, ttft_ok = await self.measure_ttft_stream(prompt)
            if success:
                latencies.append(lat_ms)
                tps_values.append(tps)
                ok += 1
            if ttft_ok:
                ttfts.append(ttft_ms)

        total_time_s = time.perf_counter() - start_all
        ram_after = psutil.Process().memory_info().rss / (1024**2)

        err = float((N_RUNS - ok) / N_RUNS)
        return BenchResult(
            config_name=f"ollama_{self.config.model.replace(':', '-')}_b{self.config.batch_size}_p{self.config.prompt_length}",
            backend="ollama",
            model=self.config.model,
            concurrency=self.config.concurrency,
            batch_size=self.config.batch_size,
            n_requests=N_RUNS,
            tokens_per_second=float(statistics.mean(tps_values)) if tps_values else 0.0,
            p99_latency_ms=p99_ms(latencies),
            memory_mb_delta=float(ram_after - ram_before),
            error_rate=err,
            requests_succeeded=ok,
            total_time_s=total_time_s,
            ttft_ms_mean=float(statistics.mean(ttfts)) if ttfts else 0.0,
        )


class HFBench(BenchBase):
    def __init__(self, config: BenchConfig, prompts: list[dict[str, Any]]):
        super().__init__(config, prompts)
        self.generator = None
        if pipeline is not None:
            try:
                device = 0 if (config.device != "cpu" and torch is not None and torch.cuda.is_available()) else -1
                self.generator = pipeline("text-generation", model=config.model, device=device)
            except Exception:
                self.generator = None

    async def available(self) -> bool:
        return self.generator is not None

    def _memory_mb(self) -> float:
        if torch is not None and torch.cuda.is_available() and self.config.device != "cpu":
            return float(torch.cuda.memory_allocated() / (1024**2))
        return float(psutil.Process().memory_info().rss / (1024**2))

    async def _generate_batch(self, prompts: list[str]) -> tuple[float, int, bool]:
        if self.generator is None:
            return 0.0, 0, False

        start = time.perf_counter()
        try:
            outputs = await asyncio.to_thread(
                self.generator,
                prompts,
                max_new_tokens=64,
                do_sample=False,
                batch_size=self.config.batch_size,
            )
            elapsed = time.perf_counter() - start

            # outputs is usually list[list[dict]] for batch input
            total_tokens = 0
            for item in outputs:
                if isinstance(item, list) and item:
                    txt = item[0].get("generated_text", "")
                elif isinstance(item, dict):
                    txt = item.get("generated_text", "")
                else:
                    txt = ""
                total_tokens += max(0, len(txt.split()))

            return elapsed, total_tokens, True
        except Exception:
            return 0.0, 0, False

    async def run(self) -> BenchResult:
        set_deterministic_seed(SEED)

        base_prompts = [p["text"] for p in self.prompts if p.get("text")]
        if not base_prompts:
            base_prompts = ["Hello world"]

        latencies_ms: list[float] = []
        total_tokens = 0
        ok = 0
        mem_before = self._memory_mb()
        start_all = time.perf_counter()

        for i in range(N_RUNS):
            batch_prompts = [
                base_prompts[(i + j) % len(base_prompts)] for j in range(self.config.batch_size)
            ]
            elapsed_s, out_tokens, success = await self._generate_batch(batch_prompts)
            if success:
                latencies_ms.append(elapsed_s * 1000.0)
                total_tokens += out_tokens
                ok += 1

        total_time_s = time.perf_counter() - start_all
        mem_after = self._memory_mb()

        tps = float(total_tokens / total_time_s) if total_time_s > 0 else 0.0
        return BenchResult(
            config_name=f"hf_{self.config.model.replace('/', '-')}_b{self.config.batch_size}_p{self.config.prompt_length}",
            backend="hf",
            model=self.config.model,
            concurrency=self.config.concurrency,
            batch_size=self.config.batch_size,
            n_requests=N_RUNS,
            tokens_per_second=tps,
            p99_latency_ms=p99_ms(latencies_ms),
            memory_mb_delta=float(mem_after - mem_before),
            error_rate=float((N_RUNS - ok) / N_RUNS),
            requests_succeeded=ok,
            total_time_s=total_time_s,
            ttft_ms_mean=0.0,
        )


def write_results_json(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat().replace(":", "-")
    p = output_dir / f"throughput_bench_{ts}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "n_runs": N_RUNS,
        "results": [asdict(r) for r in results],
    }
    p.write_text(json.dumps(payload, indent=2))
    return p


def write_markdown_summary(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "throughput_bench_summary.md"
    lines = [
        "# Inference Throughput Benchmark Summary",
        "",
        "| engine | model | batch_size | tokens/sec | p99ms | memory_MB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.backend} | {r.model} | {r.batch_size} | {r.tokens_per_second:.2f} | {r.p99_latency_ms:.2f} | {r.memory_mb_delta:.2f} |"
        )
    p.write_text("\n".join(lines) + "\n")
    return p


def plot_throughput_vs_latency(results: list[BenchResult], output_dir: Path) -> Optional[Path]:
    if plt is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "throughput_vs_latency.png"

    engines = sorted({r.backend for r in results})
    for eng in engines:
        xs = [r.p99_latency_ms for r in results if r.backend == eng]
        ys = [r.tokens_per_second for r in results if r.backend == eng]
        plt.scatter(xs, ys, label=eng)
    plt.xlabel("p99 latency (ms)")
    plt.ylabel("tokens/sec")
    plt.title("Throughput vs Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


def plot_memory_vs_batch(results: list[BenchResult], output_dir: Path) -> Optional[Path]:
    if plt is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "memory_vs_batch.png"

    engines = sorted({r.backend for r in results})
    batches = sorted({r.batch_size for r in results})
    width = 0.35
    x = list(range(len(batches)))

    for idx, eng in enumerate(engines):
        vals = []
        for b in batches:
            match = [r.memory_mb_delta for r in results if r.backend == eng and r.batch_size == b]
            vals.append(match[0] if match else 0.0)
        offsets = [v + (idx * width) for v in x]
        plt.bar(offsets, vals, width=width, label=eng)

    plt.xticks([v + width / 2 for v in x], [str(b) for b in batches])
    plt.xlabel("batch size")
    plt.ylabel("memory delta (MB)")
    plt.title("Memory vs Batch Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


async def run_benchmark(config: BenchConfig, prompts: list[dict[str, Any]]) -> BenchResult:
    if config.backend == "ollama":
        bench: BenchBase = OllamaBench(config, prompts)
    elif config.backend == "hf":
        bench = HFBench(config, prompts)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    if not await bench.available():
        raise RuntimeError(f"Backend not available: {config.backend}")

    result = await bench.run()
    if isinstance(bench, OllamaBench):
        await bench.close()
    return result


def _make_prompt(length_tokens: int) -> str:
    # Simple deterministic template prompt, roughly length-controlled by repetition.
    base = "inference benchmark token "
    repeats = max(1, length_tokens // 3)
    return (base * repeats).strip()


def build_synthetic_prompts(prompt_length: int, n_prompts: int) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "text": _make_prompt(prompt_length),
            "category": "short" if prompt_length <= 50 else "medium",
        }
        for i in range(n_prompts)
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inference throughput benchmark harness")
    parser.add_argument("--backend", type=str, choices=["ollama", "hf"], default="ollama")
    parser.add_argument("--model", type=str, default="llama3.2:1b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=50)
    parser.add_argument("--output", type=str, default="eval_reports")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = BenchConfig(
        backend=args.backend,
        model=args.model,
        concurrency=1,
        batch_size=args.batch_size,
        n_requests=N_RUNS,
        prompt_length=args.prompt_length,
        base_url=args.base_url,
        device=args.device,
    )
    prompts = build_synthetic_prompts(args.prompt_length, n_prompts=32)

    # Harness implementation exists, but we intentionally do not auto-run matrix jobs here.
    # This command runs a single config if backend is available.
    try:
        single = asyncio.run(run_benchmark(cfg, prompts))
        out_dir = Path(args.output)
        json_file = write_results_json([single], out_dir)
        md_file = write_markdown_summary([single], out_dir)
        chart1 = plot_throughput_vs_latency([single], out_dir)
        chart2 = plot_memory_vs_batch([single], out_dir)
        print(f"Saved JSON: {json_file}")
        print(f"Saved Markdown: {md_file}")
        print(f"Saved Chart 1: {chart1}")
        print(f"Saved Chart 2: {chart2}")
    except RuntimeError as e:
        print(f"Skipping benchmark: {e}")
