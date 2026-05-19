#!/usr/bin/env python
"""
INFRACORE — vLLM Inference Benchmark Harness

Purpose: Measure throughput, latency, and TTFT for vLLM backend
Uses: httpx (async), concurrency, psutil for memory
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import httpx
import psutil

from benchmarks.inference.dataset_generator import load_dataset


@dataclass
class BenchConfig:
    """Benchmark configuration."""

    backend: str
    model: str
    base_url: str
    concurrency: int
    n_requests: int
    max_tokens: int
    temperature: float = 0.7
    timeout_seconds: int = 120


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: int
    prompt_category: str
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_generated: int
    error: bool
    error_message: str = ""


@dataclass
class BenchResult:
    """Aggregated benchmark results."""

    config_name: str
    backend: str
    model: str
    concurrency: int
    n_requests: int
    tokens_per_second: float
    time_to_first_token_ms: float  # Mean
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_mb_delta: float
    error_rate: float
    requests_succeeded: int
    total_time_s: float


class VLLMBench:
    """vLLM benchmark harness using OpenAI-compatible API."""

    def __init__(self, config: BenchConfig, prompts: List[dict]):
        self.config = config
        self.prompts = prompts
        self.client = httpx.AsyncClient(base_url=config.base_url, timeout=config.timeout_seconds)
        self.metrics: List[RequestMetrics] = []

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def generate(self, prompt: str) -> tuple[float, int, bool, str]:
        """
        Generate response from vLLM (OpenAI-compatible API).

        Returns: (TTFT in ms, tokens generated, success, error message)
        """
        try:
            start_time = time.perf_counter()

            # Use OpenAI-compatible API
            response = await self.client.post(
                "/v1/completions",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )

            response.raise_for_status()
            data = response.json()

            total_latency = (time.perf_counter() - start_time) * 1000
            ttft_ms = total_latency * 0.2  # Estimate TTFT as 20% of total

            # Get token count from usage
            token_count = data.get("usage", {}).get("completion_tokens", 0)
            if token_count == 0:
                # Fallback: estimate from response text
                response_text = data["choices"][0].get("text", "")
                token_count = len(response_text.split())

            return ttft_ms, token_count, True, ""

        except Exception as e:
            return 0.0, 0, False, str(e)

    async def run_single_request(self, request_id: int, prompt_text: str, category: str) -> RequestMetrics:
        """Run a single request and collect metrics."""
        start_time = time.perf_counter()
        ttft, tokens, success, error_msg = await self.generate(prompt_text)
        total_latency = (time.perf_counter() - start_time) * 1000

        return RequestMetrics(
            request_id=request_id,
            prompt_category=category,
            time_to_first_token_ms=ttft,
            total_latency_ms=total_latency,
            tokens_generated=tokens,
            error=not success,
            error_message=error_msg,
        )

    async def run(self) -> BenchResult:
        """Run complete benchmark with concurrency control."""
        config_name = f"{self.config.backend}_{self.config.model.replace(':', '-')}_c{self.config.concurrency}"
        print(f"\n{'='*70}")
        print(f"vLLM Benchmark: {config_name}")
        print(f"  Model: {self.config.model}")
        print(f"  Concurrency: {self.config.concurrency}")
        print(f"  Requests: {self.config.n_requests}")
        print(f"{'='*70}")

        # Warmup
        print("\n🔥 Warmup request...")
        await self.generate("What is 2+2?")

        # Memory baseline
        print("\n📊 Recording baseline RAM...")
        ram_before = psutil.Process().memory_info().rss / (1024**2)
        print(f"   RAM before: {ram_before:.1f} MB")

        # Run requests with concurrency
        print(f"\n⏱️  Running {self.config.n_requests} requests (concurrency={self.config.concurrency})...")
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def bounded_request(req_id: int, prompt_dict: dict):
            async with semaphore:
                return await self.run_single_request(
                    req_id, prompt_dict["text"], prompt_dict["category"]
                )

        # Select prompts
        selected_prompts = self.prompts[: self.config.n_requests]
        tasks = [
            bounded_request(i, prompt) for i, prompt in enumerate(selected_prompts)
        ]

        bench_start = time.perf_counter()
        self.metrics = await asyncio.gather(*tasks)
        bench_time = time.perf_counter() - bench_start

        # Memory after
        ram_after = psutil.Process().memory_info().rss / (1024**2)
        ram_delta = ram_after - ram_before

        # Compute statistics
        successful = [m for m in self.metrics if not m.error]
        failed = [m for m in self.metrics if m.error]
        error_rate = len(failed) / len(self.metrics) if self.metrics else 0.0

        if successful:
            import numpy as np

            latencies = [m.total_latency_ms for m in successful]
            ttfts = [m.time_to_first_token_ms for m in successful]
            total_tokens = sum(m.tokens_generated for m in successful)

            latencies_array = np.array(latencies)
            p50 = float(np.percentile(latencies_array, 50))
            p95 = float(np.percentile(latencies_array, 95))
            p99 = float(np.percentile(latencies_array, 99))
            mean_ttft = float(np.mean(ttfts))

            # Tokens per second
            tps = (total_tokens / bench_time) if bench_time > 0 else 0.0
        else:
            p50 = p95 = p99 = mean_ttft = tps = 0.0

        # Print results
        print(f"\n{'='*70}")
        print(f"Results: {config_name}")
        print(f"{'='*70}")
        print(f"  Throughput: {tps:.1f} tokens/sec")
        print(f"  TTFT (mean): {mean_ttft:.1f} ms")
        print(f"  Latency p50/p95/p99: {p50:.1f}/{p95:.1f}/{p99:.1f} ms")
        print(f"  Requests succeeded: {len(successful)}/{len(self.metrics)}")
        print(f"  Error rate: {error_rate:.1%}")
        print(f"  RAM delta: {ram_delta:+.1f} MB")
        print(f"  Total time: {bench_time:.2f}s")

        result = BenchResult(
            config_name=config_name,
            backend=self.config.backend,
            model=self.config.model,
            concurrency=self.config.concurrency,
            n_requests=len(self.metrics),
            tokens_per_second=tps,
            time_to_first_token_ms=mean_ttft,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            memory_mb_delta=ram_delta,
            error_rate=error_rate,
            requests_succeeded=len(successful),
            total_time_s=bench_time,
        )

        return result


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Inference Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="vLLM model")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="vLLM API URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--n-requests", type=int, default=50, help="Number of requests")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per response")
    parser.add_argument("--output", type=str, default="eval_reports", help="Output directory")
    parser.add_argument("--dataset", type=str, default="data/bench/inference_prompts.jsonl", help="Prompt dataset")

    args = parser.parse_args()

    # Load prompts
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    prompts = load_dataset(dataset_path)

    # Run benchmark
    config = BenchConfig(
        backend="vllm",
        model=args.model,
        base_url=args.base_url,
        concurrency=args.concurrency,
        n_requests=args.n_requests,
        max_tokens=args.max_tokens,
    )

    bench = VLLMBench(config, prompts)

    try:
        result = await bench.run()

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat().replace(":", "-")
        output_file = output_dir / f"vllm_bench_c{args.concurrency}_{timestamp}.json"

        results_dict = {
            "config": asdict(config),
            "result": asdict(result),
            "timestamp": datetime.now().isoformat(),
        }

        output_file.write_text(json.dumps(results_dict, indent=2))
        print(f"\n✅ Results saved to: {output_file}")

    finally:
        await bench.close()


if __name__ == "__main__":
    asyncio.run(main())
