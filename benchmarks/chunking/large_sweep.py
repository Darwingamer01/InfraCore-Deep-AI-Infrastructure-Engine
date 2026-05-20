"""Larger-scale chunking benchmark sweep.

Generates a diverse synthetic corpus and runs FixedChunker and SemanticChunker.
Saves JSON results and a markdown summary.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import List

import sys, os
sys.path.insert(0, os.getcwd())

from src.infracore.chunking.fixed import FixedChunker
from src.infracore.chunking.semantic import SemanticChunker
from src.infracore.chunking.recursive import RecursiveChunker
from src.infracore.chunking.base import ChunkConfig
from benchmarks.chunking.bench_report import render_markdown


RESULT_DIR = Path("benchmarks/chunking/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def gen_sentence(rng: random.Random, min_w=5, max_w=20) -> str:
    words = [
        "system",
        "model",
        "data",
        "retrieval",
        "chunk",
        "service",
        "performance",
        "query",
        "document",
        "embedding",
        "token",
        "result",
        "index",
        "analysis",
        "response",
    ]
    n = rng.randint(min_w, max_w)
    return " ".join(rng.choice(words) for _ in range(n)) + "."


def generate_corpus(seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    docs: List[str] = []

    # Short docs
    for _ in range(100):
        docs.append(" ".join(gen_sentence(rng, 3, 8) for _ in range(2)))

    # Medium docs
    for _ in range(200):
        docs.append(" ".join(gen_sentence(rng, 6, 15) for _ in range(20)))

    # Long docs
    for _ in range(50):
        docs.append(" ".join(gen_sentence(rng, 8, 25) for _ in range(200)))

    # List-like docs
    for _ in range(50):
        items = ["- " + gen_sentence(rng, 3, 8) for _ in range(rng.randint(10, 50))]
        docs.append("\n".join(items))

    # Paragraph-heavy docs
    for _ in range(100):
        docs.append("\n\n".join(" ".join(gen_sentence(rng, 8, 30) for _ in range(rng.randint(3, 10))) for _ in range(rng.randint(3, 8))))

    return docs


async def _run_chunker_async(chunker, doc: str):
    res = chunker.chunk(doc)
    if hasattr(res, "__await__"):
        return await res
    return res


def run_sweep() -> dict:
    docs = generate_corpus()

    cfg_fixed = ChunkConfig(strategy="fixed", max_tokens=50, overlap=5, min_chunk_size=5)
    cfg_sem = ChunkConfig(strategy="semantic", max_tokens=50, overlap=1, min_chunk_size=3)
    cfg_rec = ChunkConfig(strategy="recursive", max_tokens=50, overlap=2, min_chunk_size=3)

    runners = [
        ("fixed", FixedChunker(cfg_fixed)),
        ("semantic", SemanticChunker(cfg_sem)),
        ("recursive", RecursiveChunker(cfg_rec)),
    ]

    results = {"timestamp": time.time(), "runs": []}

    for name, chunker in runners:
        latencies: List[float] = []
        total_chunks = 0
        total_chunk_sizes = []

        for doc in docs:
            t0 = time.perf_counter()
            chunks = asyncio.run(_run_chunker_async(chunker, doc))
            t1 = time.perf_counter()

            latencies.append(t1 - t0)
            total_chunks += len(chunks)
            if chunks:
                total_chunk_sizes.append(mean(len(c.text.split()) for c in chunks))

        run = {
            "name": name,
            "num_chunks": total_chunks,
            "avg_chunk_size": mean(total_chunk_sizes) if total_chunk_sizes else 0.0,
            "p50_latency": percentile(latencies, 50),
            "p95_latency": percentile(latencies, 95),
            "latencies": latencies,
        }
        results["runs"].append(run)

    out = RESULT_DIR / f"results_large_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))

    # Render markdown
    md_path = str(out.with_suffix(".md"))
    render_markdown(str(out), md_path)

    return {"json": str(out), "md": md_path, "results": results}


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


def main() -> None:
    res = run_sweep()
    print("Wrote JSON:", res["json"])
    print("Wrote MD:", res["md"])
    # Print concise summary
    for run in res["results"]["runs"]:
        print(f"{run['name'].title()} -> chunks: {run['num_chunks']}, avg_size: {run['avg_chunk_size']:.2f}, p50: {run['p50_latency']:.4f}s, p95: {run['p95_latency']:.4f}s")


if __name__ == "__main__":
    main()
