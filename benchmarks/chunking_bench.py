"""
INFRACORE BENCHMARK — Chunking Performance

Benchmarks fixed vs semantic chunking on throughput, chunk quality, and latency.
Run: python benchmarks/chunking_bench.py --dataset nq
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infracore.chunking.base import ChunkConfig
from src.infracore.chunking.fixed import FixedChunker
from src.infracore.chunking.semantic import SemanticChunker


# Sample texts for benchmarking (from NQ dev set style)
SAMPLE_TEXTS = [
    """
    The Great Wall of China is a series of fortifications that were built across 
    the historical northern borders of China to protect against various nomadic groups. 
    The wall was built by different dynasties over more than 2,000 years. The most well-known 
    section was built during the Ming Dynasty. The wall stretches over 13,000 miles in total length. 
    It is one of the most impressive architectural feats in human history.
    """,
    """
    Machine learning is a subset of artificial intelligence that provides systems the ability 
    to automatically learn and improve from experience without being explicitly programmed. 
    Machine learning focuses on the development of computer programs that can access data and use it 
    to learn for themselves. The process of learning begins with observations or data, such as examples, 
    direct experience, or instruction, in order to look for patterns in data and make better decisions 
    in the future based on the examples that we provide. The primary aim is to allow the computers to 
    learn automatically without human intervention or assistance and adjust actions accordingly.
    """,
    """
    Python is an interpreted, high-level, general-purpose programming language. Its design philosophy 
    emphasizes code readability with the use of significant indentation. Python is dynamically typed 
    and garbage collected. It supports multiple programming paradigms including structured, object-oriented, 
    and functional programming. Python was created in 1989 by Guido van Rossum and first released in 1991. 
    The language was named after the television series Monty Python's Flying Circus. Python has become one 
    of the most popular programming languages in the world due to its simplicity and versatility.
    """,
]


async def benchmark_chunker(
    chunker_name: str,
    chunker,
    texts: list,
    num_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark a chunker on a set of texts.

    Args:
        chunker_name: Name of the chunker
        chunker: Chunker instance
        texts: List of texts to chunk
        num_iterations: Number of iterations per text

    Returns:
        Benchmark results dict
    """
    total_start = time.perf_counter()
    total_chunks = 0
    total_words = 0
    total_chars = 0
    chunk_sizes = []
    latencies = []

    for text in texts:
        for _ in range(num_iterations):
            start = time.perf_counter()
            chunks = await chunker.chunk(text)
            latency = time.perf_counter() - start
            latencies.append(latency)

            total_chunks += len(chunks)
            total_words += sum(len(c.text.split()) for c in chunks)
            total_chars += sum(len(c.text) for c in chunks)
            chunk_sizes.extend([len(c.text.split()) for c in chunks])

    total_elapsed = time.perf_counter() - total_start

    # Calculate metrics
    chunks_per_second = total_chunks / total_elapsed
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0
    min_latency_ms = min(latencies) * 1000 if latencies else 0
    max_latency_ms = max(latencies) * 1000 if latencies else 0
    p50_latency_ms = sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0

    return {
        "chunker": chunker_name,
        "total_chunks": total_chunks,
        "chunks_per_second": round(chunks_per_second, 2),
        "avg_chunk_size_words": round(avg_chunk_size, 2),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "min_latency_ms": round(min_latency_ms, 4),
        "max_latency_ms": round(max_latency_ms, 4),
        "p50_latency_ms": round(p50_latency_ms, 4),
        "total_time_sec": round(total_elapsed, 2),
        "total_iterations": len(texts) * num_iterations,
    }


async def run_benchmarks(dataset: str = "default") -> Dict[str, Any]:
    """
    Run benchmarks on all chunkers.

    Args:
        dataset: Dataset name (for future extensibility)

    Returns:
        Results dict
    """
    print(f"\n{'=' * 80}")
    print(f"INFRACORE CHUNKING BENCHMARK — {dataset.upper()}")
    print(f"{'=' * 80}\n")

    # Prepare chunkers
    fixed_config = ChunkConfig(strategy="fixed", max_tokens=128, overlap=16, min_chunk_size=32)
    semantic_config = ChunkConfig(
        strategy="semantic", max_tokens=128, overlap=16, min_chunk_size=32
    )

    fixed_chunker = FixedChunker(fixed_config)
    semantic_chunker = SemanticChunker(semantic_config)

    # Run benchmarks
    print("Running Fixed Chunker benchmark...")
    fixed_results = await benchmark_chunker("fixed", fixed_chunker, SAMPLE_TEXTS, num_iterations=50)
    print(f"  ✓ {fixed_results['chunks_per_second']} chunks/sec")

    print("Running Semantic Chunker benchmark...")
    semantic_results = await benchmark_chunker("semantic", semantic_chunker, SAMPLE_TEXTS, num_iterations=50)
    print(f"  ✓ {semantic_results['chunks_per_second']} chunks/sec")

    # Compare
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}\n")

    throughput_ratio = fixed_results["chunks_per_second"] / semantic_results["chunks_per_second"]
    print(f"Throughput ratio (fixed/semantic): {throughput_ratio:.2f}x")
    print(f"  Fixed: {fixed_results['chunks_per_second']} chunks/sec")
    print(f"  Semantic: {semantic_results['chunks_per_second']} chunks/sec")

    size_diff = fixed_results["avg_chunk_size_words"] - semantic_results["avg_chunk_size_words"]
    print(f"\nAverage chunk size difference: {size_diff:.1f} words")
    print(f"  Fixed: {fixed_results['avg_chunk_size_words']} words")
    print(f"  Semantic: {semantic_results['avg_chunk_size_words']} words")

    latency_ratio = semantic_results["avg_latency_ms"] / fixed_results["avg_latency_ms"]
    print(f"\nLatency ratio (semantic/fixed): {latency_ratio:.2f}x")
    print(f"  Fixed avg: {fixed_results['avg_latency_ms']:.4f} ms")
    print(f"  Semantic avg: {semantic_results['avg_latency_ms']:.4f} ms")

    print(f"\n{'=' * 80}\n")

    return {
        "dataset": dataset,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fixed": fixed_results,
        "semantic": semantic_results,
        "comparison": {
            "throughput_ratio_fixed_over_semantic": round(throughput_ratio, 2),
            "avg_chunk_size_diff_fixed_minus_semantic": round(size_diff, 2),
            "latency_ratio_semantic_over_fixed": round(latency_ratio, 2),
        },
    }


if __name__ == "__main__":
    import sys

    dataset = "default"
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            dataset = sys.argv[idx + 1]

    results = asyncio.run(run_benchmarks(dataset))

    # Save results
    output_file = f"eval_reports/chunking_bench_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_file}")
    print(json.dumps(results, indent=2))
