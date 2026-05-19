#!/usr/bin/env python
"""
INFRACORE — Inference Benchmark Report Generator

Purpose: Load JSON results from inference benchmarks, generate charts and summary
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_files: List[str]) -> Dict[str, Any]:
    """Load and merge JSON results from benchmark files."""
    all_results = []

    for json_file in json_files:
        path = Path(json_file)
        if not path.exists():
            print(f"Warning: {json_file} not found")
            continue

        with open(path) as f:
            data = json.load(f)
            all_results.append(data["result"])

    return {"results": all_results}


def plot_throughput_vs_concurrency(results: List[Dict], output_dir: Path) -> None:
    """Generate tokens/sec vs concurrency chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by backend
    backends = {}
    for result in results:
        backend = result["backend"]
        if backend not in backends:
            backends[backend] = {"concurrency": [], "tps": []}

        backends[backend]["concurrency"].append(result["concurrency"])
        backends[backend]["tps"].append(result["tokens_per_second"])

    # Plot lines
    colors = {"ollama": "blue", "vllm": "orange"}
    for backend, data in sorted(backends.items()):
        sorted_data = sorted(zip(data["concurrency"], data["tps"]))
        concurrency, tps = zip(*sorted_data)
        ax.plot(
            concurrency,
            tps,
            marker="o",
            linewidth=2,
            markersize=8,
            label=backend.upper(),
            color=colors.get(backend, "gray"),
        )

    ax.set_xlabel("Concurrency", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tokens/Second", fontsize=12, fontweight="bold")
    ax.set_title("Inference Throughput: Tokens/Sec vs Concurrency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    output_file = output_dir / "inference_throughput_vs_concurrency.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_p99_latency_vs_concurrency(results: List[Dict], output_dir: Path) -> None:
    """Generate p99 latency vs concurrency chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by backend
    backends = {}
    for result in results:
        backend = result["backend"]
        if backend not in backends:
            backends[backend] = {"concurrency": [], "p99": []}

        backends[backend]["concurrency"].append(result["concurrency"])
        backends[backend]["p99"].append(result["p99_latency_ms"])

    # Plot lines
    colors = {"ollama": "blue", "vllm": "orange"}
    for backend, data in sorted(backends.items()):
        sorted_data = sorted(zip(data["concurrency"], data["p99"]))
        concurrency, p99 = zip(*sorted_data)
        ax.plot(
            concurrency,
            p99,
            marker="s",
            linewidth=2,
            markersize=8,
            label=backend.upper(),
            color=colors.get(backend, "gray"),
        )

    # SLA line at 100ms
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="100ms SLA")

    ax.set_xlabel("Concurrency", fontsize=12, fontweight="bold")
    ax.set_ylabel("p99 Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Tail Latency: p99 vs Concurrency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    output_file = output_dir / "inference_p99_latency_vs_concurrency.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_ttft_comparison(results: List[Dict], output_dir: Path) -> None:
    """Generate TTFT (time-to-first-token) comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    ttfts = []
    colors = []

    color_map = {"ollama": "blue", "vllm": "orange"}

    for result in results:
        config = f"{result['backend']}\nc={result['concurrency']}"
        configs.append(config)
        ttfts.append(result["time_to_first_token_ms"])
        colors.append(color_map.get(result["backend"], "gray"))

    bars = ax.bar(configs, ttfts, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, ttft in zip(bars, ttfts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{ttft:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Time-to-First-Token (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Time-to-First-Token: Streaming Latency Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "inference_ttft_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def generate_markdown_table(results: List[Dict], output_dir: Path) -> str:
    """Generate markdown summary table."""
    lines = [
        "# Inference Benchmark Summary",
        "",
        "| Backend | Concurrency | Throughput (tok/s) | TTFT (ms) | p50 (ms) | p99 (ms) | Error Rate |",
        "|---------|-------------|-------------------|-----------|----------|----------|------------|",
    ]

    for result in sorted(results, key=lambda x: (x["backend"], x["concurrency"])):
        backend = result["backend"].upper()
        concurrency = result["concurrency"]
        tps = result["tokens_per_second"]
        ttft = result["time_to_first_token_ms"]
        p50 = result["p50_latency_ms"]
        p99 = result["p99_latency_ms"]
        error_rate = result["error_rate"]

        line = f"| {backend} | {concurrency} | {tps:,.0f} | {ttft:.1f} | {p50:.1f} | {p99:.1f} | {error_rate:.1%} |"
        lines.append(line)

    markdown = "\n".join(lines)

    # Save to file
    output_file = output_dir / "inference_summary.md"
    output_file.write_text(markdown)

    return markdown


def main():
    """CLI entry point."""
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description="Inference Benchmark Report Generator")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Glob pattern for JSON result files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_reports",
        help="Output directory for charts",
    )

    args = parser.parse_args()

    # Expand glob
    json_files = sorted(glob(args.results))
    if not json_files:
        print(f"❌ No files found matching: {args.results}")
        return

    print(f"📂 Loading {len(json_files)} JSON file(s)...")
    for f in json_files:
        print(f"   {f}")

    # Load results
    data = load_results(json_files)
    results = data["results"]

    if not results:
        print("❌ No results found")
        return

    print(f"✅ Loaded {len(results)} benchmark results")

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("\n📊 Generating charts...")
    plot_throughput_vs_concurrency(results, output_dir)
    plot_p99_latency_vs_concurrency(results, output_dir)
    plot_ttft_comparison(results, output_dir)

    # Generate markdown
    print("\n📝 Generating markdown table...")
    markdown = generate_markdown_table(results, output_dir)
    print("\n" + markdown)

    print(f"\n✅ All reports generated in: {output_dir}")


if __name__ == "__main__":
    main()
