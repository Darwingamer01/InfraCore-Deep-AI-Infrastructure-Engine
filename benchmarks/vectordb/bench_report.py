#!/usr/bin/env python
"""
INFRACORE — VectorDB Benchmark Report Generator

Purpose: Load JSON results from qdrant_bench runs, generate:
  1. A Recall@10 vs QPS scatter plot (the canonical tradeoff chart)
  2. A p99 Latency bar chart per config
  3. A markdown summary table
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_files: List[str]) -> Dict[str, Any]:
    """Load and merge JSON results from multiple benchmark files."""
    all_results = []

    for json_file in json_files:
        path = Path(json_file)
        if not path.exists():
            print(f"Warning: {json_file} not found")
            continue

        with open(path) as f:
            data = json.load(f)
            all_results.extend(data.get("results", []))

    return {"results": all_results}


def extract_config_info(config_name: str) -> tuple[int, int]:
    """Parse 'm8_ef64' → (m=8, ef=64)."""
    parts = config_name.lower().split("_")
    m = int(parts[0][1:])
    ef = int(parts[1][2:])
    return m, ef


def plot_recall_vs_qps(results: List[Dict], output_dir: Path) -> None:
    """Generate Recall@10 vs QPS scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping by m value
    color_map = {8: "blue", 16: "orange", 32: "green"}
    m_values = set()

    configs = []
    qps_list = []
    recall_list = []
    colors = []

    for result in results:
        config_name = result["config_name"]
        m, ef = extract_config_info(config_name)
        m_values.add(m)

        configs.append(config_name)
        qps_list.append(result["qps"])
        recall_list.append(result["recall_at_10"])
        colors.append(color_map.get(m, "black"))

    # Scatter plot
    for i, (qps, recall, color, config) in enumerate(
        zip(qps_list, recall_list, colors, configs)
    ):
        ax.scatter(qps, recall, s=200, c=color, alpha=0.7, edgecolors="black", linewidth=2)
        # Annotate with config name
        ax.annotate(
            config,
            (qps, recall),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    # Add QPS threshold line
    ax.axvline(x=500, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="500 QPS threshold")

    ax.set_xlabel("QPS (Queries Per Second)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_title("Qdrant HNSW: Recall@10 vs QPS Tradeoff", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_file = output_dir / "qdrant_recall_vs_qps.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_p99_latency(results: List[Dict], output_dir: Path) -> None:
    """Generate p99 latency bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    p99_list = []
    colors = []

    for result in results:
        config_name = result["config_name"]
        p99 = result["p99_latency_ms"]

        configs.append(config_name)
        p99_list.append(p99)

        # Color based on SLA thresholds
        if p99 < 10:
            colors.append("green")
        elif p99 < 20:
            colors.append("yellow")
        else:
            colors.append("red")

    bars = ax.bar(configs, p99_list, color=colors, alpha=0.7, edgecolor="black", linewidth=2)

    # Add 10ms SLA line
    ax.axhline(y=10, color="blue", linestyle="--", linewidth=2, alpha=0.7, label="10ms SLA")

    # Add value labels on bars
    for bar, p99 in zip(bars, p99_list):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{p99:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("p99 Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Qdrant HNSW: p99 Search Latency by Config", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")

    plt.tight_layout()
    output_file = output_dir / "qdrant_p99_latency.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def generate_markdown_table(results: List[Dict], output_dir: Path) -> str:
    """Generate markdown summary table."""
    lines = [
        "# Qdrant HNSW Benchmark Summary",
        "",
        "| Config | Dataset | QPS | Recall@10 | p50ms | p99ms | RAM Delta |",
        "|--------|---------|-----|-----------|-------|-------|-----------|",
    ]

    for result in results:
        config = result["config_name"]
        dataset_size = result["dataset_size"]
        dims = result["vector_dims"]
        qps = result["qps"]
        recall = result["recall_at_10"]
        p50 = result["p50_latency_ms"]
        p99 = result["p99_latency_ms"]
        ram_delta = result["ram_delta_mb"]

        dataset_str = f"{dataset_size//1000}K/{dims}D"
        ram_str = f"{ram_delta:+.0f} MB"

        line = f"| {config} | {dataset_str} | {qps:,.0f} | {recall:.3f} | {p50:.1f} | {p99:.1f} | {ram_str} |"
        lines.append(line)

    markdown = "\n".join(lines)

    # Save to file
    output_file = output_dir / "bench_summary.md"
    output_file.write_text(markdown)

    return markdown


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Qdrant Benchmark Report Generator")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Glob pattern for JSON result files (e.g., eval_reports/qdrant_bench_*.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="eval_reports",
        help="Output directory for charts",
    )

    args = parser.parse_args()

    # Expand glob pattern
    from glob import glob

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
        print("❌ No results found in JSON files")
        return

    print(f"✅ Loaded {len(results)} benchmark results")

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("\n📊 Generating charts...")
    plot_recall_vs_qps(results, output_dir)
    plot_p99_latency(results, output_dir)

    # Generate markdown table
    print("\n📝 Generating markdown table...")
    markdown = generate_markdown_table(results, output_dir)
    print("\n" + markdown)

    print(f"\n✅ All reports generated in: {output_dir}")


if __name__ == "__main__":
    main()
