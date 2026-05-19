#!/usr/bin/env python
"""
INFRACORE — Retrieval Benchmark Report Generator

Purpose: Load JSON results from retrieval benchmarks, generate charts and summary
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_file: str) -> Dict[str, Any]:
    """Load JSON results from retrieval benchmark."""
    with open(json_file) as f:
        return json.load(f)


def plot_accuracy_metrics(results: List[Dict], output_dir: Path) -> None:
    """Generate bar chart comparing accuracy metrics (MRR@10, Recall@10, nDCG@10)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strategies = [r["strategy"] for r in results]
    recalls = [r["recall_at_10"] for r in results]
    mrrs = [r["mrr_at_10"] for r in results]
    ndcgs = [r["ndcg_at_10"] for r in results]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Recall@10
    axes[0].bar(strategies, recalls, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[0].set_ylabel("Recall@10", fontweight="bold")
    axes[0].set_title("Recall@10: Coverage of Relevant Docs", fontweight="bold")
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # MRR@10
    axes[1].bar(strategies, mrrs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[1].set_ylabel("MRR@10", fontweight="bold")
    axes[1].set_title("MRR@10: Ranking Quality", fontweight="bold")
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mrrs):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # nDCG@10
    axes[2].bar(strategies, ndcgs, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    axes[2].set_ylabel("nDCG@10", fontweight="bold")
    axes[2].set_title("nDCG@10: Ranking Order Quality", fontweight="bold")
    axes[2].set_ylim([0, 1.0])
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(ndcgs):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # Rotate x labels
    for ax in axes:
        ax.set_xticklabels(strategies, rotation=45, ha="right")

    plt.tight_layout()
    output_file = output_dir / "retrieval_accuracy_metrics.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_latency_comparison(results: List[Dict], output_dir: Path) -> None:
    """Generate latency bar chart (p50, p95, p99)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = [r["strategy"] for r in results]
    p50 = [r["p50_latency_ms"] for r in results]
    p95 = [r["p95_latency_ms"] for r in results]
    p99 = [r["p99_latency_ms"] for r in results]

    x = np.arange(len(strategies))
    width = 0.25

    bars1 = ax.bar(x - width, p50, width, label="p50", alpha=0.8, edgecolor="black", linewidth=1)
    bars2 = ax.bar(x, p95, width, label="p95", alpha=0.8, edgecolor="black", linewidth=1)
    bars3 = ax.bar(x + width, p99, width, label="p99", alpha=0.8, edgecolor="black", linewidth=1)

    # SLA line at 100ms
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="100ms SLA")

    ax.set_xlabel("Strategy", fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_title("Query Latency: p50/p95/p99 Percentiles", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "retrieval_latency_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_accuracy_vs_latency(results: List[Dict], output_dir: Path) -> None:
    """Generate scatter plot: Accuracy (MRR@10) vs Latency (p99)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = [r["strategy"] for r in results]
    mrrs = [r["mrr_at_10"] for r in results]
    p99s = [r["p99_latency_ms"] for r in results]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Scatter
    scatter = ax.scatter(p99s, mrrs, s=200, c=colors, alpha=0.7, edgecolors="black", linewidth=2)

    # Annotate
    for i, (x, y, strategy) in enumerate(zip(p99s, mrrs, strategies)):
        ax.annotate(
            strategy,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("p99 Latency (ms)", fontweight="bold")
    ax.set_ylabel("MRR@10", fontweight="bold")
    ax.set_title("Accuracy vs Latency Tradeoff", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "retrieval_accuracy_vs_latency.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def generate_markdown_summary(results: List[Dict], output_dir: Path) -> str:
    """Generate markdown summary table."""
    lines = [
        "# Retrieval Strategy Benchmark Summary",
        "",
        "| Strategy | Recall@10 | MRR@10 | nDCG@10 | p50 (ms) | p95 (ms) | p99 (ms) |",
        "|----------|-----------|--------|---------|----------|----------|----------|",
    ]

    for result in results:
        strategy = result["strategy"].upper()
        recall = result["recall_at_10"]
        mrr = result["mrr_at_10"]
        ndcg = result["ndcg_at_10"]
        p50 = result["p50_latency_ms"]
        p95 = result["p95_latency_ms"]
        p99 = result["p99_latency_ms"]

        line = f"| {strategy} | {recall:.3f} | {mrr:.3f} | {ndcg:.3f} | {p50:.1f} | {p95:.1f} | {p99:.1f} |"
        lines.append(line)

    markdown = "\n".join(lines)

    # Save to file
    output_file = output_dir / "retrieval_summary.md"
    output_file.write_text(markdown)

    return markdown


def generate_analysis(results: List[Dict]) -> str:
    """Generate narrative analysis of results."""
    lines = [
        "## Key Findings",
        "",
    ]

    # Find best in each metric
    best_recall = max(results, key=lambda x: x["recall_at_10"])
    best_mrr = max(results, key=lambda x: x["mrr_at_10"])
    best_latency = min(results, key=lambda x: x["p99_latency_ms"])

    lines.append(f"**Best Recall**: {best_recall['strategy'].upper()} ({best_recall['recall_at_10']:.3f})")
    lines.append(f"**Best MRR**: {best_mrr['strategy'].upper()} ({best_mrr['mrr_at_10']:.3f})")
    lines.append(f"**Best Latency**: {best_latency['strategy'].upper()} ({best_latency['p99_latency_ms']:.1f} ms p99)")
    lines.append("")

    # Reranker impact
    hybrid = next((r for r in results if r["strategy"] == "hybrid_rrf"), None)
    hybrid_ce = next((r for r in results if r["strategy"] == "hybrid_reranker_ce"), None)
    hybrid_colbert = next((r for r in results if r["strategy"] == "hybrid_reranker_colbert"), None)

    if hybrid and hybrid_ce:
        mrr_gain = (hybrid_ce["mrr_at_10"] - hybrid["mrr_at_10"]) / hybrid["mrr_at_10"] * 100
        lines.append(f"**CrossEncoder Impact**: +{mrr_gain:.1f}% MRR over hybrid baseline")

    if hybrid and hybrid_colbert:
        mrr_gain = (hybrid_colbert["mrr_at_10"] - hybrid["mrr_at_10"]) / hybrid["mrr_at_10"] * 100
        lines.append(f"**ColBERT-lite Impact**: +{mrr_gain:.1f}% MRR over hybrid baseline")

    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval Benchmark Report Generator")
    parser.add_argument("--results", type=str, required=True, help="JSON result file")
    parser.add_argument("--out", type=str, default="eval_reports", help="Output directory")

    args = parser.parse_args()

    # Load results
    data = load_results(args.results)
    results = data["results"]

    if not results:
        print("❌ No results found")
        return

    print(f"✅ Loaded {len(results)} strategy results")

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("\n📊 Generating charts...")
    plot_accuracy_metrics(results, output_dir)
    plot_latency_comparison(results, output_dir)
    plot_accuracy_vs_latency(results, output_dir)

    # Generate markdown
    print("\n📝 Generating markdown summary...")
    markdown = generate_markdown_summary(results, output_dir)
    print("\n" + markdown)

    # Generate analysis
    print("\n" + generate_analysis(results))

    print(f"\n✅ All reports generated in: {output_dir}")


if __name__ == "__main__":
    main()
