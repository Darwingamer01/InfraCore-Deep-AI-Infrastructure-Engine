#!/usr/bin/env python
"""
INFRACORE — Comparative VectorDB Report Generator

Purpose: Load Qdrant and pgvector benchmark results and generate:
  1. Recall@10 vs QPS scatter plot with both backends
  2. p99 Latency comparison
  3. RAM delta comparison
  4. Markdown summary table with analysis
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(qdrant_file: Optional[str], pgvector_file: Optional[str]) -> Dict[str, Any]:
    """Load and merge Qdrant and pgvector benchmark results."""
    results = {"qdrant": [], "pgvector": []}

    if qdrant_file:
        path = Path(qdrant_file)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                results["qdrant"].extend(data.get("results", []))
                print(f"✅ Loaded {len(data.get('results', []))} Qdrant result(s)")
        else:
            print(f"⚠️  Qdrant file not found: {qdrant_file}")

    if pgvector_file:
        path = Path(pgvector_file)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                results["pgvector"].extend(data.get("results", []))
                print(f"✅ Loaded {len(data.get('results', []))} pgvector result(s)")
        else:
            print(f"⚠️  pgvector file not found: {pgvector_file}")

    return results


def plot_recall_vs_qps_comparative(
    results: Dict[str, List[Dict]], output_dir: Path
) -> None:
    """Generate Recall@10 vs QPS scatter plot with both backends."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Qdrant results
    qdrant_results = results.get("qdrant", [])
    if qdrant_results:
        qps_qdrant = [r["qps"] for r in qdrant_results]
        recall_qdrant = [r["recall_at_10"] for r in qdrant_results]
        ax.scatter(
            qps_qdrant,
            recall_qdrant,
            s=300,
            c="blue",
            alpha=0.7,
            edgecolors="darkblue",
            linewidth=2.5,
            marker="o",
            label="Qdrant",
        )

        # Annotate Qdrant points
        for i, (qps, recall) in enumerate(zip(qps_qdrant, recall_qdrant)):
            ax.annotate(
                f"Q{i+1}",
                (qps, recall),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="darkblue",
            )

    # pgvector results
    pgvector_results = results.get("pgvector", [])
    if pgvector_results:
        qps_pgvector = [r["qps"] for r in pgvector_results]
        recall_pgvector = [r["recall_at_10"] for r in pgvector_results]
        ax.scatter(
            qps_pgvector,
            recall_pgvector,
            s=300,
            c="orange",
            alpha=0.7,
            edgecolors="darkorange",
            linewidth=2.5,
            marker="s",
            label="pgvector",
        )

        # Annotate pgvector points
        for i, (qps, recall) in enumerate(zip(qps_pgvector, recall_pgvector)):
            ax.annotate(
                f"P{i+1}",
                (qps, recall),
                xytext=(8, -12),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="darkorange",
            )

    # Add QPS threshold line
    ax.axvline(
        x=500,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="500 QPS threshold",
    )

    ax.set_xlabel("QPS (Queries Per Second)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_title("VectorDB Comparison: Recall@10 vs QPS", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_file = output_dir / "vectordb_recall_vs_qps.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_p99_latency_comparison(results: Dict[str, List[Dict]], output_dir: Path) -> None:
    """Generate p99 latency comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_configs = []
    all_latencies = []
    all_colors = []

    # Qdrant results
    qdrant_results = results.get("qdrant", [])
    for i, r in enumerate(qdrant_results):
        all_configs.append(f"Qdrant\n{r['config_name']}")
        all_latencies.append(r["p99_latency_ms"])
        all_colors.append("blue")

    # pgvector results
    pgvector_results = results.get("pgvector", [])
    for i, r in enumerate(pgvector_results):
        all_configs.append(f"pgvector\n{r['config_name']}")
        all_latencies.append(r["p99_latency_ms"])
        all_colors.append("orange")

    x_pos = np.arange(len(all_configs))
    bars = ax.bar(x_pos, all_latencies, color=all_colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for i, (bar, latency) in enumerate(zip(bars, all_latencies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{latency:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add SLA line (100ms)
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="100ms SLA")

    ax.set_ylabel("p99 Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("p99 Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_configs, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    output_file = output_dir / "vectordb_p99_latency_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def plot_ram_delta_comparison(results: Dict[str, List[Dict]], output_dir: Path) -> None:
    """Generate RAM delta comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_configs = []
    all_ram_deltas = []
    all_colors = []

    # Qdrant results
    qdrant_results = results.get("qdrant", [])
    for r in qdrant_results:
        all_configs.append(f"Qdrant\n{r['config_name']}")
        all_ram_deltas.append(r["ram_delta_mb"])
        all_colors.append("blue")

    # pgvector results
    pgvector_results = results.get("pgvector", [])
    for r in pgvector_results:
        all_configs.append(f"pgvector\n{r['config_name']}")
        all_ram_deltas.append(r["ram_delta_mb"])
        all_colors.append("orange")

    x_pos = np.arange(len(all_configs))
    bars = ax.bar(x_pos, all_ram_deltas, color=all_colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, ram_delta in zip(bars, all_ram_deltas):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{ram_delta:+.0f}MB",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("RAM Delta (MB)", fontsize=12, fontweight="bold")
    ax.set_title("RAM Usage Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_configs, fontsize=9)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "vectordb_ram_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✅ Chart saved: {output_file}")
    plt.close()


def generate_markdown_summary(results: Dict[str, List[Dict]], output_dir: Path) -> None:
    """Generate markdown summary table and analysis."""
    output_file = output_dir / "vectordb_comparison_summary.md"

    with open(output_file, "w") as f:
        f.write("# Vector Database Comparison Report\n\n")

        f.write("## Results Summary\n\n")

        # Combined table
        f.write("| Backend | Config | Dataset Size | QPS | Recall@10 | p99 Latency (ms) | RAM Delta (MB) | Upsert Time (s) |\n")
        f.write("|---------|--------|--------------|-----|-----------|-----------------|----------------|----------------|\n")

        qdrant_results = results.get("qdrant", [])
        for r in qdrant_results:
            f.write(
                f"| Qdrant | {r['config_name']} | {r['dataset_size']:,} | "
                f"{r['qps']:,.0f} | {r['recall_at_10']:.3f} | {r['p99_latency_ms']:.1f} | "
                f"{r['ram_delta_mb']:+.0f} | {r['upsert_time_s']:.2f} |\n"
            )

        pgvector_results = results.get("pgvector", [])
        for r in pgvector_results:
            f.write(
                f"| pgvector | {r['config_name']} | {r['dataset_size']:,} | "
                f"{r['qps']:,.0f} | {r['recall_at_10']:.3f} | {r['p99_latency_ms']:.1f} | "
                f"{r['ram_delta_mb']:+.0f} | {r['upsert_time_s']:.2f} |\n"
            )

        f.write("\n## Key Findings\n\n")

        # Find best in each metric
        all_results = qdrant_results + pgvector_results
        if all_results:
            best_qps = max(all_results, key=lambda r: r["qps"])
            best_latency = min(all_results, key=lambda r: r["p99_latency_ms"])
            best_recall = max(all_results, key=lambda r: r["recall_at_10"])
            best_ram = min(all_results, key=lambda r: r["ram_delta_mb"])

            backend_qps = "Qdrant" if best_qps in qdrant_results else "pgvector"
            backend_latency = "Qdrant" if best_latency in qdrant_results else "pgvector"
            backend_recall = "Qdrant" if best_recall in qdrant_results else "pgvector"
            backend_ram = "Qdrant" if best_ram in qdrant_results else "pgvector"

            f.write(f"**Best QPS**: {backend_qps} ({best_qps['qps']:,.0f})\n\n")
            f.write(f"**Best Latency**: {backend_latency} ({best_latency['p99_latency_ms']:.1f}ms p99)\n\n")
            f.write(f"**Best Recall**: {backend_recall} ({best_recall['recall_at_10']:.3f})\n\n")
            f.write(f"**Best RAM Efficiency**: {backend_ram} ({best_ram['ram_delta_mb']:+.0f}MB)\n\n")

        f.write("## Analysis\n\n")

        if qdrant_results and pgvector_results:
            qdrant_r = qdrant_results[0]
            pgvector_r = pgvector_results[0]

            qps_delta_pct = (
                ((pgvector_r["qps"] - qdrant_r["qps"]) / qdrant_r["qps"] * 100)
                if qdrant_r["qps"] > 0
                else 0
            )
            latency_delta_pct = (
                ((pgvector_r["p99_latency_ms"] - qdrant_r["p99_latency_ms"]) / qdrant_r["p99_latency_ms"] * 100)
                if qdrant_r["p99_latency_ms"] > 0
                else 0
            )

            f.write("### Throughput\n\n")
            f.write(
                f"Qdrant: {qdrant_r['qps']:,.0f} QPS | "
                f"pgvector: {pgvector_r['qps']:,.0f} QPS\n\n"
            )
            if abs(qps_delta_pct) > 5:
                winner = "Qdrant" if qps_delta_pct > 0 else "pgvector"
                delta = abs(qps_delta_pct)
                f.write(f"**{winner} is {delta:.1f}% faster**\n\n")

            f.write("### Latency\n\n")
            f.write(
                f"Qdrant: {qdrant_r['p99_latency_ms']:.1f}ms p99 | "
                f"pgvector: {pgvector_r['p99_latency_ms']:.1f}ms p99\n\n"
            )
            if abs(latency_delta_pct) > 5:
                winner = "Qdrant" if latency_delta_pct > 0 else "pgvector"
                delta = abs(latency_delta_pct)
                f.write(f"**{winner} is {delta:.1f}% faster**\n\n")

            f.write("### RAM Usage\n\n")
            f.write(
                f"Qdrant: {qdrant_r['ram_delta_mb']:+.0f}MB | "
                f"pgvector: {pgvector_r['ram_delta_mb']:+.0f}MB\n\n"
            )

            f.write("### Summary\n\n")
            f.write("- **Qdrant** excels at pure vector search performance (throughput, latency)\n")
            f.write("- **pgvector** offers tighter Postgres integration for operational simplicity\n")
            f.write("- Choose **Qdrant** for high-throughput, latency-sensitive workloads\n")
            f.write("- Choose **pgvector** for mixed SQL+vector queries or existing Postgres infrastructure\n")

    print(f"✅ Markdown saved: {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="VectorDB Comparative Report Generator")
    parser.add_argument("--qdrant", type=str, help="Path to Qdrant benchmark JSON")
    parser.add_argument("--pgvector", type=str, help="Path to pgvector benchmark JSON")
    parser.add_argument("--output", type=str, default="eval_reports", help="Output directory")

    args = parser.parse_args()

    if not args.qdrant and not args.pgvector:
        print("❌ Please provide at least one benchmark result via --qdrant or --pgvector")
        parser.print_help()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Loading benchmark results...")
    results = load_results(args.qdrant, args.pgvector)

    if not any(results.values()):
        print("❌ No benchmark results loaded")
        return

    print("\n📈 Generating charts...")
    plot_recall_vs_qps_comparative(results, output_dir)
    plot_p99_latency_comparison(results, output_dir)
    plot_ram_delta_comparison(results, output_dir)

    print("\n📝 Generating markdown summary...")
    generate_markdown_summary(results, output_dir)

    print(f"\n✅ All reports generated in: {output_dir}")


if __name__ == "__main__":
    main()
