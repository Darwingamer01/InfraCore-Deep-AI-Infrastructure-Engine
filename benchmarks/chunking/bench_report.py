"""Report generator for chunking benchmark results.

Loads JSON results from `chunking_bench` and writes a simple markdown summary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_get(run: dict, key: str, default=0.0):
    return run.get(key, default)


def _gather_metrics(data: dict) -> Tuple[List[str], List[int], List[float], List[float], List[float]]:
    names: List[str] = []
    num_chunks: List[int] = []
    avg_sizes: List[float] = []
    p50s: List[float] = []
    p95s: List[float] = []

    for run in data.get("runs", []):
        names.append(run.get("name", "unknown"))
        num_chunks.append(int(run.get("num_chunks", 0)))
        avg_sizes.append(float(run.get("avg_chunk_size", run.get("avg_size", 0.0))))

        # latency: prefer p50/p95 keys, else compute from latencies list
        if "p50_latency" in run and "p95_latency" in run:
            p50s.append(float(run.get("p50_latency", 0.0)))
            p95s.append(float(run.get("p95_latency", 0.0)))
        elif "latencies" in run and run.get("latencies"):
            lat = sorted(float(x) for x in run.get("latencies", []))
            n = len(lat)
            p50s.append(lat[int(n * 0.5)])
            p95s.append(lat[int(n * 0.95) if n > 1 else -1])
        else:
            # fallback to 0
            p50s.append(0.0)
            p95s.append(0.0)

    return names, num_chunks, avg_sizes, p50s, p95s


def _save_bar_chart(xs: List[str], ys: List[float], title: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 4.5), dpi=150)
    bars = plt.bar(xs, ys, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(xs)])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=10)
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def render_markdown(results_path: str, out_md: str | None = None) -> str:
    p = Path(results_path)
    data = json.loads(p.read_text())

    # Prepare output paths
    out_dir = p.parent
    _ensure_dir(out_dir)
    base = p.stem

    # Gather metrics
    names, num_chunks, avg_sizes, p50s, p95s = _gather_metrics(data)

    # Generate charts
    charts: List[Path] = []
    try:
        chunks_chart = out_dir / f"{base}_chunks.png"
        _save_bar_chart(names, [float(x) for x in num_chunks], "Total Chunks by Chunker", "Chunks", chunks_chart)
        charts.append(chunks_chart)

        size_chart = out_dir / f"{base}_avg_chunk_size.png"
        _save_bar_chart(names, avg_sizes, "Average Chunk Size (words)", "Avg words", size_chart)
        charts.append(size_chart)

        # Latency chart (grouped p50/p95)
        latency_chart = out_dir / f"{base}_latency.png"
        plt.figure(figsize=(8, 4.5), dpi=150)
        x = range(len(names))
        plt.plot(x, p50s, marker="o", label="p50")
        plt.plot(x, p95s, marker="o", label="p95")
        plt.xticks(x, names, rotation=10)
        plt.ylabel("Seconds")
        plt.title("Latency Comparison (p50 / p95)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(latency_chart)
        plt.close()
        charts.append(latency_chart)
    except Exception as e:
        # If matplotlib isn't available or plotting fails, continue gracefully
        charts = []

    lines = []
    lines.append(f"# Chunking Benchmark Report")
    lines.append(f"Generated from `{p.name}`")
    lines.append("")

    for run in data.get("runs", []):
        lines.append(f"## {run.get('name','unknown').title()} Chunker")
        lines.append(f"- Num chunks: {run.get('num_chunks', 0)}")
        lines.append(f"- Avg chunk size: {run.get('avg_chunk_size', 0.0):.2f} words")
        lines.append(f"- p50 latency: {run.get('p50_latency', 0.0):.4f}s")
        lines.append(f"- p95 latency: {run.get('p95_latency', 0.0):.4f}s")
        lines.append("")

    if charts:
        lines.append("## Charts")
        for c in charts:
            try:
                rel = c.relative_to(Path.cwd())
            except Exception:
                rel = c
            lines.append(f"- {rel}")
        lines.append("")

    md = "\n".join(lines)
    if out_md:
        Path(out_md).write_text(md)

    # Print generated chart paths
    if charts:
        print("Generated charts:")
        for c in charts:
            print(str(c))

    return md


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: bench_report.py <results.json> [out.md]")
        raise SystemExit(2)

    results_path = sys.argv[1]
    out_md = sys.argv[2] if len(sys.argv) > 2 else None
    print(render_markdown(results_path, out_md))
