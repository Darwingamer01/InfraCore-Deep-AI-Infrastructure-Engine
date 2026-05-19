"""
INFRACORE — RAGEvaluator

Purpose: Run a full evaluation suite over a dataset of EvalSamples
Generates markdown and JSON reports with per-metric aggregation
"""

import asyncio
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.infracore.eval.base import EvalConfig, EvalSample, EvalReport
from src.infracore.eval.metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
    AnswerCorrectnessMetric,
    MetricResult,
)


class EvalConfig(BaseModel):
    """Configuration for RAG evaluation."""

    model_config = ConfigDict(frozen=True)

    metrics: List[str] = Field(
        default=[
            "answer_relevance",
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_correctness",
        ],
        description="Metrics to compute",
    )
    output_dir: str = Field(default="eval_reports", description="Output directory for reports")
    report_format: Literal["json", "markdown"] = Field(
        default="markdown", description="Report format"
    )
    fail_threshold: float = Field(
        default=0.70, description="CI fails if any metric avg < this"
    )


@dataclass
class EvalReportData:
    """Enhanced eval report with full details."""

    samples_count: int
    per_metric_scores: dict  # metric_name → average score
    per_sample_results: List[List[MetricResult]] = field(default_factory=list)
    passed: bool = False
    timestamp: str = ""
    duration_ms: float = 0.0


class RAGEvaluator:
    """
    Run a full evaluation suite over EvalSamples.

    Computes all metrics, aggregates scores, generates reports.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.metrics_registry = {
            "answer_relevance": AnswerRelevanceMetric(),
            "context_recall": ContextRecallMetric(),
            "context_precision": ContextPrecisionMetric(),
            "faithfulness": FaithfulnessMetric(),
            "answer_correctness": AnswerCorrectnessMetric(),
        }

    async def evaluate(self, samples: List[EvalSample]) -> EvalReportData:
        """
        Evaluate a list of samples.

        Args:
            samples: List of EvalSample

        Returns:
            EvalReportData with aggregated metrics + per-sample scores
        """
        start_time = datetime.now()
        start_timestamp = start_time.isoformat()

        if not samples:
            return EvalReportData(
                samples_count=0,
                per_metric_scores={},
                per_sample_results=[],
                passed=True,
                timestamp=start_timestamp,
                duration_ms=0.0,
            )

        # Get active metrics
        active_metrics = [
            self.metrics_registry[name]
            for name in self.config.metrics
            if name in self.metrics_registry
        ]

        # Score each sample × each metric in parallel
        per_sample_results: List[List[MetricResult]] = []

        for sample in samples:
            # Run all metrics for this sample in parallel
            tasks = [metric.score(sample) for metric in active_metrics]
            results = await asyncio.gather(*tasks)
            per_sample_results.append(results)

        # Aggregate scores per metric
        per_metric_scores = {}
        for metric_name in self.config.metrics:
            scores = []
            for sample_results in per_sample_results:
                for result in sample_results:
                    if result.metric_name == metric_name:
                        scores.append(result.score)
                        break

            if scores:
                avg_score = sum(scores) / len(scores)
                per_metric_scores[metric_name] = avg_score

        # Check if passed
        passed = all(score >= self.config.fail_threshold for score in per_metric_scores.values())

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return EvalReportData(
            samples_count=len(samples),
            per_metric_scores=per_metric_scores,
            per_sample_results=per_sample_results,
            passed=passed,
            timestamp=start_timestamp,
            duration_ms=duration_ms,
        )

    async def save_report(
        self, report: EvalReportData, filename: str | None = None
    ) -> str:
        """
        Save report to file (markdown or JSON).

        Args:
            report: EvalReportData
            filename: Optional filename (defaults to eval_report_{timestamp}.{ext})

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = report.timestamp.replace(":", "-").replace(".", "-")
            ext = "md" if self.config.report_format == "markdown" else "json"
            filename = f"eval_report_{timestamp}.{ext}"

        filepath = output_dir / filename

        if self.config.report_format == "markdown":
            content = self._render_markdown(report)
        else:
            content = self._render_json(report)

        filepath.write_text(content)
        return str(filepath)

    def _render_markdown(self, report: EvalReportData) -> str:
        """Render report as markdown."""
        lines = [
            "# InfraCore RAG Evaluation Report",
            f"Generated: {report.timestamp}",
            "",
            "## Metrics Summary",
            "",
            "| Metric | Score | Status |",
            "|--------|-------|--------|",
        ]

        for metric_name, score in report.per_metric_scores.items():
            status = "✅ PASS" if score >= self.config.fail_threshold else "❌ FAIL"
            lines.append(f"| {metric_name} | {score:.3f} | {status} |")

        overall_status = "✅ PASSED" if report.passed else "❌ FAILED"
        lines.extend(
            [
                "",
                f"**Overall: {overall_status}**",
                "",
                f"Samples: {report.samples_count} | Duration: {report.duration_ms:.0f}ms",
                "",
            ]
        )

        return "\n".join(lines)

    def _render_json(self, report: EvalReportData) -> str:
        """Render report as JSON."""
        report_dict = {
            "samples_count": report.samples_count,
            "per_metric_scores": report.per_metric_scores,
            "passed": report.passed,
            "timestamp": report.timestamp,
            "duration_ms": report.duration_ms,
        }
        return json.dumps(report_dict, indent=2)

    def print_summary(self, report: EvalReportData) -> None:
        """Print colored terminal table."""
        print("\n" + "=" * 70)
        print("InfraCore RAG Evaluation Summary")
        print("=" * 70)

        # Table header
        print(f"{'Metric':<25} {'Score':<10} {'Status':<10}")
        print("-" * 70)

        # Table rows
        for metric_name, score in report.per_metric_scores.items():
            status = "✅ PASS" if score >= self.config.fail_threshold else "❌ FAIL"
            print(f"{metric_name:<25} {score:.3f}      {status:<10}")

        print("-" * 70)

        overall_status = "✅ PASSED" if report.passed else "❌ FAILED"
        print(f"Overall: {overall_status}")
        print(f"Samples: {report.samples_count} | Duration: {report.duration_ms:.0f}ms")
        print("=" * 70 + "\n")
