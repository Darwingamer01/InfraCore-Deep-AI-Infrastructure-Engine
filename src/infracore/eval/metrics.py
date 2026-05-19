"""
INFRACORE — Evaluation Metrics

Purpose: Core metric implementations for RAG pipeline evaluation
Philosophy: Each metric is a pure function + a class wrapper with Prometheus

All metrics are lexical/token-based (no LLM calls).
Implements: AnswerRelevance, ContextRecall, ContextPrecision, Faithfulness, AnswerCorrectness
"""

import re
import string
from dataclasses import dataclass
from typing import List, Set

from prometheus_client import Counter, Histogram

from src.infracore.eval.base import EvalSample


# Prometheus metrics
eval_metric_score = Histogram(
    "eval_metric_score",
    "Evaluation metric score",
    labelnames=["metric_name"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
eval_metric_runs_total = Counter(
    "eval_metric_runs_total", "Total evaluation metric runs", labelnames=["metric_name"]
)


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    metric_name: str
    score: float  # 0.0 to 1.0
    reasoning: str
    meta: dict


class AnswerRelevanceMetric:
    """
    Measures: does the answer address the question?
    Method: keyword overlap between question and answer
    """

    def __init__(self):
        self.name = "answer_relevance"
        self.stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "am",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "or",
            "but",
            "as",
            "by",
            "with",
            "from",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
        }

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text (lowercase, remove stopwords)."""
        # Lowercase and split on whitespace/punctuation
        tokens = re.findall(r"\b\w+\b", text.lower())
        # Filter stopwords and empty
        keywords = {t for t in tokens if t not in self.stopwords and len(t) > 0}
        return keywords

    async def score(self, sample: EvalSample) -> MetricResult:
        """Compute answer relevance score."""
        eval_metric_runs_total.labels(metric_name=self.name).inc()

        question_kw = self._extract_keywords(sample.query)
        answer_kw = self._extract_keywords(sample.answer)

        if not question_kw:
            # Question has no keywords after filtering -> perfect score
            score = 1.0
            reasoning = "Question has no keywords after stopword removal"
        else:
            overlap = len(question_kw & answer_kw)
            score = overlap / len(question_kw)
            reasoning = f"Keyword overlap: {overlap}/{len(question_kw)}"

        eval_metric_score.labels(metric_name=self.name).observe(score)

        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=reasoning,
            meta={"question_keywords": len(question_kw), "answer_keywords": len(answer_kw)},
        )


class ContextRecallMetric:
    """
    Measures: does the retrieved context contain the ground truth?
    Method: what fraction of ground_truth sentences appear in at least one context?
    """

    def __init__(self):
        self.name = "context_recall"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple split on ". " or "! " or "? "
        sentences = re.split(r"[.!?]\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    async def score(self, sample: EvalSample) -> MetricResult:
        """Compute context recall score."""
        eval_metric_runs_total.labels(metric_name=self.name).inc()

        gt_sentences = self._split_sentences(sample.ground_truth)
        contexts = sample.context if isinstance(sample.context, str) else " ".join(sample.context)

        if not gt_sentences:
            score = 1.0
            reasoning = "Ground truth has no sentences"
        else:
            matching = 0
            contexts_lower = contexts.lower()
            for sent in gt_sentences:
                if sent.lower() in contexts_lower:
                    matching += 1

            score = matching / len(gt_sentences)
            reasoning = f"Sentences found: {matching}/{len(gt_sentences)}"

        eval_metric_score.labels(metric_name=self.name).observe(score)

        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=reasoning,
            meta={"gt_sentences": len(gt_sentences), "matching_sentences": matching},
        )


class ContextPrecisionMetric:
    """
    Measures: what fraction of retrieved contexts are actually relevant to the question?
    Method: keyword overlap between question and each context
    """

    def __init__(self):
        self.name = "context_precision"
        self.stopwords = AnswerRelevanceMetric().stopwords

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        keywords = {t for t in tokens if t not in self.stopwords and len(t) > 0}
        return keywords

    async def score(self, sample: EvalSample) -> MetricResult:
        """Compute context precision score."""
        eval_metric_runs_total.labels(metric_name=self.name).inc()

        question_kw = self._extract_keywords(sample.query)

        # Handle both string and list contexts
        contexts = (
            [sample.context] if isinstance(sample.context, str) else sample.context
        )

        if not contexts or not question_kw:
            score = 1.0 if not contexts else 0.0
            reasoning = "No contexts or question keywords"
        else:
            relevant_count = 0
            for ctx in contexts:
                ctx_kw = self._extract_keywords(ctx)
                overlap = len(question_kw & ctx_kw)
                if overlap > 0:  # At least 1 keyword match
                    relevant_count += 1

            score = relevant_count / len(contexts)
            reasoning = f"Relevant contexts: {relevant_count}/{len(contexts)}"

        eval_metric_score.labels(metric_name=self.name).observe(score)

        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=reasoning,
            meta={"total_contexts": len(contexts), "relevant_contexts": relevant_count},
        )


class FaithfulnessMetric:
    """
    Measures: is every claim in the answer supported by the contexts?
    Method: sentence-level keyword entailment
    """

    def __init__(self):
        self.name = "faithfulness"
        self.stopwords = AnswerRelevanceMetric().stopwords

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        keywords = {t for t in tokens if t not in self.stopwords and len(t) > 0}
        return keywords

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    async def score(self, sample: EvalSample) -> MetricResult:
        """Compute faithfulness score."""
        eval_metric_runs_total.labels(metric_name=self.name).inc()

        answer_sentences = self._split_sentences(sample.answer)
        contexts = sample.context if isinstance(sample.context, str) else " ".join(sample.context)
        context_kw = self._extract_keywords(contexts)

        supported = 0
        if not answer_sentences:
            score = 1.0
            reasoning = "Answer has no sentences"
        else:
            for sent in answer_sentences:
                sent_kw = self._extract_keywords(sent)
                if not sent_kw:
                    # Sentence has no keywords -> consider it supported
                    supported += 1
                else:
                    # Check coverage: >50% of keywords in context
                    coverage = len(sent_kw & context_kw) / len(sent_kw)
                    if coverage >= 0.5:
                        supported += 1

            score = supported / len(answer_sentences)
            reasoning = f"Supported sentences: {supported}/{len(answer_sentences)}"

        eval_metric_score.labels(metric_name=self.name).observe(score)

        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=reasoning,
            meta={"total_sentences": len(answer_sentences), "supported_sentences": supported},
        )


class AnswerCorrectnessMetric:
    """
    Measures: how similar is the answer to the ground truth?
    Method: token-level F1 score (SQuAD evaluation)
    """

    def __init__(self):
        self.name = "answer_correctness"

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text: lowercase, split on whitespace and punctuation."""
        # Remove punctuation and split
        text_clean = text.lower()
        tokens = re.findall(r"\b\w+\b", text_clean)
        return set(tokens)

    async def score(self, sample: EvalSample) -> MetricResult:
        """Compute answer correctness (F1) score."""
        eval_metric_runs_total.labels(metric_name=self.name).inc()

        answer_tokens = self._tokenize(sample.answer)
        gt_tokens = self._tokenize(sample.ground_truth)

        if not answer_tokens and not gt_tokens:
            score = 1.0
            reasoning = "Both empty"
        elif not answer_tokens or not gt_tokens:
            score = 0.0
            reasoning = "Answer or ground truth is empty"
        else:
            common = answer_tokens & gt_tokens
            if len(common) == 0:
                score = 0.0
                reasoning = "No token overlap"
            else:
                precision = len(common) / len(answer_tokens)
                recall = len(common) / len(gt_tokens)
                score = 2 * precision * recall / (precision + recall)
                reasoning = f"F1 score: {score:.3f} (P={precision:.3f}, R={recall:.3f})"

        eval_metric_score.labels(metric_name=self.name).observe(score)

        return MetricResult(
            metric_name=self.name,
            score=score,
            reasoning=reasoning,
            meta={
                "answer_tokens": len(answer_tokens),
                "gt_tokens": len(gt_tokens),
                "common_tokens": len(common) if answer_tokens and gt_tokens else 0,
            },
        )
