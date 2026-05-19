"""
INFRACORE TEST — Evaluation Metrics + RAGEvaluator

Complete test coverage for all 5 metrics and the RAGEvaluator.
"""

import pytest
from pathlib import Path

from src.infracore.eval.base import EvalSample
from src.infracore.eval.metrics import (
    AnswerRelevanceMetric,
    ContextRecallMetric,
    ContextPrecisionMetric,
    FaithfulnessMetric,
    AnswerCorrectnessMetric,
)
from src.infracore.eval.evaluator import EvalConfig, RAGEvaluator


# ========== Fixtures ==========


@pytest.fixture
def perfect_sample():
    """Sample with perfect overlap between answer and ground truth."""
    gt = "Machine learning involves algorithms learning from data."
    return EvalSample(
        query="What is machine learning?",
        context=f"Machine learning is an important field. {gt} Supervised learning is one approach.",
        answer="Machine learning involves algorithms learning from data.",
        ground_truth=gt,
    )


@pytest.fixture
def zero_sample():
    """Sample with zero overlap (hallucination test)."""
    return EvalSample(
        query="What is Python?",
        context="The Eiffel Tower is located in Paris. It was built in 1889.",
        answer="Python is a type of snake found in tropical regions.",
        ground_truth="Python is a high-level programming language.",
    )


@pytest.fixture
def partial_sample():
    """Sample with partial overlap."""
    return EvalSample(
        query="What is machine learning?",
        context="Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        answer="Machine learning is a field of AI.",
        ground_truth="Machine learning is a subset of artificial intelligence where algorithms learn from data.",
    )


@pytest.fixture
def stopword_only_query():
    """Query with only stopwords (edge case)."""
    return EvalSample(
        query="the is and for",
        context="Some content here.",
        answer="Some answer here.",
        ground_truth="Some truth here.",
    )


@pytest.fixture
def empty_sample():
    """Sample with empty fields."""
    return EvalSample(
        query="",
        context="",
        answer="",
        ground_truth="",
    )


# ========== AnswerRelevanceMetric Tests ==========


@pytest.mark.asyncio
async def test_answer_relevance_perfect(perfect_sample):
    """Perfect sample should have high overlap."""
    metric = AnswerRelevanceMetric()
    result = await metric.score(perfect_sample)

    assert result.metric_name == "answer_relevance"
    assert result.score >= 0.5  # Should have significant overlap
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_answer_relevance_zero(zero_sample):
    """Low keyword overlap score."""
    metric = AnswerRelevanceMetric()
    result = await metric.score(zero_sample)

    assert result.metric_name == "answer_relevance"
    # "Python" appears in both, so not truly zero
    assert result.score <= 0.5


@pytest.mark.asyncio
async def test_answer_relevance_stopwords_only(stopword_only_query):
    """Query with only stopwords should score 1.0."""
    metric = AnswerRelevanceMetric()
    result = await metric.score(stopword_only_query)

    assert result.score == 1.0  # All stopwords filtered -> perfect score


@pytest.mark.asyncio
async def test_answer_relevance_partial(partial_sample):
    """Partial overlap should be between 0 and 1."""
    metric = AnswerRelevanceMetric()
    result = await metric.score(partial_sample)

    assert 0.0 < result.score < 1.0


# ========== ContextRecallMetric Tests ==========


@pytest.mark.asyncio
async def test_context_recall_perfect(perfect_sample):
    """Ground truth fully contained in context."""
    metric = ContextRecallMetric()
    result = await metric.score(perfect_sample)

    assert result.metric_name == "context_recall"
    # Ground truth appears in context (substring match)
    assert result.score > 0.5


@pytest.mark.asyncio
async def test_context_recall_zero(zero_sample):
    """Ground truth not in context."""
    metric = ContextRecallMetric()
    result = await metric.score(zero_sample)

    assert result.score == 0.0


@pytest.mark.asyncio
async def test_context_recall_partial(partial_sample):
    """Partial overlap."""
    metric = ContextRecallMetric()
    result = await metric.score(partial_sample)

    # Some sentences match, so 0 < score <= 1
    assert 0.0 <= result.score <= 1.0


# ========== ContextPrecisionMetric Tests ==========


@pytest.mark.asyncio
async def test_context_precision_relevant_context(perfect_sample):
    """Context relevant to question."""
    metric = ContextPrecisionMetric()
    result = await metric.score(perfect_sample)

    assert result.metric_name == "context_precision"
    assert result.score > 0.0  # Some relevance


@pytest.mark.asyncio
async def test_context_precision_irrelevant_context(zero_sample):
    """Context irrelevant to question."""
    metric = ContextPrecisionMetric()
    result = await metric.score(zero_sample)

    assert result.score == 0.0  # Eiffel Tower not about Python


@pytest.mark.asyncio
async def test_context_precision_with_list_contexts():
    """Test with multiple contexts as list."""
    sample = EvalSample(
        query="What is machine learning?",
        context=[
            "Machine learning is artificial intelligence.",
            "The weather is sunny today.",
            "Neural networks are used in deep learning.",
        ],
        answer="Machine learning is AI.",
        ground_truth="Machine learning is a field of AI.",
    )
    metric = ContextPrecisionMetric()
    result = await metric.score(sample)

    # 2 out of 3 contexts relevant (ML-related)
    assert 0.0 < result.score <= 1.0


# ========== FaithfulnessMetric Tests ==========


@pytest.mark.asyncio
async def test_faithfulness_perfect(perfect_sample):
    """Answer fully grounded in context."""
    metric = FaithfulnessMetric()
    result = await metric.score(perfect_sample)

    assert result.metric_name == "faithfulness"
    assert result.score >= 0.5  # Well grounded


@pytest.mark.asyncio
async def test_faithfulness_hallucination(zero_sample):
    """Hallucinated answer not grounded in context."""
    metric = FaithfulnessMetric()
    result = await metric.score(zero_sample)

    assert result.score <= 0.2  # Not supported by context


@pytest.mark.asyncio
async def test_faithfulness_empty_answer(empty_sample):
    """Empty answer should get perfect score."""
    metric = FaithfulnessMetric()
    result = await metric.score(empty_sample)

    assert result.score == 1.0
    assert result.reasoning == "Answer has no sentences"


# ========== AnswerCorrectnessMetric Tests ==========


@pytest.mark.asyncio
async def test_answer_correctness_perfect_match(perfect_sample):
    """Perfect token match."""
    metric = AnswerCorrectnessMetric()
    result = await metric.score(perfect_sample)

    assert result.metric_name == "answer_correctness"
    assert result.score == 1.0  # Perfect F1 (same text)


@pytest.mark.asyncio
async def test_answer_correctness_zero_overlap(zero_sample):
    """Low token overlap (zero_sample has some common words)."""
    metric = AnswerCorrectnessMetric()
    result = await metric.score(zero_sample)

    # "Python" appears in both query and answer, so not truly zero F1
    assert result.score < 0.5


@pytest.mark.asyncio
async def test_answer_correctness_partial_overlap(partial_sample):
    """Partial token overlap."""
    metric = AnswerCorrectnessMetric()
    result = await metric.score(partial_sample)

    assert 0.0 < result.score < 1.0


@pytest.mark.asyncio
async def test_answer_correctness_empty_fields():
    """Empty answer and ground truth."""
    sample = EvalSample(
        query="test",
        context="test",
        answer="",
        ground_truth="",
    )
    metric = AnswerCorrectnessMetric()
    result = await metric.score(sample)

    assert result.score == 1.0  # Both empty


# ========== RAGEvaluator Tests ==========


@pytest.mark.asyncio
async def test_rag_evaluator_single_perfect_sample(perfect_sample):
    """Evaluate single perfect sample."""
    config = EvalConfig(
        metrics=["answer_relevance", "faithfulness", "answer_correctness"],
        output_dir="eval_reports",
        report_format="markdown",
        fail_threshold=0.7,
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    assert report.samples_count == 1
    assert len(report.per_metric_scores) == 3
    # All metrics should have scores >= 0.5 for perfect sample
    assert all(score >= 0.5 for score in report.per_metric_scores.values())


@pytest.mark.asyncio
async def test_rag_evaluator_single_bad_sample(zero_sample):
    """Evaluate single bad (hallucinated) sample."""
    config = EvalConfig(
        metrics=["answer_relevance", "faithfulness"],
        output_dir="eval_reports",
        report_format="markdown",
        fail_threshold=0.7,
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([zero_sample])

    assert report.samples_count == 1
    assert report.passed is False  # Should fail (scores low)
    assert len(report.per_metric_scores) == 2


@pytest.mark.asyncio
async def test_rag_evaluator_empty_samples():
    """Evaluate with empty sample list."""
    config = EvalConfig(output_dir="eval_reports")
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([])

    assert report.samples_count == 0
    assert len(report.per_metric_scores) == 0
    assert report.passed is True


@pytest.mark.asyncio
async def test_rag_evaluator_multiple_samples(perfect_sample, partial_sample, zero_sample):
    """Evaluate multiple samples."""
    config = EvalConfig(
        metrics=["answer_relevance", "context_recall"],
        fail_threshold=0.5,
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample, partial_sample, zero_sample])

    assert report.samples_count == 3
    assert len(report.per_metric_scores) == 2
    # Each metric should have average score
    for score in report.per_metric_scores.values():
        assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_rag_evaluator_per_metric_scores_has_5_keys(perfect_sample):
    """Per-metric scores should have all 5 metrics."""
    config = EvalConfig(
        metrics=[
            "answer_relevance",
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_correctness",
        ],
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    assert len(report.per_metric_scores) == 5


@pytest.mark.asyncio
async def test_rag_evaluator_save_report_markdown(perfect_sample, tmp_path):
    """Save markdown report."""
    config = EvalConfig(
        metrics=["answer_relevance"],
        output_dir=str(tmp_path),
        report_format="markdown",
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    filepath = await evaluator.save_report(report)

    assert Path(filepath).exists()
    assert filepath.endswith(".md")
    content = Path(filepath).read_text()
    assert "InfraCore RAG Evaluation Report" in content
    assert "answer_relevance" in content


@pytest.mark.asyncio
async def test_rag_evaluator_save_report_json(perfect_sample, tmp_path):
    """Save JSON report."""
    config = EvalConfig(
        metrics=["answer_relevance"],
        output_dir=str(tmp_path),
        report_format="json",
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    filepath = await evaluator.save_report(report)

    assert Path(filepath).exists()
    assert filepath.endswith(".json")
    content = Path(filepath).read_text()
    assert "per_metric_scores" in content


@pytest.mark.asyncio
async def test_rag_evaluator_print_summary(perfect_sample, capsys):
    """Print summary to stdout."""
    config = EvalConfig(metrics=["answer_relevance"])
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    evaluator.print_summary(report)

    captured = capsys.readouterr()
    assert "InfraCore RAG Evaluation Summary" in captured.out
    assert "answer_relevance" in captured.out
    assert "PASSED" in captured.out or "FAILED" in captured.out


@pytest.mark.asyncio
async def test_rag_evaluator_timestamp_format(perfect_sample):
    """Report should have valid timestamp."""
    config = EvalConfig()
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate([perfect_sample])

    assert report.timestamp != ""
    # ISO format check
    assert "T" in report.timestamp
    assert report.duration_ms >= 0


# ========== Integration Tests ==========


@pytest.mark.asyncio
async def test_full_eval_pipeline():
    """Full evaluation pipeline integration test."""
    samples = [
        EvalSample(
            query="What is AI?",
            context="Artificial intelligence is a field of computer science.",
            answer="AI is a field of computer science.",
            ground_truth="AI is artificial intelligence.",
        ),
        EvalSample(
            query="What is ML?",
            context="Machine learning is a subset of AI.",
            answer="Machine learning is a subset of AI.",
            ground_truth="Machine learning is a subset of artificial intelligence.",
        ),
    ]

    config = EvalConfig(
        metrics=[
            "answer_relevance",
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_correctness",
        ],
        fail_threshold=0.5,
    )
    evaluator = RAGEvaluator(config)
    report = await evaluator.evaluate(samples)

    assert report.samples_count == 2
    assert len(report.per_metric_scores) == 5
    assert all(0.0 <= score <= 1.0 for score in report.per_metric_scores.values())
    evaluator.print_summary(report)
