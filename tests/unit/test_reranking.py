#!/usr/bin/env python
"""
Unit tests for retrieval reranking modules.
"""

import pytest
import numpy as np

from infracore.retrieval.base import RetrievalResult
from infracore.retrieval.cross_encoder_reranker import CrossEncoderReranker, RerankedResult
from infracore.retrieval.colbert_lite import ColBERTLiteReranker, MockReranker


@pytest.fixture
def sample_candidates():
    """Sample retrieval results for testing."""
    return [
        RetrievalResult(
            text="Machine learning is a subset of artificial intelligence",
            score=0.9,
            metadata={"doc_id": "doc1", "source": "test"},
        ),
        RetrievalResult(
            text="Deep learning uses neural networks with many layers",
            score=0.8,
            metadata={"doc_id": "doc2", "source": "test"},
        ),
        RetrievalResult(
            text="Computer science is the study of computation",
            score=0.7,
            metadata={"doc_id": "doc3", "source": "test"},
        ),
    ]


@pytest.mark.asyncio
async def test_cross_encoder_reranker_basic(sample_candidates):
    """Test CrossEncoderReranker basic reranking."""
    reranker = CrossEncoderReranker()
    query = "What is machine learning?"

    results = await reranker.rerank(query, sample_candidates, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RerankedResult) for r in results)
    # Check that results are sorted by score descending
    assert results[0].rerank_score >= results[1].rerank_score


@pytest.mark.asyncio
async def test_cross_encoder_reranker_score_single():
    """Test CrossEncoderReranker single scoring."""
    reranker = CrossEncoderReranker()

    score1 = reranker.score_single("machine learning", "Machine learning is AI")
    score2 = reranker.score_single("machine learning", "Random unrelated text")

    # Relevant doc should have higher score
    assert score1 > score2
    assert 0 <= score1 <= 1
    assert 0 <= score2 <= 1


@pytest.mark.asyncio
async def test_cross_encoder_reranker_no_candidates(sample_candidates):
    """Test reranker with empty candidate list."""
    reranker = CrossEncoderReranker()
    results = await reranker.rerank("query", [])

    assert results == []


@pytest.mark.asyncio
async def test_cross_encoder_reranker_top_k(sample_candidates):
    """Test top_k truncation."""
    reranker = CrossEncoderReranker()
    query = "machine learning"

    # Request top-1
    results = await reranker.rerank(query, sample_candidates, top_k=1)
    assert len(results) == 1

    # Request top-5 (more than available)
    results = await reranker.rerank(query, sample_candidates, top_k=5)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_cross_encoder_ranking_order(sample_candidates):
    """Test that results are properly ranked."""
    reranker = CrossEncoderReranker()
    results = await reranker.rerank("machine learning", sample_candidates)

    # Check ranking indices are sequential
    ranks = [r.rank for r in results]
    assert ranks == list(range(len(results)))


@pytest.mark.asyncio
async def test_mock_reranker_basic(sample_candidates):
    """Test MockReranker (fallback for testing)."""
    reranker = MockReranker()
    query = "test query"

    results = await reranker.rerank(query, sample_candidates, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RerankedResult) for r in results)
    # Mock reranker uses original scores
    assert results[0].rerank_score >= results[1].rerank_score


@pytest.mark.asyncio
async def test_mock_reranker_preserves_original_score(sample_candidates):
    """Test that MockReranker preserves original scores."""
    reranker = MockReranker()

    results = await reranker.rerank("query", sample_candidates)

    for result in results:
        assert result.rerank_score == result.original_score


@pytest.mark.asyncio
async def test_reranked_result_fields():
    """Test RerankedResult dataclass fields."""
    result = RerankedResult(
        doc_id="doc1",
        text="sample text",
        source="test",
        rerank_score=0.95,
        original_score=0.80,
        rank=0,
    )

    assert result.doc_id == "doc1"
    assert result.text == "sample text"
    assert result.source == "test"
    assert result.rerank_score == 0.95
    assert result.original_score == 0.80
    assert result.rank == 0


def test_colbert_lite_init():
    """Test ColBERTLiteReranker initialization."""
    reranker = ColBERTLiteReranker()
    assert reranker.model is not None
    assert reranker.tokenizer is not None


@pytest.mark.asyncio
async def test_colbert_lite_reranker_basic(sample_candidates):
    """Test ColBERTLiteReranker basic reranking."""
    reranker = ColBERTLiteReranker()
    query = "What is machine learning?"

    results = await reranker.rerank(query, sample_candidates, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RerankedResult) for r in results)
    assert results[0].rerank_score >= results[1].rerank_score


@pytest.mark.asyncio
async def test_colbert_lite_single_scoring():
    """Test ColBERTLiteReranker single scoring."""
    reranker = ColBERTLiteReranker()

    score1 = reranker.score_single("machine learning", "Machine learning is artificial intelligence")
    score2 = reranker.score_single("machine learning", "Completely unrelated sentence about cooking")

    # Relevant doc should score higher
    assert score1 >= score2


@pytest.mark.asyncio
async def test_colbert_lite_no_candidates():
    """Test ColBERTLiteReranker with empty candidates."""
    reranker = ColBERTLiteReranker()
    results = await reranker.rerank("query", [])

    assert results == []


@pytest.mark.asyncio
async def test_retrieval_result_base_fields():
    """Test RetrievalResult base fields."""
    result = RetrievalResult(
        doc_id="doc1",
        text="sample text",
        source="test_source",
        score=0.95,
    )

    assert result.doc_id == "doc1"
    assert result.text == "sample text"
    assert result.source == "test_source"
    assert result.score == 0.95


def test_reranked_result_sorting():
    """Test that RerankedResult can be sorted by score."""
    results = [
        RerankedResult("doc1", "text1", "source", 0.5),
        RerankedResult("doc2", "text2", "source", 0.9),
        RerankedResult("doc3", "text3", "source", 0.7),
    ]

    sorted_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

    assert sorted_results[0].rerank_score == 0.9
    assert sorted_results[1].rerank_score == 0.7
    assert sorted_results[2].rerank_score == 0.5


@pytest.mark.asyncio
async def test_cross_encoder_batch_scoring(sample_candidates):
    """Test batch scoring consistency."""
    reranker = CrossEncoderReranker()
    query = "machine learning"

    # Rerank multiple times
    results1 = await reranker.rerank(query, sample_candidates)
    results2 = await reranker.rerank(query, sample_candidates)

    # Scores should be consistent
    for r1, r2 in zip(results1, results2):
        assert abs(r1.rerank_score - r2.rerank_score) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
