"""Test suite for HybridRetriever and DenseRetriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.infracore.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridConfig,
    BM25Index,
)
from src.infracore.retrieval.dense_retriever import DenseRetriever, DenseConfig
from src.infracore.retrieval.base import RetrievalResult
from src.infracore.vectordb.base import SearchResult


# ============================================================================
# HybridRetriever Tests (1-8)
# ============================================================================


@pytest.mark.asyncio
async def test_hybrid_retrieve_calls_embedder_once():
    """Test 1: retrieve() calls embedder.embed_single() exactly once."""
    config = HybridConfig()
    vector_store = AsyncMock()
    embedder = AsyncMock()

    # Mock embedder to return 384-dim vector
    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((5, 384))

    # Mock vector store search
    vector_store.search.return_value = [
        SearchResult(id="0", score=0.9, payload={"text": "doc 0"})
    ]

    retriever = HybridRetriever(config, vector_store, embedder)

    # Build index
    await retriever.build_index(["doc 0", "doc 1", "doc 2"])

    # Retrieve
    await retriever.retrieve("test query")

    # Verify embedder.embed_single called exactly once
    embedder.embed_single.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_retrieve_calls_vector_store_with_dense_top_k():
    """Test 2: retrieve() calls vector_store.search() with dense_top_k."""
    config = HybridConfig(dense_top_k=15)
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((5, 384))
    vector_store.search.return_value = []

    retriever = HybridRetriever(config, vector_store, embedder)
    await retriever.build_index(["doc 0", "doc 1", "doc 2"])
    await retriever.retrieve("test query")

    # Verify search was called with dense_top_k=15
    vector_store.search.assert_called()
    call_args = vector_store.search.call_args
    assert call_args.kwargs["top_k"] == 15


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion_formula():
    """Test 3: RRF fusion uses correct formula."""
    config = HybridConfig(
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        dense_top_k=5,
        sparse_top_k=5,
    )
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((3, 384))

    # Mock dense results: doc_0 at rank 0, doc_1 at rank 1
    vector_store.search.return_value = [
        SearchResult(id="0", score=0.95, payload={"text": "dense result 0"}),
        SearchResult(id="1", score=0.85, payload={"text": "dense result 1"}),
    ]

    retriever = HybridRetriever(config, vector_store, embedder)
    await retriever.build_index(["doc0", "doc1", "doc2"])

    # Mock BM25 to return different ranking: doc_1 at rank 0, doc_0 at rank 1
    retriever.bm25.search = MagicMock(return_value=[(1, 2.5), (0, 2.0)])

    results = await retriever.retrieve("query")

    # Expected RRF scores:
    # doc_0: 0.7/(60+0) + 0.3/(60+1) = 0.7/60 + 0.3/61 ≈ 0.01167 + 0.00492 ≈ 0.01659
    # doc_1: 0.7/(60+1) + 0.3/(60+0) = 0.7/61 + 0.3/60 ≈ 0.01148 + 0.00500 ≈ 0.01648

    assert len(results) > 0
    # doc_0 should rank slightly higher
    assert results[0].metadata["doc_id"] == "0"


@pytest.mark.asyncio
async def test_hybrid_retrieve_returns_at_most_top_k():
    """Test 4: retrieve() returns at most top_k results."""
    config = HybridConfig(top_k=3, dense_top_k=10, sparse_top_k=10)
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((5, 384))

    # Mock 5 dense results
    vector_store.search.return_value = [
        SearchResult(id=str(i), score=0.9 - i * 0.1, payload={"text": f"doc{i}"})
        for i in range(5)
    ]

    retriever = HybridRetriever(config, vector_store, embedder)
    await retriever.build_index([f"doc{i}" for i in range(5)])

    # Mock BM25
    retriever.bm25.search = MagicMock(
        return_value=[(i, 1.0) for i in range(5)]
    )

    results = await retriever.retrieve("query")

    assert len(results) <= 3


@pytest.mark.asyncio
async def test_hybrid_results_sorted_by_fused_score():
    """Test 5: Results are sorted descending by fused score."""
    config = HybridConfig(dense_top_k=5, sparse_top_k=5)
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((3, 384))

    vector_store.search.return_value = [
        SearchResult(id="0", score=0.9, payload={"text": "doc0"}),
        SearchResult(id="1", score=0.8, payload={"text": "doc1"}),
        SearchResult(id="2", score=0.7, payload={"text": "doc2"}),
    ]

    retriever = HybridRetriever(config, vector_store, embedder)
    await retriever.build_index(["doc0", "doc1", "doc2"])

    retriever.bm25.search = MagicMock(
        return_value=[(0, 3.0), (1, 2.5), (2, 2.0)]
    )

    results = await retriever.retrieve("query")

    # Verify sorted descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_hybrid_retrieval_method():
    """Test 6: RetrievalResult.retrieval_method == 'hybrid'."""
    config = HybridConfig(dense_top_k=1, sparse_top_k=1)
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((1, 384))

    vector_store.search.return_value = [
        SearchResult(id="0", score=0.9, payload={"text": "result"})
    ]

    retriever = HybridRetriever(config, vector_store, embedder)
    await retriever.build_index(["result"])
    retriever.bm25.search = MagicMock(return_value=[(0, 1.0)])

    results = await retriever.retrieve("query")

    assert len(results) > 0
    assert results[0].metadata["retrieval_method"] == "hybrid"


def test_bm25_index_build():
    """Test 7: build_index() creates searchable index."""
    bm25 = BM25Index()
    corpus = [
        "machine learning algorithms",
        "deep neural networks",
        "machine learning is powerful",
    ]
    bm25.build(corpus)

    assert bm25.num_docs == 3
    assert len(bm25.doc_index) == 3
    assert bm25.avg_doc_length > 0


def test_bm25_higher_tf_scores_higher():
    """Test 8: BM25 - higher TF term scores higher than lower TF term."""
    bm25 = BM25Index()
    corpus = [
        "test test test test",  # High TF for 'test'
        "test example",          # Low TF for 'test'
    ]
    bm25.build(corpus)

    results = bm25.search("test", top_k=2)

    assert len(results) == 2
    # Doc 0 (high TF) should score higher than doc 1 (low TF)
    assert results[0][0] == 0
    assert results[0][1] > results[1][1]


# ============================================================================
# DenseRetriever Tests (9-12)
# ============================================================================


@pytest.mark.asyncio
async def test_dense_results_sorted_descending():
    """Test 9: retrieve() returns results sorted by score descending."""
    config = DenseConfig()
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)

    # Mock results with descending scores
    vector_store.search.return_value = [
        SearchResult(id="0", score=0.95, payload={"text": "top result"}),
        SearchResult(id="1", score=0.85, payload={"text": "middle result"}),
        SearchResult(id="2", score=0.75, payload={"text": "lower result"}),
    ]

    retriever = DenseRetriever(config, vector_store, embedder)
    results = await retriever.retrieve("query")

    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_dense_score_threshold_filtering():
    """Test 10: score_threshold filters out low-scoring results."""
    config = DenseConfig(score_threshold=0.8)
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)

    vector_store.search.return_value = [
        SearchResult(id="0", score=0.95, payload={"text": "good"}),
        SearchResult(id="1", score=0.75, payload={"text": "bad"}),
    ]

    retriever = DenseRetriever(config, vector_store, embedder)
    results = await retriever.retrieve("query")

    # Only result with score >= 0.8 should be included
    assert len(results) == 1
    assert results[0].score == 0.95


@pytest.mark.asyncio
async def test_dense_retrieval_method():
    """Test 11: RetrievalResult.retrieval_method == 'dense'."""
    config = DenseConfig()
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)

    vector_store.search.return_value = [
        SearchResult(id="0", score=0.9, payload={"text": "result"})
    ]

    retriever = DenseRetriever(config, vector_store, embedder)
    results = await retriever.retrieve("query")

    assert len(results) == 1
    assert results[0].metadata["retrieval_method"] == "dense"


@pytest.mark.asyncio
async def test_dense_empty_results():
    """Test 12: retrieve() with empty vector_store results returns empty list."""
    config = DenseConfig()
    vector_store = AsyncMock()
    embedder = AsyncMock()

    embedder.embed_single.return_value = np.zeros(384)
    vector_store.search.return_value = []

    retriever = DenseRetriever(config, vector_store, embedder)
    results = await retriever.retrieve("query")

    assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
