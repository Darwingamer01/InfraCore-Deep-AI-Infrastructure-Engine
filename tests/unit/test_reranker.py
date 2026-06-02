"""Unit tests for CrossEncoderReranker using mock CrossEncoder model."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from src.infracore.retrieval.base import RetrievalResult
from src.infracore.retrieval.reranker import CrossEncoderReranker, RerankerConfig


def test_reranker_config_validation():
    """Test RerankerConfig type validation and default values."""
    config = RerankerConfig()
    assert config.model_name == "BAAI/bge-reranker-large"
    assert config.top_n is None
    assert config.score_threshold is None
    assert config.batch_size == 32

    # Verify custom validation works
    custom_config = RerankerConfig(
        model_name="custom-model",
        top_n=5,
        score_threshold=0.7,
        batch_size=16
    )
    assert custom_config.model_name == "custom-model"
    assert custom_config.top_n == 5
    assert custom_config.score_threshold == 0.7
    assert custom_config.batch_size == 16

    # Verify Pydantic validation error on incorrect types
    with pytest.raises(ValidationError):
        RerankerConfig(batch_size="not-an-int")


@pytest.mark.asyncio
@patch("src.infracore.retrieval.reranker.CrossEncoder")
async def test_rerank_basic_sorting(mock_cross_encoder):
    """Test that rerank() returns results sorted by scores descending."""
    # Setup mock predict output
    # raw logits: 2.0 (sigmoid ~0.88), -1.0 (sigmoid ~0.27), 0.5 (sigmoid ~0.62)
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model
    mock_model.predict.return_value = np.array([2.0, -1.0, 0.5])

    config = RerankerConfig(model_name="test-model")
    reranker = CrossEncoderReranker(config)

    candidates = [
        RetrievalResult(doc_id="doc1", text="text1", score=0.9, metadata={"custom_key": "val1"}),
        RetrievalResult(doc_id="doc2", text="text2", score=0.8, metadata={"custom_key": "val2"}),
        RetrievalResult(doc_id="doc3", text="text3", score=0.7, metadata={"custom_key": "val3"}),
    ]

    results = await reranker.rerank("query", candidates)

    # 3 results returned, sorted by reranker score descending
    assert len(results) == 3
    assert results[0].doc_id == "doc1"  # sigmoid ~0.88
    assert results[1].doc_id == "doc3"  # sigmoid ~0.62
    assert results[2].doc_id == "doc2"  # sigmoid ~0.27

    # Check reranker_score metadata exists and original scores are preserved
    assert results[0].metadata["reranker_score"] > 0.8
    assert results[0].score == 0.9
    assert results[0].metadata["custom_key"] == "val1"
    
    assert results[1].metadata["reranker_score"] > 0.6
    assert results[1].score == 0.7
    assert results[1].metadata["custom_key"] == "val3"

    assert results[2].metadata["reranker_score"] < 0.3
    assert results[2].score == 0.8
    assert results[2].metadata["custom_key"] == "val2"


@pytest.mark.asyncio
@patch("src.infracore.retrieval.reranker.CrossEncoder")
async def test_rerank_top_n(mock_cross_encoder):
    """Test that top_n limits the number of results returned."""
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model
    mock_model.predict.return_value = np.array([2.0, -1.0, 0.5])

    config = RerankerConfig(top_n=2)
    reranker = CrossEncoderReranker(config)

    candidates = [
        RetrievalResult(doc_id="doc1", text="text1", score=0.9),
        RetrievalResult(doc_id="doc2", text="text2", score=0.8),
        RetrievalResult(doc_id="doc3", text="text3", score=0.7),
    ]

    results = await reranker.rerank("query", candidates)

    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[1].doc_id == "doc3"


@pytest.mark.asyncio
@patch("src.infracore.retrieval.reranker.CrossEncoder")
async def test_rerank_score_threshold(mock_cross_encoder):
    """Test that score_threshold filters out low relevance scores."""
    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model
    # sigmoids: ~0.88, ~0.27, ~0.62
    mock_model.predict.return_value = np.array([2.0, -1.0, 0.5])

    config = RerankerConfig(score_threshold=0.5)
    reranker = CrossEncoderReranker(config)

    candidates = [
        RetrievalResult(doc_id="doc1", text="text1", score=0.9),
        RetrievalResult(doc_id="doc2", text="text2", score=0.8),
        RetrievalResult(doc_id="doc3", text="text3", score=0.7),
    ]

    results = await reranker.rerank("query", candidates)

    # doc2 with score ~0.27 is filtered out
    assert len(results) == 2
    assert results[0].doc_id == "doc1"
    assert results[1].doc_id == "doc3"
    assert all(r.metadata["reranker_score"] >= 0.5 for r in results)


@pytest.mark.asyncio
async def test_rerank_empty_input():
    """Test that empty candidate list returns empty results immediately."""
    reranker = CrossEncoderReranker()
    results = await reranker.rerank("query", [])
    assert results == []


@pytest.mark.asyncio
async def test_reranker_missing_libraries():
    """Test that missing sentence-transformers library raises ImportError."""
    reranker = CrossEncoderReranker()
    with patch("src.infracore.retrieval.reranker.CrossEncoder", None):
        with pytest.raises(ImportError) as exc_info:
            await reranker.rerank("query", [RetrievalResult(text="doc")])
        assert "sentence-transformers is required" in str(exc_info.value)


@pytest.mark.asyncio
@patch("src.infracore.retrieval.reranker.CrossEncoder")
async def test_hybrid_retriever_with_reranker(mock_cross_encoder):
    """Test HybridRetriever calls rerank() when reranker is set."""
    from unittest.mock import AsyncMock
    from src.infracore.retrieval.hybrid_retriever import HybridRetriever, HybridConfig
    from src.infracore.vectordb.base import SearchResult

    vector_store = AsyncMock()
    embedder = AsyncMock()
    embedder.embed_single.return_value = np.zeros(384)
    embedder.embed.return_value = np.zeros((2, 384))
    
    vector_store.search.return_value = [
        SearchResult(id="0", score=0.9, payload={"text": "doc0"}),
        SearchResult(id="1", score=0.8, payload={"text": "doc1"}),
    ]

    mock_model = MagicMock()
    mock_cross_encoder.return_value = mock_model
    mock_model.predict.return_value = np.array([2.0, -1.0])

    config = HybridConfig(top_k=2)
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever(config, vector_store, embedder, reranker=reranker)

    # Mock BM25 index docs
    retriever.doc_id_to_text = {"0": "doc0", "1": "doc1"}
    retriever.bm25.search = MagicMock(return_value=[(0, 1.0), (1, 0.5)])

    results = await retriever.retrieve("query")

    assert len(results) == 2
    assert results[0].doc_id == "0"
    assert results[0].metadata["reranker_score"] > 0.8
    assert results[1].doc_id == "1"
    assert results[1].metadata["reranker_score"] < 0.3

