"""
INFRACORE TEST — BGEEmbedder + E5Embedder

Comprehensive embedding tests using lightweight test models.
"""

import numpy as np
import pytest

from src.infracore.embedding.base import EmbedConfig
from src.infracore.embedding.bge_m3 import BGEEmbedder
from src.infracore.embedding.e5_embedder import E5Embedder, E5EmbedConfig


# ============================================================================
# BGEEmbedder Tests
# ============================================================================


@pytest.mark.asyncio
async def test_bge_embedder_output_shape():
    """Test that BGE embedder returns correct shape."""
    # Use lightweight test model
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384  # MiniLM output dim

    texts = ["hello world", "test text", "another example"]
    embeddings = await embedder.embed(texts)

    assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_bge_embedder_normalization():
    """Test that BGE embeddings are L2-normalized."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    texts = ["hello world", "test text"]
    embeddings = await embedder.embed(texts)

    # Check L2 norm is approximately 1.0
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Embedding {i} has norm {norm}, expected ~1.0"


@pytest.mark.asyncio
async def test_bge_embedder_batch_processing():
    """Test batch processing with multiple batches."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=4, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    # Create 10 texts (will require 3 batches with batch_size=4)
    texts = [f"text number {i}" for i in range(10)]
    embeddings = await embedder.embed(texts)

    assert embeddings.shape == (10, 384), f"Expected shape (10, 384), got {embeddings.shape}"


@pytest.mark.asyncio
async def test_bge_embedder_single():
    """Test embed_single returns (384,) not (1, 384)."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    embedding = await embedder.embed_single("hello world")

    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert embedding.dtype == np.float32


@pytest.mark.asyncio
async def test_bge_embedder_empty_input():
    """Test that empty input returns empty array."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    embeddings = await embedder.embed([])

    assert embeddings.shape == (0, 384), f"Expected shape (0, 384), got {embeddings.shape}"


@pytest.mark.asyncio
async def test_bge_embedder_cosine_similarity():
    """Test that identical texts have cosine similarity close to 1.0."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    text = "machine learning is a subset of artificial intelligence"
    embeddings = await embedder.embed([text, text])

    # For normalized vectors, cosine similarity = dot product
    similarity = np.dot(embeddings[0], embeddings[1])
    assert abs(similarity - 1.0) < 1e-4, f"Cosine similarity is {similarity}, expected ~1.0"


@pytest.mark.asyncio
async def test_bge_embedder_different_texts_similarity():
    """Test that different texts have lower cosine similarity."""
    config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, normalize=True)
    embedder = BGEEmbedder(config)
    embedder.embedding_dim = 384

    embeddings = await embedder.embed(["hello world", "goodbye moon"])

    # For normalized vectors, cosine similarity = dot product
    similarity = np.dot(embeddings[0], embeddings[1])
    assert similarity < 0.99, f"Different texts should have lower similarity, got {similarity}"


# ============================================================================
# E5Embedder Tests
# ============================================================================


@pytest.mark.asyncio
async def test_e5_embedder_output_shape():
    """Test that E5 embedder returns correct shape."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    texts = ["passage one", "passage two", "passage three"]
    embeddings = await embedder.embed(texts)

    assert embeddings.shape == (3, 384), f"Expected shape (3, 384), got {embeddings.shape}"


@pytest.mark.asyncio
async def test_e5_embedder_query_prefix():
    """Test that E5 embedder adds 'query: ' prefix when mode='query'."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="query",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    # Mock the _add_prefix to verify it's called correctly
    texts = ["what is AI?"]
    prefixed = embedder._add_prefix(texts)

    assert prefixed[0] == "query: what is AI?", f"Expected 'query: what is AI?', got '{prefixed[0]}'"


@pytest.mark.asyncio
async def test_e5_embedder_passage_prefix():
    """Test that E5 embedder adds 'passage: ' prefix when mode='passage'."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    texts = ["AI is artificial intelligence"]
    prefixed = embedder._add_prefix(texts)

    assert prefixed[0] == "passage: AI is artificial intelligence", f"Expected 'passage: ...' prefix, got '{prefixed[0]}'"


@pytest.mark.asyncio
async def test_e5_embedder_normalization():
    """Test that E5 embeddings are L2-normalized."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    texts = ["test passage one", "test passage two"]
    embeddings = await embedder.embed(texts)

    # Check L2 norm is approximately 1.0
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5, f"Embedding {i} has norm {norm}, expected ~1.0"


@pytest.mark.asyncio
async def test_e5_embedder_empty_input():
    """Test that empty input returns empty array."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    embeddings = await embedder.embed([])

    assert embeddings.shape == (0, 384), f"Expected shape (0, 384), got {embeddings.shape}"


@pytest.mark.asyncio
async def test_e5_embedder_single():
    """Test E5 embed_single."""
    config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )
    embedder = E5Embedder(config)
    embedder.embedding_dim = 384

    embedding = await embedder.embed_single("test passage")

    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"


@pytest.mark.asyncio
async def test_e5_embedder_query_vs_passage_different():
    """Test that query and passage embeddings of same text are different."""
    query_config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="query",
    )
    passage_config = E5EmbedConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize=True,
        mode="passage",
    )

    query_embedder = E5Embedder(query_config)
    query_embedder.embedding_dim = 384
    passage_embedder = E5Embedder(passage_config)
    passage_embedder.embedding_dim = 384

    text = "machine learning"
    query_emb = await query_embedder.embed([text])
    passage_emb = await passage_embedder.embed([text])

    # Query and passage embeddings should be different (different prefixes)
    similarity = np.dot(query_emb[0], passage_emb[0])
    assert similarity < 0.99, f"Query and passage embeddings should differ, similarity={similarity}"


# ============================================================================
# Comparative Tests
# ============================================================================


@pytest.mark.asyncio
async def test_bge_and_e5_consistency():
    """Test that both embedders handle same texts consistently."""
    bge_config = EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8)
    e5_config = E5EmbedConfig(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=8, mode="passage")

    bge_embedder = BGEEmbedder(bge_config)
    bge_embedder.embedding_dim = 384
    e5_embedder = E5Embedder(e5_config)
    e5_embedder.embedding_dim = 384

    texts = ["hello world", "test"]
    bge_emb = await bge_embedder.embed(texts)
    e5_emb = await e5_embedder.embed(texts)

    # Both should return same shape
    assert bge_emb.shape == e5_emb.shape, f"Shape mismatch: BGE {bge_emb.shape} vs E5 {e5_emb.shape}"
