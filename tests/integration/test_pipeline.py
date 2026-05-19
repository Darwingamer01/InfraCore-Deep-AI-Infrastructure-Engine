"""Integration test for full E2E RAG pipeline."""

import pytest
import asyncio
import numpy as np

from src.infracore.chunking.fixed import FixedChunker, FixedChunkConfig
from src.infracore.embedding.bge_m3 import BGEEmbedder, BGEConfig
from src.infracore.vectordb.qdrant_store import QdrantVectorStore, QdrantConfig
from src.infracore.retrieval.dense_retriever import DenseRetriever, DenseConfig
from src.infracore.retrieval.base import RetrievalResult


@pytest.fixture(scope="module")
async def pipeline_setup():
    """
    Module-scoped fixture for pipeline components.
    
    Uses lightweight models to avoid slow downloads.
    """
    # 1. Chunker: FixedChunker with small chunks for testing
    chunker_config = FixedChunkConfig(max_tokens=50, overlap=10)
    chunker = FixedChunker(chunker_config)

    # 2. Embedder: Lightweight all-MiniLM-L6-v2 (384-dim, ~90MB)
    #    Override the default BGE-M3 model
    embedder_config = BGEConfig()
    embedder = BGEEmbedder(embedder_config)
    # Force model to lightweight version
    embedder.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # Re-initialize to load the lightweight model
    await asyncio.to_thread(embedder._load_model)

    # 3. Vector Store: In-memory Qdrant
    #    QdrantClient(":memory:") is sync only, so we need to wrap in asyncio.to_thread
    #    For now, use standard localhost:6333 config and expect docker, or use in-memory flag
    vector_store_config = QdrantConfig(
        collection_name="integration_test",
        vector_size=384,  # all-MiniLM-L6-v2 outputs 384-dim
    )
    vector_store = QdrantVectorStore(vector_store_config)

    # 4. Retriever
    retriever_config = DenseConfig(top_k=3)
    retriever = DenseRetriever(retriever_config, vector_store, embedder)

    yield {
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
        "retriever": retriever,
    }

    # Cleanup
    try:
        await vector_store.delete_collection()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_pipeline_ingest_and_retrieve(pipeline_setup):
    """Test 1: Full pipeline ingest and retrieve."""
    chunker = pipeline_setup["chunker"]
    embedder = pipeline_setup["embedder"]
    vector_store = pipeline_setup["vector_store"]
    retriever = pipeline_setup["retriever"]

    # Test corpus
    docs = [
        "Vector databases store high-dimensional embeddings for similarity search.",
        "RAG combines retrieval with language model generation.",
        "HNSW is an approximate nearest neighbor algorithm used in vector search.",
        "Chunking splits documents into smaller pieces before embedding.",
        "Semantic search uses meaning rather than keyword matching.",
    ]

    # Step 1: Chunk documents
    all_chunks = []
    for doc in docs:
        chunks = await chunker.chunk(doc)
        all_chunks.extend([c.text for c in chunks])

    assert len(all_chunks) > 0, "Should produce chunks"

    # Step 2: Embed chunks
    embeddings = await embedder.embed(all_chunks)
    assert embeddings.shape[0] == len(all_chunks)
    assert embeddings.shape[1] == 384

    # Step 3: Create collection and upsert
    await vector_store.create_collection()

    payloads = [{"text": chunk} for chunk in all_chunks]
    await vector_store.upsert(
        vectors=embeddings,
        payloads=payloads,
        ids=[str(i) for i in range(len(all_chunks))],
    )

    # Verify insertion
    count = await vector_store.count()
    assert count == len(all_chunks)


@pytest.mark.asyncio
async def test_pipeline_retrieve_vector_similarity(pipeline_setup):
    """Test 2: Retrieve returns top result relevant to query."""
    chunker = pipeline_setup["chunker"]
    embedder = pipeline_setup["embedder"]
    vector_store = pipeline_setup["vector_store"]
    retriever = pipeline_setup["retriever"]

    docs = [
        "Vector databases store high-dimensional embeddings for similarity search.",
        "RAG combines retrieval with language model generation.",
        "HNSW is an approximate nearest neighbor algorithm used in vector search.",
        "Chunking splits documents into smaller pieces before embedding.",
        "Semantic search uses meaning rather than keyword matching.",
    ]

    # Ingest
    all_chunks = []
    for doc in docs:
        chunks = await chunker.chunk(doc)
        all_chunks.extend([c.text for c in chunks])

    embeddings = await embedder.embed(all_chunks)

    await vector_store.create_collection()
    payloads = [{"text": chunk} for chunk in all_chunks]
    await vector_store.upsert(
        vectors=embeddings,
        payloads=payloads,
        ids=[str(i) for i in range(len(all_chunks))],
    )

    # Query
    query = "how does vector similarity search work"
    results = await retriever.retrieve(query, top_k=3)

    assert len(results) > 0, "Should return results"
    top_result = results[0]
    # Top result should mention vectors or embeddings
    assert (
        "vector" in top_result.text.lower()
        or "embedding" in top_result.text.lower()
        or "search" in top_result.text.lower()
    ), f"Expected vector/embedding/search in: {top_result.text}"


@pytest.mark.asyncio
async def test_pipeline_retrieve_rag_query(pipeline_setup):
    """Test 3: Retrieve for RAG query."""
    chunker = pipeline_setup["chunker"]
    embedder = pipeline_setup["embedder"]
    vector_store = pipeline_setup["vector_store"]
    retriever = pipeline_setup["retriever"]

    docs = [
        "Vector databases store high-dimensional embeddings for similarity search.",
        "RAG combines retrieval with language model generation.",
        "HNSW is an approximate nearest neighbor algorithm used in vector search.",
        "Chunking splits documents into smaller pieces before embedding.",
        "Semantic search uses meaning rather than keyword matching.",
    ]

    # Ingest
    all_chunks = []
    for doc in docs:
        chunks = await chunker.chunk(doc)
        all_chunks.extend([c.text for c in chunks])

    embeddings = await embedder.embed(all_chunks)

    await vector_store.create_collection()
    payloads = [{"text": chunk} for chunk in all_chunks]
    await vector_store.upsert(
        vectors=embeddings,
        payloads=payloads,
        ids=[str(i) for i in range(len(all_chunks))],
    )

    # Query
    query = "what is RAG"
    results = await retriever.retrieve(query)

    assert len(results) > 0
    # At least one result should mention RAG or retrieval
    found = any(
        "RAG" in r.text or "retrieval" in r.text.lower() for r in results
    )
    assert found, f"Expected RAG/retrieval in results: {[r.text for r in results]}"


@pytest.mark.asyncio
async def test_pipeline_results_are_retrieval_result_objects(pipeline_setup):
    """Test 4: Results are RetrievalResult objects (not dicts)."""
    chunker = pipeline_setup["chunker"]
    embedder = pipeline_setup["embedder"]
    vector_store = pipeline_setup["vector_store"]
    retriever = pipeline_setup["retriever"]

    docs = [
        "Test document one.",
        "Test document two.",
        "Test document three.",
    ]

    all_chunks = []
    for doc in docs:
        chunks = await chunker.chunk(doc)
        all_chunks.extend([c.text for c in chunks])

    embeddings = await embedder.embed(all_chunks)

    await vector_store.create_collection()
    payloads = [{"text": chunk} for chunk in all_chunks]
    await vector_store.upsert(
        vectors=embeddings,
        payloads=payloads,
        ids=[str(i) for i in range(len(all_chunks))],
    )

    results = await retriever.retrieve("test")

    assert len(results) > 0
    # Verify results are RetrievalResult objects
    from src.infracore.retrieval.base import RetrievalResult
    for result in results:
        assert isinstance(result, RetrievalResult)
        assert hasattr(result, "text")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata")


@pytest.mark.asyncio
async def test_pipeline_all_results_have_positive_scores(pipeline_setup):
    """Test 5: All results have score > 0.0."""
    chunker = pipeline_setup["chunker"]
    embedder = pipeline_setup["embedder"]
    vector_store = pipeline_setup["vector_store"]
    retriever = pipeline_setup["retriever"]

    docs = [
        "First document about AI.",
        "Second document about ML.",
        "Third document about DL.",
    ]

    all_chunks = []
    for doc in docs:
        chunks = await chunker.chunk(doc)
        all_chunks.extend([c.text for c in chunks])

    embeddings = await embedder.embed(all_chunks)

    await vector_store.create_collection()
    payloads = [{"text": chunk} for chunk in all_chunks]
    await vector_store.upsert(
        vectors=embeddings,
        payloads=payloads,
        ids=[str(i) for i in range(len(all_chunks))],
    )

    results = await retriever.retrieve("document")

    assert len(results) > 0
    for result in results:
        assert result.score > 0.0, f"Expected positive score, got {result.score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
