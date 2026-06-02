"""Integration tests for Qdrant server mode.

All tests are marked as integration and skip automatically if Qdrant is unreachable.
"""

import subprocess
import pytest
import asyncio
from typing import Dict, Any
from infracore.multimodal import QdrantRetriever, Source


class DummyEmbedder:
    async def embed_texts(self, texts):
        # Generate simple distinguishable vectors for testing ranking
        vecs = []
        for t in texts:
            v = [0.0] * 128
            if "first" in t:
                v[0] = 1.0
            elif "second" in t:
                v[1] = 1.0
            elif "third" in t:
                v[2] = 1.0
            else:
                v[0] = 0.5
            vecs.append(v)
        return vecs

    async def embed_images(self, images):
        return [[0.1] * 128 for _ in images]


@pytest.fixture
async def qdrant_conn_retriever():
    """Fixture: Qdrant retriever pointing to local server (skip if unavailable)."""
    try:
        ret = QdrantRetriever(
            collection_name="test_integration_server",
            url="http://localhost:6333",
            embedder=DummyEmbedder(),
            vector_size=128,
        )
        # Test connectivity
        ret.get_collection_info()
        yield ret
        # Cleanup
        try:
            await ret.clear()
        except Exception:
            pass
    except Exception as e:
        pytest.skip(f"Qdrant server not available on localhost:6333: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_server_upsert_and_search(qdrant_conn_retriever):
    """Upserts 3 dummy vectors, searches, and asserts top-1 returns correct Source."""
    ret = qdrant_conn_retriever

    # Index 3 items
    await ret.index(doc_id="doc1", text="first dummy document", page=1, confidence=0.99)
    await ret.index(doc_id="doc2", text="second dummy document", page=2, confidence=0.88)
    await ret.index(doc_id="doc3", text="third dummy document", page=3, confidence=0.77)

    # Search for "first"
    results = await ret.search_by_text("first", top_k=1)
    assert len(results) > 0
    assert results[0].source_id == "doc1"
    assert isinstance(results[0], Source)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_server_payload_keys(qdrant_conn_retriever):
    """Verifies all 5 payload keys exist (source_type, doc_id, page, bbox, snippet)."""
    ret = qdrant_conn_retriever
    bbox_val = {"left": 10, "top": 20, "width": 100, "height": 50}

    await ret.index(
        doc_id="doc_keys",
        text="Sample snippet for key validation",
        page=42,
        bbox=bbox_val,
        confidence=0.95,
    )

    # Directly retrieve the point payload using qdrant_client
    point_id = ret._make_point_id("doc_keys", 0)
    points = ret.client.retrieve(collection_name=ret.collection_name, ids=[point_id], with_payload=True)
    assert len(points) == 1
    
    payload = points[0].payload
    assert payload is not None
    assert payload.get("source_type") == "retrieved"
    assert payload.get("doc_id") == "doc_keys"
    assert payload.get("page") == 42
    assert payload.get("bbox") == bbox_val
    assert payload.get("snippet") == "Sample snippet for key validation"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_server_idempotent_upsert(qdrant_conn_retriever):
    """Upserts the same point twice, asserts collection size stays at 1."""
    ret = qdrant_conn_retriever

    await ret.index(doc_id="doc_dup", text="Idempotent content", page=1)
    info1 = ret.get_collection_info()
    initial_count = info1["points_count"]

    await ret.index(doc_id="doc_dup", text="Idempotent content", page=1)
    info2 = ret.get_collection_info()
    final_count = info2["points_count"]

    assert final_count == initial_count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_server_graceful_disconnect():
    """Stops Qdrant, confirms retriever raises a clean ConnectionError (not a raw exception)."""
    # 1. Stop Qdrant container
    try:
        stop_res = subprocess.run(["docker", "stop", "qdrant-local"], capture_output=True, text=True, check=True)
    except Exception as e:
        pytest.skip(f"Could not stop docker container 'qdrant-local': {e}")

    # 2. Confirm ConnectionError is raised when trying to initialize/connect
    try:
        with pytest.raises(ConnectionError) as exc_info:
            ret = QdrantRetriever(
                collection_name="test_integration_server_disconnected",
                url="http://localhost:6333",
                embedder=DummyEmbedder(),
                vector_size=128,
            )
        assert "Failed to connect" in str(exc_info.value)
    finally:
        # 3. Ensure we start the container back up
        subprocess.run(["docker", "start", "qdrant-local"], capture_output=True, text=True, check=True)
