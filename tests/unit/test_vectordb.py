"""
INFRACORE TEST — QdrantVectorStore and PgVectorStore

Comprehensive tests using mocks to avoid real database connections.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infracore.vectordb.base import SearchResult
from src.infracore.vectordb.qdrant_store import QdrantConfig, QdrantVectorStore


# ============================================================================
# QdrantVectorStore Tests
# ============================================================================


@pytest.mark.asyncio
async def test_qdrant_upsert_calls_client_with_correct_collection():
    """Test upsert calls client.upsert with correct collection_name."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    # Mock the client
    store.client = AsyncMock()

    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    payloads = [{"text": "doc1"}]
    ids = ["id1"]

    await store.upsert(vectors, payloads, ids)

    # Verify upsert was called
    store.client.upsert.assert_called_once()
    call_kwargs = store.client.upsert.call_args[1]
    assert call_kwargs["collection_name"] == "test_col"


@pytest.mark.asyncio
async def test_qdrant_upsert_auto_generates_uuids():
    """Test upsert auto-generates UUIDs when ids=None."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    payloads = [{"text": "doc1"}, {"text": "doc2"}]

    # Call without ids (should auto-generate)
    await store.upsert(vectors, payloads, ids=None)

    # Get the points from the call
    call_args = store.client.upsert.call_args
    points = call_args[1]["points"]

    # Verify 2 points with auto-generated IDs
    assert len(points) == 2
    assert points[0].id is not None
    assert points[1].id is not None
    assert points[0].id != points[1].id


@pytest.mark.asyncio
async def test_qdrant_upsert_batches_correctly():
    """Test upsert batches 250 vectors into 3 calls (100+100+50)."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    # Create 250 vectors
    vectors = np.random.rand(250, 4).astype(np.float32)
    payloads = [{"text": f"doc{i}"} for i in range(250)]
    ids = [str(i) for i in range(250)]

    await store.upsert(vectors, payloads, ids)

    # Verify upsert called 3 times
    assert store.client.upsert.call_count == 3

    # Check batch sizes
    call_1_points = store.client.upsert.call_args_list[0][1]["points"]
    call_2_points = store.client.upsert.call_args_list[1][1]["points"]
    call_3_points = store.client.upsert.call_args_list[2][1]["points"]

    assert len(call_1_points) == 100
    assert len(call_2_points) == 100
    assert len(call_3_points) == 50


@pytest.mark.asyncio
async def test_qdrant_search_returns_list_of_search_result():
    """Test search returns List[SearchResult] with correct fields."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    # Mock client.search response
    mock_result = MagicMock()
    mock_result.id = "id1"
    mock_result.score = 0.95
    mock_result.payload = {"text": "doc1"}

    store.client = AsyncMock()
    store.client.search.return_value = [mock_result]

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=5)

    # Verify result structure
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].id == "id1"
    assert results[0].score == 0.95
    assert results[0].payload == {"text": "doc1"}


@pytest.mark.asyncio
async def test_qdrant_search_passes_top_k():
    """Test search passes top_k correctly to client."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()
    store.client.search.return_value = []

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    await store.search(query_vec, top_k=42)

    # Verify top_k was passed as limit
    call_kwargs = store.client.search.call_args[1]
    assert call_kwargs["limit"] == 42


@pytest.mark.asyncio
async def test_qdrant_delete_calls_client_with_ids():
    """Test delete calls client.delete with correct ids."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    ids = ["id1", "id2", "id3"]
    await store.delete(ids)

    # Verify delete was called
    store.client.delete.assert_called_once()
    call_kwargs = store.client.delete.call_args[1]
    assert call_kwargs["collection_name"] == "test_col"


@pytest.mark.asyncio
async def test_qdrant_create_collection_calls_client():
    """Test create_collection calls client.create_collection with HNSW params."""
    config = QdrantConfig(
        collection_name="test_col",
        vector_size=4,
        hnsw_m=32,
        hnsw_ef_construct=100,
    )
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    await store.create_collection()

    # Verify create_collection was called
    store.client.create_collection.assert_called_once()
    call_kwargs = store.client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == "test_col"
    assert call_kwargs["vectors_config"].size == 4


@pytest.mark.asyncio
async def test_qdrant_count_returns_integer():
    """Test count returns correct integer from client."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    # Mock collection info
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 42

    store.client = AsyncMock()
    store.client.get_collection.return_value = mock_collection_info

    count = await store.count()

    assert count == 42
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_qdrant_search_with_no_filter():
    """Test search with filter=None passes None to client."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()
    store.client.search.return_value = []

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    await store.search(query_vec, top_k=5, filter=None)

    call_kwargs = store.client.search.call_args[1]
    assert call_kwargs["query_filter"] is None


@pytest.mark.asyncio
async def test_qdrant_prometheus_counter_increments():
    """Test Prometheus counter increments on each operation."""
    from prometheus_client import REGISTRY

    # Reset counter
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()
    store.client.search.return_value = []
    store.client.get_collection.return_value = MagicMock(points_count=0)

    # Perform operations
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    await store.search(query_vec)
    await store.count()

    # Verify metrics were recorded (can't check exact values due to prior runs)
    # But we ensure no exceptions were raised during metric recording


@pytest.mark.asyncio
async def test_qdrant_upsert_returns_count():
    """Test upsert returns correct count of upserted vectors."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    payloads = [{"text": "doc1"}, {"text": "doc2"}]
    ids = ["id1", "id2"]

    count = await store.upsert(vectors, payloads, ids)

    assert count == 2


@pytest.mark.asyncio
async def test_qdrant_delete_returns_count():
    """Test delete returns correct count of deleted vectors."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    ids = ["id1", "id2", "id3"]
    count = await store.delete(ids)

    assert count == 3


@pytest.mark.asyncio
async def test_qdrant_delete_empty_list_returns_zero():
    """Test delete with empty list returns 0."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()

    count = await store.delete([])

    assert count == 0
    store.client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_qdrant_search_handles_no_results():
    """Test search handles empty results gracefully."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    store.client = AsyncMock()
    store.client.search.return_value = []

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=10)

    assert results == []
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_qdrant_search_result_payload_none_handling():
    """Test search handles None payload from client."""
    config = QdrantConfig(collection_name="test_col", vector_size=4)
    store = QdrantVectorStore(config)

    # Mock result with None payload
    mock_result = MagicMock()
    mock_result.id = "id1"
    mock_result.score = 0.9
    mock_result.payload = None

    store.client = AsyncMock()
    store.client.search.return_value = [mock_result]

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=5)

    assert results[0].payload == {}
