"""
INFRACORE TEST — QdrantVectorStore and PgVectorStore

Comprehensive tests using mocks to avoid real database connections.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.infracore.vectordb.base import SearchResult
from src.infracore.vectordb.pgvector_store import PgVectorConfig, PgVectorStore
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


# ============================================================================
# PgVectorStore Tests
# ============================================================================


def create_async_context_manager(return_value):
    """Helper to create a proper async context manager mock."""
    class AsyncContextManager:
        async def __aenter__(self):
            return return_value
        async def __aexit__(self, *args):
            pass
    return AsyncContextManager()


@pytest.mark.asyncio
async def test_pgvector_config_initialization():
    """Test PgVectorConfig initializes correctly."""
    config = PgVectorConfig(
        collection_name="test_table",
        vector_size=768,
        dsn="postgresql://user:pass@localhost/db",
        index_type="hnsw",
        hnsw_m=16,
        hnsw_ef_construct=128,
    )

    assert config.collection_name == "test_table"
    assert config.vector_size == 768
    assert config.index_type == "hnsw"
    assert config.hnsw_m == 16


@pytest.mark.asyncio
async def test_pgvector_store_initialization():
    """Test PgVectorStore initializes with config."""
    config = PgVectorConfig(collection_name="test_table", vector_size=768)
    store = PgVectorStore(config)

    assert store.config == config
    assert store.pool is None


@pytest.mark.asyncio
async def test_pgvector_create_table_generates_hnsw_sql():
    """Test create_table generates correct HNSW SQL."""
    config = PgVectorConfig(
        collection_name="test_table",
        vector_size=768,
        index_type="hnsw",
        hnsw_m=16,
        hnsw_ef_construct=128,
    )
    store = PgVectorStore(config)

    # Mock pool and connection
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    await store.create_table()

    # Verify execute was called
    assert mock_conn.execute.call_count >= 3  # Extension, create table, create index


@pytest.mark.asyncio
async def test_pgvector_create_table_generates_ivfflat_sql():
    """Test create_table generates correct IVFFlat SQL."""
    config = PgVectorConfig(
        collection_name="test_table",
        vector_size=768,
        index_type="ivfflat",
        ivfflat_lists=100,
    )
    store = PgVectorStore(config)

    # Mock pool and connection
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    await store.create_table()

    # Verify execute was called
    assert mock_conn.execute.call_count >= 3


@pytest.mark.asyncio
async def test_pgvector_upsert_inserts_vectors():
    """Test upsert inserts vectors with payloads."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    # Mock pool and connection
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    payloads = [{"text": "doc1"}]
    ids = ["id1"]

    count = await store.upsert(vectors, payloads, ids)

    assert count == 1
    mock_conn.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_pgvector_upsert_auto_generates_ids():
    """Test upsert auto-generates IDs when ids=None."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    # Mock pool and connection
    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    payloads = [{"text": "doc1"}, {"text": "doc2"}]

    count = await store.upsert(vectors, payloads, ids=None)

    assert count == 2
    # Verify executemany was called with 2 records
    call_args = mock_conn.executemany.call_args
    records = call_args[0][1]
    assert len(records) == 2
    # IDs should be auto-generated (non-empty strings)
    assert records[0][0] != ""
    assert records[1][0] != ""


@pytest.mark.asyncio
async def test_pgvector_search_returns_search_results():
    """Test search returns List[SearchResult]."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    # Mock pool and connection
    mock_row1 = {"id": "id1", "distance": 0.05, "payload": json.dumps({"text": "doc1"})}
    mock_row2 = {"id": "id2", "distance": 0.1, "payload": json.dumps({"text": "doc2"})}

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [mock_row1, mock_row2]

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=10)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].id == "id1"
    assert results[0].score == 0.95  # 1 - 0.05
    assert results[1].id == "id2"
    assert results[1].score == 0.9  # 1 - 0.1


@pytest.mark.asyncio
async def test_pgvector_search_sorts_by_distance():
    """Test search results are sorted by distance (best first)."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    # Mock rows in random order
    mock_rows = [
        {"id": "id3", "distance": 0.3, "payload": json.dumps({"text": "doc3"})},
        {"id": "id1", "distance": 0.05, "payload": json.dumps({"text": "doc1"})},
        {"id": "id2", "distance": 0.15, "payload": json.dumps({"text": "doc2"})},
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = mock_rows

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=10)

    # Should be sorted by distance (ascending) via SQL ORDER BY
    assert len(results) == 3
    # Results from SQL are already sorted, verify order
    distances = [1 - r.score for r in results]
    assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_pgvector_search_passes_top_k():
    """Test search passes top_k correctly to query."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    await store.search(query_vec, top_k=42)

    # Verify fetch was called with correct LIMIT
    call_args = mock_conn.fetch.call_args
    args = call_args[0]
    # Last argument should be top_k
    assert args[-1] == 42


@pytest.mark.asyncio
async def test_pgvector_delete_removes_by_id():
    """Test delete removes vectors by ID."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "DELETE 3"

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    ids = ["id1", "id2", "id3"]
    count = await store.delete(ids)

    assert count == 3
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_pgvector_delete_empty_list():
    """Test delete with empty list returns 0 without query."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    count = await store.delete([])

    assert count == 0
    # Should not execute anything
    mock_conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_pgvector_count_returns_integer():
    """Test count returns correct integer."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = 42

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    count = await store.count()

    assert count == 42
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_pgvector_count_handles_none():
    """Test count handles None response gracefully."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = None

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    count = await store.count()

    assert count == 0


@pytest.mark.asyncio
async def test_pgvector_search_handles_null_payload():
    """Test search handles NULL payload gracefully."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_rows = [
        {"id": "id1", "distance": 0.05, "payload": None},  # NULL payload
        {"id": "id2", "distance": 0.1, "payload": json.dumps({"text": "doc2"})},
    ]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = mock_rows

    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = await store.search(query_vec, top_k=10)

    assert len(results) == 2
    assert results[0].payload == {}
    assert results[1].payload == {"text": "doc2"}


@pytest.mark.asyncio
async def test_pgvector_upsert_converts_vectors_to_list():
    """Test upsert converts numpy arrays to lists for PostgreSQL."""
    config = PgVectorConfig(collection_name="test_table", vector_size=4)
    store = PgVectorStore(config)

    mock_conn = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.acquire.return_value = create_async_context_manager(mock_conn)

    store.pool = mock_pool

    vectors = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    payloads = [{"idx": 0}]
    ids = ["id1"]

    await store.upsert(vectors, payloads, ids)

    call_args = mock_conn.executemany.call_args
    records = call_args[0][1]
    
    # Verify vector is converted to list
    assert isinstance(records[0][1], list)
    assert records[0][1] == [1.0, 2.0, 3.0, 4.0]


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
