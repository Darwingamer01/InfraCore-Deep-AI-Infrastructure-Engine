import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from pydantic import ValidationError

from src.infracore.vectordb.weaviate_store import WeaviateConfig, WeaviateVectorStore
from src.infracore.vectordb.base import SearchResult


class MockMetadata:
    def __init__(self, certainty=None, score=None, distance=None):
        self.certainty = certainty
        self.score = score
        self.distance = distance


class MockObject:
    def __init__(self, uuid, properties, certainty=None, score=None, distance=None):
        self.uuid = uuid
        self.properties = properties
        self.metadata = MockMetadata(certainty, score, distance)


class MockQueryReturn:
    def __init__(self, objects):
        self.objects = objects


@pytest.fixture
def weaviate_config():
    return WeaviateConfig(
        collection_name="TestCollection",
        vector_size=128,
        host="test-host",
        port=8080,
        distance_metric="cosine",
        top_k=5,
    )


@pytest.mark.asyncio
async def test_weaviate_config_validation():
    """Test WeaviateConfig validates parameters correctly."""
    # Test valid configuration
    cfg = WeaviateConfig(
        collection_name="ValidColl",
        vector_size=128,
    )
    assert cfg.host == "localhost"
    assert cfg.port == 8080
    assert cfg.distance_metric == "cosine"
    assert cfg.top_k == 5

    # Test invalid field types raise ValidationError
    with pytest.raises(ValidationError):
        WeaviateConfig(collection_name="Test", vector_size="not-an-int")

    with pytest.raises(ValidationError):
        WeaviateConfig(collection_name="Test", vector_size=128, port="not-an-int")


@pytest.mark.asyncio
async def test_weaviate_store_upsert(weaviate_config):
    """Test upsert maps parameters and calls insert_many in batches."""
    vectors = np.random.rand(5, 128)
    payloads = [{"text": f"chunk {i}"} for i in range(5)]
    ids = [f"00000000-0000-0000-0000-00000000000{i}" for i in range(5)]

    # Mock async client
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    
    mock_collections = MagicMock()
    mock_client.collections = mock_collections
    mock_collections.exists = AsyncMock(return_value=False)
    mock_collections.create = AsyncMock()
    
    mock_collection = MagicMock()
    mock_collection.data.insert_many = AsyncMock()
    mock_collections.get.return_value = mock_collection

    store = WeaviateVectorStore(weaviate_config)

    with patch("weaviate.use_async_with_local", return_value=mock_client) as mock_use_local:
        await store.upsert(vectors, payloads, ids)

        # Assert correct host and port passed to connection helper
        mock_use_local.assert_called_once_with(host="test-host", port=8080)
        
        # Assert collection existence checked and created
        mock_collections.exists.assert_called_once_with("TestCollection")
        mock_collections.create.assert_called_once()
        
        # Assert batch insert called
        mock_collection.data.insert_many.assert_called_once()
        
        called_args, _ = mock_collection.data.insert_many.call_args
        inserted_objects = called_args[0]
        assert len(inserted_objects) == 5
        assert inserted_objects[0].properties == payloads[0]
        assert inserted_objects[0].uuid == ids[0]
        assert np.allclose(inserted_objects[0].vector, vectors[0])


@pytest.mark.asyncio
async def test_weaviate_store_search(weaviate_config):
    """Test search handles query and correctly maps SearchResult properties."""
    query_vector = np.random.rand(128)
    mock_uuid = "00000000-0000-0000-0000-000000000001"
    mock_payload = {"text": "found chunk"}

    # Mock query response with certainty
    mock_obj = MockObject(uuid=mock_uuid, properties=mock_payload, certainty=0.92)
    mock_response = MockQueryReturn(objects=[mock_obj])

    # Mock client
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    
    mock_collections = MagicMock()
    mock_client.collections = mock_collections
    mock_collections.exists = AsyncMock(return_value=True)
    
    mock_collection = MagicMock()
    mock_collection.query.near_vector = AsyncMock(return_value=mock_response)
    mock_collections.get.return_value = mock_collection

    store = WeaviateVectorStore(weaviate_config)

    with patch("weaviate.use_async_with_local", return_value=mock_client):
        results = await store.search(query_vector, top_k=3)

        # Assert search queries are executed near vector
        mock_collection.query.near_vector.assert_called_once()
        _, kwargs = mock_collection.query.near_vector.call_args
        assert kwargs["near_vector"] == query_vector.tolist()
        assert kwargs["limit"] == 3

        assert len(results) == 1
        res = results[0]
        assert isinstance(res, SearchResult)
        assert res.id == mock_uuid
        assert res.score == 0.92
        assert res.payload == mock_payload


@pytest.mark.asyncio
async def test_weaviate_store_search_metadata_score_mapping(weaviate_config):
    """Test search correctly maps distance and scores to SearchResult.score."""
    query_vector = np.random.rand(128)
    mock_uuid = "00000000-0000-0000-0000-000000000001"
    mock_payload = {"text": "found"}

    # 1. Test score mapping
    mock_obj_score = MockObject(uuid=mock_uuid, properties=mock_payload, score=0.85)
    mock_response_score = MockQueryReturn(objects=[mock_obj_score])

    # 2. Test distance mapping (score = 1 - distance)
    mock_obj_distance = MockObject(uuid=mock_uuid, properties=mock_payload, distance=0.15)
    mock_response_distance = MockQueryReturn(objects=[mock_obj_distance])

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    
    mock_collections = MagicMock()
    mock_client.collections = mock_collections
    mock_collections.exists = AsyncMock(return_value=True)
    
    mock_collection = MagicMock()
    mock_collections.get.return_value = mock_collection

    store = WeaviateVectorStore(weaviate_config)

    with patch("weaviate.use_async_with_local", return_value=mock_client):
        # Verify score mapping
        mock_collection.query.near_vector = AsyncMock(return_value=mock_response_score)
        results = await store.search(query_vector, top_k=1)
        assert results[0].score == 0.85

        # Verify distance mapping
        mock_collection.query.near_vector = AsyncMock(return_value=mock_response_distance)
        results = await store.search(query_vector, top_k=1)
        assert results[0].score == 0.85
