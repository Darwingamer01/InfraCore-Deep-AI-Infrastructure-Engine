"""
Smoke test for QdrantVectorStore against real Qdrant instance.

Requires: docker compose up qdrant (listening on localhost:6333)
Run with: pytest tests/smoke/test_qdrant_integration.py -v
"""

import asyncio
import numpy as np
import pytest
from infracore.config import QdrantConfig
from infracore.vectordb.qdrant_store import QdrantVectorStore, SearchResult


@pytest.mark.asyncio
async def test_qdrant_smoke_create_and_search():
    """
    Smoke test: Create collection, insert vectors, search, cleanup.
    
    Requires real Qdrant instance running on localhost:6333.
    """
    config = QdrantConfig(
        collection_name="smoke_test_col",
        vector_size=4,
        host="localhost",
        port=6333,
    )
    
    store = QdrantVectorStore(config)
    
    try:
        # Create collection
        await store.create_collection()
        
        # Insert 5 random 4-dimensional vectors
        vectors = np.random.randn(5, 4).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # L2 normalize
        
        payloads = [{"text": f"Document {i}", "index": i} for i in range(5)]
        
        count = await store.upsert(vectors, payloads)
        assert count == 5, f"Expected 5 vectors upserted, got {count}"
        
        # Count
        total = await store.count()
        assert total == 5, f"Expected 5 vectors in collection, got {total}"
        
        # Search for top 3 most similar to first vector
        results = await store.search(vectors[0:1], top_k=3)
        assert isinstance(results, list), "Search should return list"
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # First result should be the query vector itself (score ~1.0)
        assert isinstance(results[0], SearchResult), "Results should be SearchResult objects"
        assert results[0].score > 0.99, f"Top match should be ~1.0, got {results[0].score}"
        assert results[0].payload == {"text": "Document 0", "index": 0}
        
        # Delete by IDs
        delete_count = await store.delete([results[0].id])
        assert delete_count == 1, f"Expected 1 vector deleted, got {delete_count}"
        
        # Final count
        final_count = await store.count()
        assert final_count == 4, f"Expected 4 vectors after delete, got {final_count}"
        
    finally:
        # Cleanup
        await store.delete_collection()


@pytest.mark.asyncio
async def test_qdrant_smoke_batch_insert():
    """
    Smoke test: Large batch insert (250 vectors → 3 batch calls).
    """
    config = QdrantConfig(
        collection_name="smoke_test_batch",
        vector_size=4,
        host="localhost",
        port=6333,
    )
    
    store = QdrantVectorStore(config)
    
    try:
        await store.create_collection()
        
        # Insert 250 vectors in batches of 100
        vectors = np.random.randn(250, 4).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        payloads = [{"batch": i // 100, "idx": i % 100} for i in range(250)]
        
        count = await store.upsert(vectors, payloads)
        assert count == 250, f"Expected 250 vectors, got {count}"
        
        total = await store.count()
        assert total == 250, f"Expected 250 in collection, got {total}"
        
    finally:
        await store.delete_collection()


if __name__ == "__main__":
    # Run with: python -m pytest tests/smoke/test_qdrant_integration.py -v
    pytest.main([__file__, "-v"])
