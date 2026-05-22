"""Optional integration test for Qdrant server mode.

Requires a local Qdrant server running (e.g., via Docker).
Skips gracefully if server is unavailable.

Run with:
  docker run -p 6333:6333 qdrant/qdrant
  PYTHONPATH=src pytest tests/integration/test_qdrant_server.py -q -vv -s
"""

import asyncio
import pytest
from infracore.multimodal import QdrantRetriever, Source, OCRPipeline


@pytest.fixture
async def qdrant_server_retriever():
    """Fixture: Qdrant retriever pointing to local server (skip if unavailable)."""
    try:
        retriever = QdrantRetriever(
            collection_name="test_multimodal_server",
            url="http://localhost:6333",
            vector_size=384,
        )
        # Test connectivity
        info = retriever.get_collection_info()
        yield retriever
        # Cleanup
        await retriever.clear()
    except Exception as e:
        pytest.skip(f"Qdrant server not available: {e}")


@pytest.mark.asyncio
async def test_qdrant_server_connection(qdrant_server_retriever):
    """Test connection to Qdrant server."""
    retriever = qdrant_server_retriever
    assert retriever.client is not None
    assert retriever.url == "http://localhost:6333"
    
    info = retriever.get_collection_info()
    assert "name" in info
    assert info["name"] == "test_multimodal_server"
    print(f"✓ Connected to Qdrant server: {info}")


@pytest.mark.asyncio
async def test_qdrant_server_persistence(qdrant_server_retriever):
    """Test that data persists across operations."""
    retriever = qdrant_server_retriever
    
    # Index document
    await retriever.index(
        doc_id="doc:persistent-001",
        text="This document should persist in the server.",
        page=1,
        confidence=0.92,
    )
    
    info = retriever.get_collection_info()
    initial_count = info["points_count"]
    assert initial_count > 0, "Document should be indexed"
    
    # Search
    results = await retriever.search_by_text("persist", top_k=1)
    assert len(results) > 0
    assert results[0].source_id == "doc:persistent-001"
    
    print(f"✓ Document persisted: {initial_count} points in collection")


@pytest.mark.asyncio
async def test_qdrant_server_multimodal_workflow(qdrant_server_retriever):
    """Test realistic multimodal workflow: index docs, search by text, return Source provenance."""
    retriever = qdrant_server_retriever
    
    # Simulate multimodal document indexing
    documents = [
        {
            "doc_id": "invoice:2024-001",
            "text": "Invoice #2024-001: Total Amount: $5,000.00. Payment Terms: Net 30.",
            "page": 1,
            "confidence": 0.98,
        },
        {
            "doc_id": "invoice:2024-002",
            "text": "Invoice #2024-002: Total Amount: $2,750.50. Payment Terms: Net 45.",
            "page": 1,
            "confidence": 0.96,
        },
        {
            "doc_id": "receipt:2024-Q2",
            "text": "Q2 2024 Receipt Summary. Total received: $7,750.50. Items processed: 15",
            "page": 1,
            "confidence": 0.95,
        },
    ]
    
    # Index all documents
    for doc in documents:
        await retriever.index(
            doc_id=doc["doc_id"],
            text=doc["text"],
            page=doc["page"],
            confidence=doc["confidence"],
        )
    
    info = retriever.get_collection_info()
    assert info["points_count"] == len(documents), "All documents should be indexed"
    
    # Perform multimodal queries
    queries = [
        ("What is the invoice total?", "invoice:2024-001"),
        ("receipt Q2", "receipt:2024-Q2"),
        ("payment terms", "invoice:"),  # Should match either invoice
    ]
    
    for query_text, expected_source in queries:
        results = await retriever.search_by_text(query_text, top_k=1)
        assert len(results) > 0, f"Should find result for: {query_text}"
        
        source = results[0]
        assert isinstance(source, Source)
        assert expected_source in source.source_id
        assert source.snippet is not None
        assert source.confidence is not None
        
        print(f"✓ Query '{query_text}' → {source.source_id} (confidence={source.confidence:.2f})")


@pytest.mark.asyncio
async def test_qdrant_server_large_batch(qdrant_server_retriever):
    """Test indexing a larger batch of documents."""
    retriever = qdrant_server_retriever
    
    # Index 50 documents
    for i in range(50):
        await retriever.index(
            doc_id=f"doc:batch-{i:03d}",
            text=f"Document {i}: This is test content for batch indexing. Keywords: batch, test, document, number {i}.",
            page=i // 10 + 1,
            confidence=0.90 + (i % 10) * 0.01,
        )
    
    info = retriever.get_collection_info()
    assert info["points_count"] == 50, "All 50 documents should be indexed"
    
    # Search
    results = await retriever.search_by_text("batch", top_k=10)
    assert len(results) > 0, "Search should return results"
    
    print(f"✓ Indexed {info['points_count']} documents, search returned {len(results)} results")


@pytest.mark.asyncio
async def test_qdrant_server_clear_and_reuse(qdrant_server_retriever):
    """Test clearing and reusing collection."""
    retriever = qdrant_server_retriever
    
    # Index and clear
    await retriever.index(doc_id="doc:temp-001", text="Temporary content")
    info1 = retriever.get_collection_info()
    assert info1["points_count"] > 0
    
    await retriever.clear()
    info2 = retriever.get_collection_info()
    assert info2["points_count"] == 0, "Collection should be empty after clear"
    
    # Reuse
    await retriever.index(doc_id="doc:new-001", text="New content after clear")
    info3 = retriever.get_collection_info()
    assert info3["points_count"] > 0, "Should be able to index after clear"
    
    print("✓ Collection cleared and reused successfully")
