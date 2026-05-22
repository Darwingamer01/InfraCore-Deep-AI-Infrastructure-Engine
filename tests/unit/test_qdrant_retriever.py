"""Unit tests for QdrantRetriever using in-memory mode."""

import asyncio
import pytest
from infracore.multimodal import QdrantRetriever, Source


@pytest.fixture
async def qdrant_retriever():
    """Fixture: create in-memory Qdrant retriever for testing."""
    # Don't specify vector_size; let it auto-detect from embedder
    retriever = QdrantRetriever(
        collection_name="test_multimodal",
        url=":memory:",
    )
    yield retriever
    # Cleanup: clear collection after test
    await retriever.clear()


@pytest.mark.asyncio
async def test_qdrant_initialization():
    """Test Qdrant client initialization in in-memory mode."""
    retriever = QdrantRetriever(url=":memory:")
    assert retriever.client is not None
    assert retriever.collection_name == "multimodal_documents"
    assert retriever.url == ":memory:"
    await retriever.clear()


@pytest.mark.asyncio
async def test_qdrant_index_and_search(qdrant_retriever):
    """Test indexing and searching documents in Qdrant."""
    # Index a document
    await qdrant_retriever.index(
        doc_id="doc:invoice-001",
        text="Invoice Total: $5,000.00. Payment due within 30 days.",
        page=1,
        confidence=0.95,
    )
    
    # Verify collection has the document
    info = qdrant_retriever.get_collection_info()
    assert info["points_count"] > 0, "Collection should have at least one point"
    
    # Search by text
    results = await qdrant_retriever.search_by_text(
        query_text="What is the invoice total?",
        top_k=1,
    )
    
    assert len(results) > 0, "Search should return results"
    source = results[0]
    assert isinstance(source, Source), "Result should be a Source object"
    assert source.source_id == "doc:invoice-001"
    assert source.source_type == "retrieved"
    assert source.page == 1
    assert source.confidence == 0.95


@pytest.mark.asyncio
async def test_qdrant_multiple_documents(qdrant_retriever):
    """Test indexing multiple documents and searching."""
    # Index multiple documents
    docs = [
        ("doc:policy-001", "Payment must be made within 30 days."),
        ("doc:invoice-002", "Invoice amount is $2,500.00."),
        ("doc:receipt-003", "Receipt for purchase of office supplies: $150.00"),
    ]
    
    for doc_id, text in docs:
        await qdrant_retriever.index(doc_id=doc_id, text=text)
    
    info = qdrant_retriever.get_collection_info()
    assert info["points_count"] == len(docs), "All documents should be indexed"
    
    # Search for payment-related docs
    results = await qdrant_retriever.search_by_text(
        query_text="payment terms",
        top_k=2,
    )
    
    assert len(results) > 0, "Should find payment-related documents"
    # Top result should be the policy doc
    assert results[0].source_id == "doc:policy-001"


@pytest.mark.asyncio
async def test_qdrant_source_provenance(qdrant_retriever):
    """Test that Source objects carry full provenance metadata."""
    # Index document with full metadata
    bbox = {"x": 10, "y": 20, "width": 100, "height": 50}
    await qdrant_retriever.index(
        doc_id="doc:form-001",
        text="Customer Name: John Doe. Amount Due: $1,000.00",
        page=2,
        bounding_box=bbox,
        confidence=0.88,
    )
    
    # Search and verify provenance
    results = await qdrant_retriever.search_by_text("customer name", top_k=1)
    assert len(results) > 0
    
    source = results[0]
    assert source.source_id == "doc:form-001"
    assert source.page == 2
    assert source.bounding_box == bbox
    assert source.confidence == 0.88
    assert source.snippet is not None
    assert len(source.snippet) > 0


@pytest.mark.asyncio
async def test_qdrant_empty_search(qdrant_retriever):
    """Test search on empty collection returns empty list."""
    results = await qdrant_retriever.search_by_text("nonexistent query", top_k=5)
    assert results == [], "Empty collection should return empty search results"


@pytest.mark.asyncio
async def test_qdrant_clear(qdrant_retriever):
    """Test clearing collection."""
    # Index a document
    await qdrant_retriever.index(
        doc_id="doc:test-001",
        text="Test document",
    )
    
    info = qdrant_retriever.get_collection_info()
    assert info["points_count"] > 0, "Collection should have documents"
    
    # Clear
    await qdrant_retriever.clear()
    
    # Verify empty
    info = qdrant_retriever.get_collection_info()
    assert info["points_count"] == 0, "Collection should be empty after clear"


@pytest.mark.asyncio
async def test_qdrant_snippet_generation(qdrant_retriever):
    """Test that snippets are properly generated from text."""
    # Use text within token limits for CLIP model (max ~77 tokens)
    long_text = ("A sample document with important information. " * 8).strip()[:300]
    
    await qdrant_retriever.index(
        doc_id="doc:long-001",
        text=long_text,
    )
    
    results = await qdrant_retriever.search_by_text("sample", top_k=1)
    assert len(results) > 0, "Should find document"
    
    source = results[0]
    # Snippet should be truncated to 500 chars (or less if original was shorter)
    assert len(source.snippet) <= 500
    assert source.snippet.startswith("A")


@pytest.mark.asyncio
async def test_qdrant_search_with_filter(qdrant_retriever):
    """Test filtering search results by source_type."""
    # Index documents with different source types (simulated via payload)
    await qdrant_retriever.index(
        doc_id="doc:retrieved-001",
        text="Retrieved document text",
    )
    
    # Search with source_type filter
    results = await qdrant_retriever.search_by_text(
        query_text="document",
        top_k=5,
        source_type_filter="retrieved",
    )
    
    assert len(results) > 0
    for source in results:
        assert source.source_type == "retrieved"


@pytest.mark.asyncio
async def test_qdrant_index_without_text(qdrant_retriever):
    """Test that indexing without text logs warning and returns."""
    # Index with no text and no image
    await qdrant_retriever.index(doc_id="doc:empty", text=None)
    
    # Collection should still be empty
    info = qdrant_retriever.get_collection_info()
    assert info["points_count"] == 0


@pytest.mark.asyncio
async def test_qdrant_deterministic_point_ids(qdrant_retriever):
    """Test that point IDs are deterministic for same doc_id."""
    doc_id = "doc:test-001"
    
    # Get point ID twice
    point_id_1 = qdrant_retriever._make_point_id(doc_id, 0)
    point_id_2 = qdrant_retriever._make_point_id(doc_id, 0)
    
    assert point_id_1 == point_id_2, "Point IDs should be deterministic"
    
    # Different indices should produce different IDs
    point_id_3 = qdrant_retriever._make_point_id(doc_id, 1)
    assert point_id_1 != point_id_3, "Different chunk indices should produce different IDs"
