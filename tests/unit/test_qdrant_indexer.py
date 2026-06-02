import asyncio
import pytest

from qdrant_client import QdrantClient

from infracore.multimodal.qdrant_indexer import QdrantMultimodalIndexer


class DummyEmbedder:
    async def embed_texts(self, texts):
        # deterministic vector matching default indexer size
        return [[0.2] * 128 for _ in texts]

    async def embed_images(self, images):
        return [[0.2] * 128 for _ in images]


@pytest.mark.asyncio
async def test_qdrant_indexer_text_and_image_search():
    client = QdrantClient(":memory:")
    embedder = DummyEmbedder()
    indexer = QdrantMultimodalIndexer(collection_name="test_idx", embedder=embedder, client=client)

    # index a text document
    await indexer.index_point(doc_id="doc1", text="The quick brown fox jumps", page=1, bounding_box={"left": 1}, confidence=0.9)

    # index a second document
    await indexer.index_point(doc_id="doc2", text="Cats and other small animals", page=2, confidence=0.75)

    # search by text
    text_results = await indexer.search_by_text("cats", top_k=2)
    assert len(text_results) >= 1
    ids = {s.source_id for s in text_results}
    assert "doc2" in ids

    # search by image (dummy image bytes)
    img_results = await indexer.search_by_image(b"\x00\x01\x02", top_k=2)
    assert isinstance(img_results, list)
    assert len(img_results) >= 1


@pytest.mark.asyncio
async def test_qdrant_indexer_bbox_mapping():
    client = QdrantClient(":memory:")
    embedder = DummyEmbedder()
    indexer = QdrantMultimodalIndexer(collection_name="test_idx_bbox", embedder=embedder, client=client)

    test_bbox = {"left": 10, "top": 20, "width": 100, "height": 50}

    # Index using bbox parameter
    await indexer.index_point(doc_id="doc_bbox", text="Text with bbox", bbox=test_bbox)

    # Index using bounding_box parameter
    await indexer.index_point(doc_id="doc_bounding_box", text="Text with bounding_box", bounding_box=test_bbox)

    # Inspect Qdrant client payload directly
    points, _ = client.scroll(collection_name="test_idx_bbox", with_payload=True)
    assert len(points) == 2

    for p in points:
        payload = p.payload
        assert "bbox" in payload
        assert "bounding_box" in payload
        assert payload["bbox"] == test_bbox
        assert payload["bounding_box"] == test_bbox

    # Search and verify Source mapping
    results = await indexer.search_by_text("Text", top_k=2)
    assert len(results) == 2
    for s in results:
        assert s.bounding_box == test_bbox

