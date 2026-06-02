import asyncio
import pytest

from qdrant_client import QdrantClient

from infracore.multimodal.qdrant_indexer import QdrantMultimodalIndexer


class DummyEmbedder:
    async def embed_texts(self, texts):
        # deterministic small vector
        return [[0.2] * 16 for _ in texts]

    async def embed_images(self, images):
        return [[0.2] * 16 for _ in images]


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
