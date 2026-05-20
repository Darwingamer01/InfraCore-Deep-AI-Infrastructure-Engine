from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from .clip_embedder import BaseImageEmbedder, get_image_embedder
from .ocr import OCRPipeline


class ImageTextRetriever:
    """In-memory image-text retriever for prototyping.

    Stores documents as (id, image_bytes, text, embedding). Embeddings are
    produced on demand by the provided embedder.
    """

    def __init__(self, embedder: BaseImageEmbedder | None = None):
        self.embedder = embedder or get_image_embedder()
        self.ocr = OCRPipeline()
        self._store: Dict[str, Tuple[bytes, str, np.ndarray]] = {}

    async def index(self, doc_id: str, image: bytes, text: str | None = None):
        if text is None:
            ocr_texts = await self.ocr.ocr([image])
            text = ocr_texts[0]
        emb = await self.embedder.embed_images([image])
        self._store[doc_id] = (image, text, emb[0])

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na = a / (np.linalg.norm(a) + 1e-12)
        nb = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(na, nb))

    async def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        # Use the embedder's text encoder when available
        q_emb = (await self.embedder.embed_texts([query_text]))[0]
        scores = []
        for doc_id, (_, _, emb) in self._store.items():
            scores.append((doc_id, self._cosine(q_emb, emb)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
