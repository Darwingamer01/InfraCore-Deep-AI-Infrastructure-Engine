"""QdrantMultimodalIndexer: stores OCR text + image embeddings in a single collection.

Designed for tests and production: accepts an embedder with async methods
`embed_texts` and `embed_images`. Uses an in-memory Qdrant client when
`url=":memory:"` or when a `client` is provided (useful for unit tests).

Each point payload includes: source_type, doc_id, page, bounding_box, snippet, confidence
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:  # pragma: no cover - dependency error surfaces at import time
    raise

import numpy as np

from .vlm import Source

logger = logging.getLogger(__name__)


class QdrantMultimodalIndexer:
    def __init__(
        self,
        collection_name: str = "multimodal_index",
        embedder: Any | None = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: Optional[int] = None,
        client: Optional[QdrantClient] = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder
        self.url = url or os.getenv("QDRANT_URL", ":memory:")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self._vector_size_override = vector_size
        self.vector_size = vector_size or 128
        self._vector_size_detected = False

        if client is not None:
            self.client = client
        else:
            if self.url == ":memory:":
                logger.info("Using Qdrant in-memory mode for indexer")
                self.client = QdrantClient(":memory:")
            elif self.url.startswith("http://") or self.url.startswith("https://"):
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self.client = QdrantClient(path=self.url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collection = self.client.get_collection(self.collection_name)
            # If collection exists but vector size differs, recreate it
            try:
                existing_size = collection.config.vectors.size
                if existing_size != self.vector_size:
                    logger.info(
                        "Collection '%s' exists with size %d but expected %d; recreating",
                        self.collection_name,
                        existing_size,
                        self.vector_size,
                    )
                    try:
                        self.client.delete_collection(self.collection_name)
                    except Exception:
                        pass
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                    )
            except Exception:
                # If we can't introspect, ensure a collection exists
                pass
        except Exception:
            logger.info("Creating collection '%s' with vector size %d", self.collection_name, self.vector_size)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _make_point_id(self, doc_id: str, chunk_idx: int = 0) -> int:
        combined = f"{doc_id}#{chunk_idx}".encode()
        return int(hash(combined) % (2**63 - 1))

    async def _auto_detect_vector_size(self) -> None:
        if self._vector_size_detected:
            return
        if self.embedder is None:
            return
        try:
            sample = (await self.embedder.embed_texts(["test"]))[0]
            detected = len(sample)
            if self._vector_size_override is None:
                self.vector_size = detected
            self._vector_size_detected = True
            # recreate if mismatch
            try:
                col = self.client.get_collection(self.collection_name)
                if col.config.vectors.size != self.vector_size:
                    self.client.delete_collection(self.collection_name)
                    self._ensure_collection()
            except Exception:
                self._ensure_collection()
        except Exception as e:
            logger.debug("Auto-detect vector size failed: %s", e)

    async def index_point(
        self,
        doc_id: str,
        text: Optional[str] = None,
        image: Optional[bytes] = None,
        source_type: str = "ocr",
        page: Optional[int] = None,
        bounding_box: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        bbox: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index text and/or image for a document into a single collection."""
        await self._auto_detect_vector_size()

        if not text and not image:
            logger.warning("Skipping index_point: no text/image for doc_id=%s", doc_id)
            return

        # Prefer text embedding when text is present
        try:
            if text and hasattr(self.embedder, "embed_texts"):
                emb = await self.embedder.embed_texts([text])
                vector = emb[0]
            elif image and hasattr(self.embedder, "embed_images"):
                emb = await self.embedder.embed_images([image])
                vector = emb[0]
            else:
                logger.warning("No suitable embedder method for indexing doc_id=%s", doc_id)
                return
        except Exception as e:
            logger.error("Embedding failed for doc_id=%s: %s", doc_id, e)
            return

        try:
            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
        except Exception:
            vector_list = list(vector)

        # Ensure collection matches actual vector size (recover if constructor created a different size)
        try:
            actual_len = len(vector_list)
            if self._vector_size_override is None:
                self.vector_size = actual_len
        except Exception:
            actual_len = self.vector_size

        # Ensure collection matches actual vector size before upsert; recreate if mismatch
        try:
            try:
                col = self.client.get_collection(self.collection_name)
                existing_size = col.config.vectors.size
                if existing_size != actual_len:
                    try:
                        self.client.delete_collection(self.collection_name)
                    except Exception:
                        pass
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=actual_len, distance=Distance.COSINE),
                    )
            except Exception:
                # collection missing or introspection failed
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=actual_len, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.debug("Failed to ensure proper collection size pre-upsert: %s", e)

        payload: Dict[str, Any] = {
            "source_type": source_type,
            "doc_id": doc_id,
            "snippet": (text or "")[:500],
        }
        if page is not None:
            payload["page"] = page
        
        actual_bbox = bbox or bounding_box
        if actual_bbox is not None:
            payload["bounding_box"] = actual_bbox
            payload["bbox"] = actual_bbox

        if confidence is not None:
            payload["confidence"] = confidence

        point = PointStruct(id=self._make_point_id(doc_id, 0), vector=vector_list, payload=payload)

        # Try upsert, recreate collection on dimension mismatch
        try:
            self.client.upsert(collection_name=self.collection_name, points=[point])
        except Exception as e:
            msg = str(e)
            if "expected dim" in msg or "Vector dimension" in msg:
                # Try to recover by recreating collection with target size
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=len(vector_list), distance=Distance.COSINE),
                )
                self.client.upsert(collection_name=self.collection_name, points=[point])
            else:
                raise

    async def search_by_text(self, query_text: str, top_k: int = 5) -> List[Source]:
        try:
            q_emb = await self.embedder.embed_texts([query_text])
            q_vec = q_emb[0]
        except Exception as e:
            logger.error("Failed to embed query text: %s", e)
            return []

        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec.tolist() if isinstance(q_vec, np.ndarray) else q_vec,
                limit=max(top_k, 10),
            )
            points = res.points
        except Exception as e:
            logger.error("Qdrant text search failed: %s", e)
            return []

        sources: List[Source] = []
        for p in points:
            payload = p.payload or {}
            source = Source(
                source_type=payload.get("source_type", "retrieved"),
                source_id=payload.get("doc_id", "unknown"),
                snippet=payload.get("snippet", ""),
                page=payload.get("page"),
                bounding_box=payload.get("bounding_box") or payload.get("bbox"),
                confidence=payload.get("confidence", p.score),
            )
            sources.append(source)

        return sources[:top_k]

    async def search_by_image(self, image: bytes, top_k: int = 5) -> List[Source]:
        try:
            q_emb = await self.embedder.embed_images([image])
            q_vec = q_emb[0]
        except Exception as e:
            logger.error("Failed to embed query image: %s", e)
            return []

        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec.tolist() if isinstance(q_vec, np.ndarray) else q_vec,
                limit=max(top_k, 10),
            )
            points = res.points
        except Exception as e:
            logger.error("Qdrant image search failed: %s", e)
            return []

        sources: List[Source] = []
        for p in points:
            payload = p.payload or {}
            source = Source(
                source_type=payload.get("source_type", "retrieved"),
                source_id=payload.get("doc_id", "unknown"),
                snippet=payload.get("snippet", ""),
                page=payload.get("page"),
                bounding_box=payload.get("bounding_box") or payload.get("bbox"),
                confidence=payload.get("confidence", p.score),
            )
            sources.append(source)

        return sources[:top_k]

    async def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
        except Exception as e:
            logger.error("Failed to clear collection: %s", e)
