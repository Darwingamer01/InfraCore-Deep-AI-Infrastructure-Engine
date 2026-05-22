"""Qdrant-backed multimodal retriever with local and remote modes.

Supports:
- Local mode: in-memory (:memory:) or path-based persistence
- Remote mode: URL + API key for production Qdrant servers
- Rich payload metadata: source_type, doc_id, page, bbox, snippet, confidence
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except ImportError:
    raise ImportError("qdrant-client is required for QdrantRetriever. Install with: pip install qdrant-client")

import numpy as np

from .clip_embedder import BaseImageEmbedder, get_image_embedder
from .ocr import OCRPipeline
from .vlm import Source

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """Qdrant-backed multimodal retriever.
    
    Modes:
    - Local in-memory: url=":memory:"
    - Local path: url="/path/to/qdrant"
    - Remote: url="http://localhost:6333", api_key="..."
    """

    def __init__(
        self,
        collection_name: str = "multimodal_documents",
        embedder: BaseImageEmbedder | None = None,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int | None = None,  # Auto-detect if None
    ):
        """Initialize Qdrant retriever.
        
        Args:
            collection_name: Qdrant collection name
            embedder: Image/text embedder (default: CLIP)
            url: Qdrant URL (":memory:" for in-memory, "/path" for local, "http://..." for remote)
            api_key: API key for remote Qdrant (optional)
            vector_size: Embedding vector dimensionality (auto-detected if None)
        """
        self.collection_name = collection_name
        self.embedder = embedder or get_image_embedder()
        self.ocr = OCRPipeline()
        self._vector_size_override = vector_size
        self.vector_size = vector_size or 512  # Default; will be auto-detected on first embed
        self._vector_size_detected = False
        self.url = url or os.getenv("QDRANT_URL", ":memory:")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize Qdrant client
        self._init_client()
        self._ensure_collection()

    def _init_client(self):
        """Initialize Qdrant client based on URL."""
        if self.url == ":memory:":
            logger.info("Using Qdrant in-memory mode")
            self.client = QdrantClient(":memory:")
        elif self.url.startswith("http://") or self.url.startswith("https://"):
            logger.info("Using Qdrant remote mode: %s", self.url)
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            logger.info("Using Qdrant local path mode: %s", self.url)
            self.client = QdrantClient(path=self.url)

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.debug("Collection '%s' already exists", self.collection_name)
        except Exception:
            logger.info("Creating collection '%s' with vector size %d", self.collection_name, self.vector_size)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    async def _auto_detect_vector_size(self):
        """Auto-detect embedding vector size from embedder on first use."""
        if self._vector_size_detected:
            return
        
        try:
            # Get one sample embedding to determine dimensions
            sample_embedding = (await self.embedder.embed_texts(["test"]))[0]
            detected_size = len(sample_embedding)
            
            if self._vector_size_override is None:
                self.vector_size = detected_size
                logger.info("Auto-detected vector size: %d", self.vector_size)
            
            self._vector_size_detected = True
            
            # Recreate collection if size mismatch
            try:
                collection = self.client.get_collection(self.collection_name)
                if collection.config.vectors.size != self.vector_size:
                    logger.info("Vector size mismatch (%d != %d), recreating collection", 
                               collection.config.vectors.size, self.vector_size)
                    self.client.delete_collection(self.collection_name)
                    self._ensure_collection()
            except Exception:
                self._ensure_collection()
        except Exception as e:
            logger.error("Failed to auto-detect vector size: %s", e)

    def _make_point_id(self, doc_id: str, chunk_idx: int = 0) -> int:
        """Create a unique point ID from doc_id and chunk index.
        
        Uses hash to ensure deterministic IDs while staying within uint64 range.
        """
        combined = f"{doc_id}#{chunk_idx}".encode()
        return int(hash(combined) % (2**63 - 1))

    async def index(
        self,
        doc_id: str,
        image: bytes | None = None,
        text: str | None = None,
        page: int | None = None,
        bounding_box: Dict[str, Any] | None = None,
        confidence: float | None = None,
    ):
        """Index a document with optional image and text.
        
        Args:
            doc_id: Unique document identifier
            image: Image bytes (optional)
            text: Document text (optional; if not provided, extracted from image via OCR)
            page: Page number (for multi-page documents)
            bounding_box: Bounding box metadata
            confidence: Confidence score (e.g., from OCR)
        """
        # Auto-detect vector size on first call
        await self._auto_detect_vector_size()
        
        # Extract text from image if not provided
        if text is None and image is not None:
            ocr_results = await self.ocr.ocr([image])
            text = ocr_results[0].text if ocr_results and ocr_results[0].success else ""
        
        if not text:
            logger.warning("No text to index for doc_id=%s", doc_id)
            return

        # Generate embedding using text encoder
        try:
            embeddings = await self.embedder.embed_texts([text])
            embedding = embeddings[0]
        except Exception as e:
            logger.error("Failed to embed text for doc_id=%s: %s", doc_id, e)
            return

        # Create Qdrant point with rich payload
        point_id = self._make_point_id(doc_id, 0)
        payload: Dict[str, Any] = {
            "source_type": "retrieved",
            "doc_id": doc_id,
            "text": text,
            "snippet": text[:500],  # First 500 chars as snippet
        }
        
        if page is not None:
            payload["page"] = page
        if bounding_box is not None:
            payload["bounding_box"] = bounding_box
        if confidence is not None:
            payload["confidence"] = confidence

        # Upsert point into Qdrant
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            payload=payload,
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.debug("Indexed doc_id=%s, point_id=%s", doc_id, point_id)

    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        source_type_filter: str | None = None,
    ) -> List[Source]:
        """Search by text query and return Source objects.
        
        Args:
            query_text: Query text
            top_k: Number of results
            source_type_filter: Filter by source_type (optional)
        
        Returns:
            List of Source objects with full provenance
        """
        # Generate query embedding
        try:
            query_embeddings = await self.embedder.embed_texts([query_text])
            query_vector = query_embeddings[0]
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            return []

        # Build filter if source_type provided
        query_filter = None
        if source_type_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value=source_type_filter),
                    )
                ]
            )

        # Search Qdrant using query_points (compatible with both in-memory and remote)
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
            results = result.points
        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            return []

        # Convert search results to Source objects
        sources: List[Source] = []
        for scored_point in results:
            payload = scored_point.payload or {}
            source = Source(
                source_type=payload.get("source_type", "retrieved"),
                source_id=payload.get("doc_id", "unknown"),
                snippet=payload.get("snippet", ""),
                page=payload.get("page"),
                bounding_box=payload.get("bounding_box"),
                confidence=payload.get("confidence", scored_point.score),
            )
            sources.append(source)

        logger.debug("Search returned %d sources for query: %s", len(sources), query_text[:50])
        return sources

    async def search_by_image(
        self,
        image: bytes,
        top_k: int = 5,
    ) -> List[Source]:
        """Search by image and return Source objects.
        
        Args:
            image: Image bytes
            top_k: Number of results
        
        Returns:
            List of Source objects with full provenance
        """
        # Generate image embedding
        try:
            embeddings = await self.embedder.embed_images([image])
            query_vector = embeddings[0]
        except Exception as e:
            logger.error("Failed to embed image: %s", e)
            return []

        # Search Qdrant
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                limit=top_k,
            )
            results = result.points
        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            return []

        # Convert to Source objects
        sources: List[Source] = []
        for scored_point in results:
            payload = scored_point.payload or {}
            source = Source(
                source_type=payload.get("source_type", "retrieved"),
                source_id=payload.get("doc_id", "unknown"),
                snippet=payload.get("snippet", ""),
                page=payload.get("page"),
                bounding_box=payload.get("bounding_box"),
                confidence=payload.get("confidence", scored_point.score),
            )
            sources.append(source)

        logger.debug("Image search returned %d sources", len(sources))
        return sources

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection.points_count,
                "vectors_config": {
                    "size": self.vector_size,
                    "distance": "COSINE",
                },
            }
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {}

    async def clear(self):
        """Delete all points in the collection (for testing)."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("Cleared collection '%s'", self.collection_name)
        except Exception as e:
            logger.error("Failed to clear collection: %s", e)
