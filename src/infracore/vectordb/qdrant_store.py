"""
INFRACORE — QdrantVectorStore

Qdrant vector store implementation using AsyncQdrantClient.
Production-grade with batching, metrics, and async operations.
"""

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

from src.infracore.vectordb.base import BaseVectorStore, SearchResult, VectorStoreConfig

# Prometheus metrics
vectordb_operations = Counter(
    "vectordb_operations_total",
    "Total vectordb operations",
    ["operation", "collection"],
)
vectordb_latency = Histogram(
    "vectordb_latency_seconds",
    "VectorDB operation latency",
    ["operation", "collection"],
)


class QdrantConfig(VectorStoreConfig):
    """Qdrant-specific configuration."""

    model_config = ConfigDict(frozen=True)

    store_type: str = Field(default="qdrant", description="Store type")
    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    hnsw_m: int = Field(default=16, description="HNSW M parameter")
    hnsw_ef_construct: int = Field(default=200, description="HNSW ef_construct")
    hnsw_ef: int = Field(default=128, description="HNSW ef search parameter")
    on_disk: bool = Field(default=False, description="Use on-disk storage")
    quantization: Optional[str] = Field(
        default=None, description="Quantization: None, 'scalar', 'product'"
    )


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store with async operations, batching, and metrics.

    Config:
        host: Qdrant server host (default: localhost)
        port: Qdrant server port (default: 6333)
        collection_name: Collection name
        vector_size: Embedding dimension
        distance_metric: Distance metric (cosine, euclidean, dot)
        hnsw_m, hnsw_ef_construct, hnsw_ef: HNSW parameters

    Example:
        cfg = QdrantConfig(collection_name="documents", vector_size=1024)
        store = QdrantVectorStore(cfg)
        await store.create_collection()
        await store.upsert(vectors, payloads)
        results = await store.search(query_vector, top_k=5)
    """

    def __init__(self, config: QdrantConfig):
        super().__init__(config)
        self.config = config
        self.client = AsyncQdrantClient(host=config.host, port=config.port)
        self.batch_size = 100

    async def create_collection(self) -> None:
        """Create collection if not exists with HNSW configuration."""
        import time

        start = time.perf_counter()

        # Map distance metric to Qdrant enum
        distance_map = {"cosine": Distance.COSINE, "euclidean": Distance.EUCLID, "dot": Distance.DOT}
        distance = distance_map.get(self.config.distance_metric, Distance.COSINE)

        try:
            await self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance,
                ),
                hnsw_config=HnswConfigDiff(
                    m=self.config.hnsw_m,
                    ef_construct=self.config.hnsw_ef_construct,
                    on_disk=self.config.on_disk,
                ),
            )
        except Exception as e:
            # Collection might already exist, which is fine
            if "already exists" not in str(e):
                raise

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(
            operation="create_collection", collection=self.config.collection_name
        ).inc()
        vectordb_latency.labels(
            operation="create_collection", collection=self.config.collection_name
        ).observe(elapsed)

    async def delete_collection(self) -> None:
        """Delete collection (for cleanup/testing)."""
        import time

        start = time.perf_counter()

        try:
            await self.client.delete_collection(collection_name=self.config.collection_name)
        except Exception:
            # Collection might not exist, which is fine
            pass

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(
            operation="delete_collection", collection=self.config.collection_name
        ).inc()
        vectordb_latency.labels(
            operation="delete_collection", collection=self.config.collection_name
        ).observe(elapsed)

    async def upsert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert vectors and payloads in batches.

        Args:
            vectors: shape (N, vector_size)
            payloads: metadata for each vector (length N)
            ids: optional IDs; auto-generated if None

        Returns:
            Count of upserted vectors
        """
        import time

        start = time.perf_counter()

        N = len(vectors)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(N)]

        # Upsert in batches
        total_upserted = 0
        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            batch_vectors = vectors[i:batch_end]
            batch_payloads = payloads[i:batch_end]
            batch_ids = ids[i:batch_end]

            # Convert to PointStruct list
            points = [
                PointStruct(id=id_, vector=vec.tolist(), payload=payload)
                for id_, vec, payload in zip(batch_ids, batch_vectors, batch_payloads)
            ]

            # Upsert batch
            await self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
            total_upserted += len(points)

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(operation="upsert", collection=self.config.collection_name).inc()
        vectordb_latency.labels(operation="upsert", collection=self.config.collection_name).observe(
            elapsed
        )

        return total_upserted

    async def search(
        self, query_vector: np.ndarray, top_k: int = 10, filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: shape (vector_size,)
            top_k: number of results
            filter: optional Qdrant filter dict

        Returns:
            List[SearchResult] sorted by score (best first)
        """
        import time

        start = time.perf_counter()

        # Search in Qdrant
        search_results = await self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=filter,
            limit=top_k,
        )

        # Convert to SearchResult dataclass
        results = [
            SearchResult(
                id=str(result.id),
                score=result.score,
                payload=result.payload or {},
            )
            for result in search_results
        ]

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(operation="search", collection=self.config.collection_name).inc()
        vectordb_latency.labels(operation="search", collection=self.config.collection_name).observe(
            elapsed
        )

        return results

    async def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by ID.

        Args:
            ids: list of vector IDs to delete

        Returns:
            Count of deleted vectors
        """
        import time

        start = time.perf_counter()

        if not ids:
            return 0

        # Convert string IDs to integers (Qdrant uses uint64)
        int_ids = [int(id_) if id_.isdigit() else hash(id_) % (2**63) for id_ in ids]

        await self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=int_ids,
        )

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(operation="delete", collection=self.config.collection_name).inc()
        vectordb_latency.labels(operation="delete", collection=self.config.collection_name).observe(
            elapsed
        )

        return len(ids)

    async def count(self) -> int:
        """Return total vector count in collection."""
        import time

        start = time.perf_counter()

        collection_info = await self.client.get_collection(self.config.collection_name)
        count = collection_info.points_count

        elapsed = time.perf_counter() - start
        vectordb_operations.labels(operation="count", collection=self.config.collection_name).inc()
        vectordb_latency.labels(operation="count", collection=self.config.collection_name).observe(
            elapsed
        )

        return count
