"""INFRACORE — WeaviateVectorStore

Weaviate vector store implementation using the Weaviate v4 Python async client.
Async-first, typed payloads, result dataclasses, and Prometheus metrics.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from prometheus_client import Counter, Histogram, REGISTRY
from pydantic import Field, ConfigDict

import weaviate
import weaviate.classes as wvc
from weaviate.classes.data import DataObject

from src.infracore.vectordb.base import BaseVectorStore, SearchResult, VectorStoreConfig


def _get_or_create_metric(factory, name: str, description: str, labelnames: List[str]):
    """Return an existing Prometheus metric if the collector was already registered."""
    try:
        return factory(name, description, labelnames)
    except ValueError:
        for candidate_name in (name, name.removesuffix("_total"), f"{name}_created"):
            collector = REGISTRY._names_to_collectors.get(candidate_name)
            if collector is not None:
                return collector
        for collector, names in REGISTRY._collector_to_names.items():
            if name in names or name.removesuffix("_total") in names:
                return collector
        raise


weaviate_search_latency = _get_or_create_metric(
    Histogram,
    "weaviate_search_latency_seconds",
    "Weaviate search latency in seconds",
    ["collection"],
)

weaviate_upsert_throughput = _get_or_create_metric(
    Counter,
    "weaviate_upsert_operations_total",
    "Total Weaviate upsert operations",
    ["collection"],
)


class WeaviateConfig(VectorStoreConfig):
    """Weaviate-specific configuration."""

    model_config = ConfigDict(frozen=True)

    store_type: str = Field(default="weaviate", description="Store type")
    host: str = Field(default="localhost", description="Weaviate host")
    port: int = Field(default=8080, description="Weaviate port")
    distance_metric: str = Field(default="cosine", description="Distance metric")
    top_k: int = Field(default=5, description="Top K search results")


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store with async operations, batching, and metrics."""

    def __init__(self, config: WeaviateConfig):
        super().__init__(config)
        self.config = config
        self.batch_size = 100

    async def _ensure_collection(self, client) -> None:
        """Create collection if it doesn't exist."""
        exists = await client.collections.exists(self.config.collection_name)
        if not exists:
            metric_map = {
                "cosine": wvc.config.VectorDistances.COSINE,
                "dot": wvc.config.VectorDistances.DOT,
                "l2": wvc.config.VectorDistances.L2_SQUARED,
            }
            distance = metric_map.get(
                self.config.distance_metric.lower(), wvc.config.VectorDistances.COSINE
            )

            await client.collections.create(
                name=self.config.collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=distance
                ),
            )

    async def upsert(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Upsert vectors and payloads in batches of 100."""
        N = len(vectors)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(N)]

        async with weaviate.use_async_with_local(
            host=self.config.host, port=self.config.port
        ) as client:
            await self._ensure_collection(client)
            collection = client.collections.get(self.config.collection_name)

            for i in range(0, N, self.batch_size):
                batch_end = min(i + self.batch_size, N)
                batch_vectors = vectors[i:batch_end]
                batch_payloads = payloads[i:batch_end]
                batch_ids = ids[i:batch_end]

                objects = [
                    DataObject(
                        properties=payload,
                        vector=vec.tolist() if isinstance(vec, np.ndarray) else list(vec),
                        uuid=uid,
                    )
                    for vec, payload, uid in zip(batch_vectors, batch_payloads, batch_ids)
                ]

                await collection.data.insert_many(objects)
                weaviate_upsert_throughput.labels(
                    collection=self.config.collection_name
                ).inc(len(objects))

    async def search(
        self, query_vector: np.ndarray, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        limit = top_k if top_k is not None else self.config.top_k
        start_time = time.perf_counter()

        query_vector_list = (
            query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)
        )

        async with weaviate.use_async_with_local(
            host=self.config.host, port=self.config.port
        ) as client:
            await self._ensure_collection(client)
            collection = client.collections.get(self.config.collection_name)

            response = await collection.query.near_vector(
                near_vector=query_vector_list,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True, score=True),
            )

        elapsed = time.perf_counter() - start_time
        weaviate_search_latency.labels(collection=self.config.collection_name).observe(elapsed)

        results = []
        for obj in response.objects:
            score = 1.0
            if obj.metadata is not None:
                if obj.metadata.certainty is not None:
                    score = obj.metadata.certainty
                elif obj.metadata.score is not None:
                    score = obj.metadata.score
                elif obj.metadata.distance is not None:
                    # Cosine distance to similarity: 1 - distance
                    score = 1.0 - obj.metadata.distance

            results.append(
                SearchResult(
                    id=str(obj.uuid),
                    score=float(score),
                    payload=obj.properties or {},
                )
            )

        return results
