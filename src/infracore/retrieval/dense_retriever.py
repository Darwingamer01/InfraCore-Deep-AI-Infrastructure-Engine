"""DenseRetriever - Pure vector similarity retrieval."""

import time
from typing import Optional

from prometheus_client import Counter as PrometheusCounter, Histogram
from pydantic import ConfigDict, Field

from src.infracore.embedding.base import BaseEmbedder
from src.infracore.retrieval.base import BaseRetriever, RetrieverConfig, RetrievalResult
from src.infracore.vectordb.base import BaseVectorStore


class DenseConfig(RetrieverConfig):
    """Dense retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    score_threshold: float = Field(default=0.0, description="Minimum score threshold")


class DenseRetriever(BaseRetriever):
    """Pure dense retrieval using vector similarity."""

    def __init__(
        self,
        config: DenseConfig,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
    ):
        super().__init__(config)
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder

        # Prometheus metrics
        self._counter = PrometheusCounter(
            "dense_retrieval_queries_total",
            "Total dense retrieval queries",
        )
        self._histogram = Histogram(
            "dense_retrieval_latency_seconds",
            "Dense retrieval latency",
        )

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """
        Retrieve using vector similarity search.

        Args:
            query: Search query
            top_k: Number of results (uses config.top_k if None)

        Returns:
            List of RetrievalResult sorted by score descending
        """
        if top_k is None:
            top_k = self.config.top_k

        start = time.time()

        try:
            # Embed query
            query_embedding = await self.embedder.embed_single(query)

            # Search vector store
            search_results = await self.vector_store.search(
                query_vector=query_embedding.reshape(1, -1),
                top_k=top_k,
            )

            # Convert to RetrievalResult
            results = []
            for result in search_results:
                if result.score < self.config.score_threshold:
                    continue

                text = result.payload.get("text", "") if result.payload else ""

                results.append(
                    RetrievalResult(
                        text=text,
                        score=result.score,
                        metadata={
                            "doc_id": result.id,
                            "retrieval_method": "dense",
                            **(result.payload or {}),
                        },
                    )
                )

            # Record metrics
            self._counter.inc()
            latency = time.time() - start
            self._histogram.observe(latency)

            return results

        except Exception as e:
            raise Exception(f"Dense retrieval failed: {str(e)}") from e
