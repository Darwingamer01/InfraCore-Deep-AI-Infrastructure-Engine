"""Cross-Encoder Reranking Module.

Rerank retrieved documents using semantic similarity scores from a cross-encoder model.
Pure async-first, lazy loading, Pydantic configuration, and Prometheus metrics.
"""

import asyncio
from copy import deepcopy
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.infracore.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class RerankerConfig(BaseModel):
    """Configuration for CrossEncoderReranker."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(
        default="BAAI/bge-reranker-large",
        description="HuggingFace model ID or path for the cross-encoder reranker",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of top results to return (top_k equivalent)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold to include a result after reranking",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for query-passage scoring inference",
    )


# Prometheus metrics
reranker_operations_total = Counter(
    "reranker_operations_total",
    "Total reranking operations completed",
    ["model_name"],
)

reranker_latency_seconds = Histogram(
    "reranker_latency_seconds",
    "Reranking latency in seconds",
    ["model_name"],
)


class CrossEncoderReranker:
    """Reranker using CrossEncoder model for deep query-document relevance scoring."""

    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize with configuration."""
        self.config = config or RerankerConfig()
        self.model = None

    def _load_model(self) -> None:
        """Lazy load the CrossEncoder model."""
        if self.model is not None:
            return

        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Please install it via: pip install sentence-transformers"
            )

        self.model = CrossEncoder(self.config.model_name)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Rerank a list of candidates against the query asynchronously.

        Args:
            query: User search query.
            candidates: List of retrieval candidates.

        Returns:
            List of RetrievalResult sorted by reranker score descending,
            with score stored under metadata["reranker_score"].
        """
        if not candidates:
            return []

        self._load_model()
        start_time = time.perf_counter()

        # Prepare query-document pairs
        pairs = []
        valid_candidates = []
        for candidate in candidates:
            if candidate.text is not None:
                pairs.append([query, candidate.text])
                valid_candidates.append(candidate)

        if not pairs:
            return []

        # Predict relevance scores in threadpool
        scores = await asyncio.to_thread(
            self._score_pairs,
            pairs,
            self.config.batch_size,
        )

        # Create new results without mutating original objects
        results = []
        for candidate, score in zip(valid_candidates, scores):
            meta = deepcopy(candidate.metadata) if candidate.metadata is not None else {}
            meta["reranker_score"] = float(score)

            reranked = RetrievalResult(
                doc_id=candidate.doc_id,
                text=candidate.text,
                source=candidate.source,
                score=candidate.score,  # Keep original score
                metadata=meta,
            )
            results.append(reranked)

        # Sort results by reranker score descending
        results = sorted(
            results,
            key=lambda r: r.metadata.get("reranker_score", 0.0),
            reverse=True,
        )

        # Filter by threshold if configured
        if self.config.score_threshold is not None:
            results = [
                r for r in results
                if r.metadata.get("reranker_score", 0.0) >= self.config.score_threshold
            ]

        # Respect top_n truncation if configured
        if self.config.top_n is not None:
            results = results[: self.config.top_n]

        # Record metrics
        elapsed = time.perf_counter() - start_time
        reranker_operations_total.labels(model_name=self.config.model_name).inc()
        reranker_latency_seconds.labels(model_name=self.config.model_name).observe(elapsed)

        return results

    def _score_pairs(self, pairs: List[List[str]], batch_size: int) -> np.ndarray:
        """Run batch prediction on query-doc pairs and map using sigmoid."""
        raw_scores = self.model.predict(pairs, batch_size=batch_size)
        # Handle single scoring return type variance
        if isinstance(raw_scores, (int, float)):
            raw_scores = np.array([raw_scores])
        
        # Apply sigmoid function for scaling score to [0, 1]
        return 1.0 / (1.0 + np.exp(-raw_scores))
