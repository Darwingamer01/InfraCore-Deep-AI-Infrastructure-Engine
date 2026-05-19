"""
INFRACORE — E5Embedder

intfloat/e5-large-v2 embeddings via sentence-transformers.
Supports query/passage prefixes for retrieval-optimized embeddings.
"""

import time
from typing import List, Literal

import numpy as np
import torch
from prometheus_client import Counter, Histogram
from pydantic import Field
from sentence_transformers import SentenceTransformer

from src.infracore.embedding.base import BaseEmbedder, EmbedConfig

# Prometheus metrics (unique names to avoid conflicts)
e5_embeddings_processed = Counter(
    "e5_embeddings_processed_total",
    "Total embeddings processed by E5",
    ["model_name"],
)
e5_embedding_latency = Histogram(
    "e5_embedding_latency_seconds",
    "E5 embedding latency in seconds",
    ["model_name"],
)


class E5EmbedConfig(EmbedConfig):
    """Extended config for E5 with mode (query/passage)."""

    mode: Literal["query", "passage"] = Field(
        default="passage",
        description="Query or passage mode for E5 embeddings",
    )


class E5Embedder(BaseEmbedder):
    """
    E5-large-v2 embedder using sentence-transformers.

    E5 requires query/passage prefixes for optimal retrieval performance.
    Produces 1024-dimensional embeddings.

    Example:
        # For retrieval queries
        query_config = E5EmbedConfig(model_name="intfloat/e5-large-v2", mode="query")
        embedder = E5Embedder(query_config)
        query_emb = await embedder.embed(["what is machine learning?"])

        # For document passages
        doc_config = E5EmbedConfig(model_name="intfloat/e5-large-v2", mode="passage")
        embedder = E5Embedder(doc_config)
        doc_emb = await embedder.embed(["Machine learning is..."])
    """

    def __init__(self, config: E5EmbedConfig):
        # Convert to E5EmbedConfig if needed
        if not isinstance(config, E5EmbedConfig):
            config = E5EmbedConfig(
                model_name=config.model_name,
                batch_size=config.batch_size,
                max_length=config.max_length,
                normalize=config.normalize,
            )

        super().__init__(config)
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self.embedding_dim = 1024
        self.device = self._detect_device()
        self.model.to(self.device)

    def _detect_device(self) -> str:
        """Detect best available device: cuda > mps > cpu."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _add_prefix(self, texts: List[str]) -> List[str]:
        """Add query or passage prefix to texts."""
        prefix = "query: " if self.config.mode == "query" else "passage: "
        return [prefix + text for text in texts]

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts with appropriate prefix.

        Args:
            texts: List of text strings (prefix added automatically)

        Returns:
            numpy array shape (len(texts), 1024) — L2-normalized embeddings
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        start_time = time.perf_counter()

        # Add prefix
        prefixed_texts = self._add_prefix(texts)
        all_embeddings = []

        # Process in batches
        for i in range(0, len(prefixed_texts), self.config.batch_size):
            batch = prefixed_texts[i : i + self.config.batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)

            # L2 normalize if requested
            if self.config.normalize:
                batch_embeddings = self._normalize_embeddings(batch_embeddings)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

        # Record metrics
        elapsed = time.perf_counter() - start_time
        e5_embeddings_processed.labels(model_name=self.config.model_name).inc(len(texts))
        e5_embedding_latency.labels(model_name=self.config.model_name).observe(elapsed)

        return result.astype(np.float32)

    async def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text (convenience wrapper).

        Args:
            text: Single text string

        Returns:
            numpy array shape (1024,) — L2-normalized embedding
        """
        result = await self.embed([text])
        return result[0] if len(result) > 0 else np.zeros(self.embedding_dim, dtype=np.float32)

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
