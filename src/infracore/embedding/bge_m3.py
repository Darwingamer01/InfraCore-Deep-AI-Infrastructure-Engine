"""
INFRACORE — BGEEmbedder

BAAI/bge-m3 embeddings via sentence-transformers.
High-quality multilingual embeddings. Batch processing with Prometheus metrics.
"""

import time
from typing import List

import numpy as np
import torch
from prometheus_client import Counter, Histogram
from sentence_transformers import SentenceTransformer

from src.infracore.embedding.base import BaseEmbedder, EmbedConfig

# Prometheus metrics (unique names to avoid conflicts)
bge_embeddings_processed = Counter(
    "bge_embeddings_processed_total",
    "Total embeddings processed by BGE",
    ["model_name"],
)
bge_embedding_latency = Histogram(
    "bge_embedding_latency_seconds",
    "BGE embedding latency in seconds",
    ["model_name"],
)


class BGEEmbedder(BaseEmbedder):
    """
    BGE-M3 embedder using sentence-transformers.

    Produces 1024-dimensional embeddings. Supports batch processing and auto device selection.

    Example:
        config = EmbedConfig(model_name="BAAI/bge-m3", batch_size=32, normalize=True)
        embedder = BGEEmbedder(config)
        embeddings = await embedder.embed(["text 1", "text 2"])
        # embeddings: np.ndarray shape (2, 1024)
    """

    def __init__(self, config: EmbedConfig):
        super().__init__(config)
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

    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array shape (len(texts), 1024) — L2-normalized embeddings
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        start_time = time.perf_counter()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)

            # L2 normalize if requested
            if self.config.normalize:
                batch_embeddings = self._normalize_embeddings(batch_embeddings)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        result = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

        # Record metrics
        elapsed = time.perf_counter() - start_time
        bge_embeddings_processed.labels(model_name=self.config.model_name).inc(len(texts))
        bge_embedding_latency.labels(model_name=self.config.model_name).observe(elapsed)

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
