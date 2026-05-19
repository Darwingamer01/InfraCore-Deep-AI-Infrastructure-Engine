"""
BaseEmbedder – Abstract interface for text embedding.

Pure async, batch processing, numpy output, Pydantic config.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EmbedConfig(BaseModel):
    """Pydantic config for embedding models."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(..., description="HuggingFace model ID or path")
    batch_size: int = Field(default=32, description="Batch size for inference")
    max_length: int = Field(default=512, description="Max sequence length")
    normalize: bool = Field(default=True, description="L2 normalize embeddings")


class BaseEmbedder(ABC):
    """
    Abstract base class for text embeddings.

    Subclasses must implement:
    - embed(texts: List[str]) -> np.ndarray

    All embedders are async. Output shape is (N, dim).
    """

    def __init__(self, config: EmbedConfig):
        self.config = config
        self.embedding_dim: int = 0  # Subclass must set

    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array shape (len(texts), embedding_dim)
        """
        pass
