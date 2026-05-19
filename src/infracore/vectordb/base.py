"""
BaseVectorStore – Abstract interface for vector databases.

Pure async, typed payloads, result dataclasses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class VectorStoreConfig(BaseModel):
    """Pydantic config for vector store."""

    model_config = ConfigDict(frozen=True)

    store_type: str = Field(..., description="VectorDB type: qdrant, weaviate, pgvector")
    collection_name: str = Field(default="infracore", description="Collection/index name")
    vector_size: int = Field(..., description="Embedding dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric")


@dataclass
class SearchResult:
    """Single result from vector search."""

    payload: Dict[str, Any]
    score: float
    id: str


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    Subclasses must implement:
    - upsert(vectors: np.ndarray, payloads: List[dict], ids: List[str]) -> None
    - search(query_vector: np.ndarray, top_k: int) -> List[SearchResult]
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config

    @abstractmethod
    async def upsert(
        self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]
    ) -> None:
        """
        Upsert vectors and payloads.

        Args:
            vectors: shape (N, vector_size)
            payloads: metadata for each vector
            ids: unique IDs for each vector
        """
        pass

    @abstractmethod
    async def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: shape (vector_size,)
            top_k: number of results

        Returns:
            Sorted list of SearchResult (best first)
        """
        pass
