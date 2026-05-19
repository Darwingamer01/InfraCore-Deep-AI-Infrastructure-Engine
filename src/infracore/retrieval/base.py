"""
BaseRetriever – Abstract interface for retrieval (combining chunking + embedding + search).

Pure async, typed configs and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class RetrieverConfig(BaseModel):
    """Pydantic config for retrieval."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=10, description="Number of results to return")
    score_threshold: float = Field(default=0.0, description="Min score to include")
    rerank: bool = Field(default=False, description="Apply reranking")


@dataclass
class RetrievalResult:
    """Single retrieval result — a chunk + score."""

    text: str
    score: float
    metadata: dict


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval.

    Subclasses must implement:
    - retrieve(query: str) -> List[RetrievalResult]

    Handles the full pipeline: query → embedding → search → results
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config

    @abstractmethod
    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: Natural language query

        Returns:
            Sorted list of RetrievalResult (best first)
        """
        pass
