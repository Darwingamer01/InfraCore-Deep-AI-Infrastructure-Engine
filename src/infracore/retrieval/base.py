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
    """Single retrieval result — flexible fields used across tests.

    Supports construction with either explicit `doc_id`/`source`/`score`
    kwargs or the older `metadata` dict form. `__post_init__` copies
    values from `metadata` when present to maintain backward compatibility.
    """

    doc_id: str | None = None
    text: str | None = None
    source: str | None = None
    score: float | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata:
            if self.doc_id is None and "doc_id" in self.metadata:
                self.doc_id = self.metadata.get("doc_id")
            if self.source is None and "source" in self.metadata:
                self.source = self.metadata.get("source")
            if self.score is None and "score" in self.metadata:
                self.score = self.metadata.get("score")


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
