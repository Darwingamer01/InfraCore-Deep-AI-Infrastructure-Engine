"""
BaseChunker – Abstract interface for all text chunking strategies.

Every chunker must implement the chunk() method and expose its config.
No LangChain imports. Pure async, typed Pydantic configs, result dataclasses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ChunkConfig(BaseModel):
    """Pydantic config for chunking strategies."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(..., description="Chunking strategy: fixed, semantic, recursive, late")
    max_tokens: int = Field(default=512, description="Max tokens per chunk")
    overlap: int = Field(default=0, description="Token overlap between chunks")
    min_chunk_size: int = Field(default=50, description="Min tokens per chunk")


@dataclass
class Chunk:
    """Result of chunking — a single text segment with metadata."""

    text: str
    start_idx: int
    end_idx: int
    metadata: dict = field(default_factory=dict)


class BaseChunker(ABC):
    """
    Abstract base class for text chunking.

    Subclasses must implement:
    - chunk(text: str) -> List[Chunk]

    All chunkers are async. No blocking I/O.
    """

    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    async def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text into segments.

        Args:
            text: Raw text to chunk

        Returns:
            List of Chunk objects with start_idx, end_idx, metadata
        """
        pass
