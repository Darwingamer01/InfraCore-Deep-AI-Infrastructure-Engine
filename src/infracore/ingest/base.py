"""
BaseIngester – Abstract interface for document ingestion.

Handles PDFs, docx, markdown, HTML. Returns structured documents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class IngestConfig(BaseModel):
    """Pydantic config for ingestion."""

    model_config = ConfigDict(frozen=True)

    supported_formats: List[str] = Field(
        default=["pdf", "docx", "md", "html", "txt"],
        description="Supported file formats",
    )
    extract_metadata: bool = Field(default=True, description="Extract metadata")
    preserve_tables: bool = Field(default=True, description="Extract tables as-is")


@dataclass
class IngestedDocument:
    """Single ingested document."""

    text: str
    source: str
    metadata: dict
    format: str


class BaseIngester(ABC):
    """
    Abstract base class for document ingestion.

    Subclasses must implement:
    - ingest(file_path: str) -> IngestedDocument
    """

    def __init__(self, config: IngestConfig):
        self.config = config

    @abstractmethod
    async def ingest(self, file_path: str) -> IngestedDocument:
        """
        Ingest a document file.

        Args:
            file_path: Path to document

        Returns:
            IngestedDocument with text, metadata, source info
        """
        pass
