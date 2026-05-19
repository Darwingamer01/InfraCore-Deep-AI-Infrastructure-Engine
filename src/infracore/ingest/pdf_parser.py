"""PDFParser - Extract text from PDF files using pypdfium2."""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pypdfium2
from prometheus_client import Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field

from infracore.ingest.base import BaseIngester, IngestConfig


class IngestError(Exception):
    """Raised when ingestion fails."""

    pass


@dataclass
class IngestResult:
    """Result of document ingestion."""

    text: str
    meta: dict
    source: str
    chunks: List[str] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


class PDFConfig(IngestConfig):
    """PDF ingestion configuration."""

    model_config = ConfigDict(frozen=True)

    store_type: str = Field(default="pdf", description="Parser type")
    extract_tables: bool = Field(
        default=False, description="Extract tables (reserved for Sprint 4)"
    )
    extract_images: bool = Field(
        default=False, description="Extract images (reserved for Sprint 4)"
    )
    min_page_chars: int = Field(default=50, description="Skip pages with fewer chars")
    normalize_whitespace: bool = Field(
        default=True, description="Collapse multiple spaces"
    )
    page_separator: str = Field(default="\n\n", description="String between pages")


class PDFParser(BaseIngester):
    """PDF text extraction using pypdfium2."""

    def __init__(self, config: PDFConfig):
        super().__init__(config)
        self.config = config

        # Prometheus metrics (unique per parser type)
        self._counter = Counter(
            "pdf_ingest_documents_total",
            "Total PDF documents ingested",
        )
        self._histogram = Histogram(
            "pdf_ingest_latency_seconds",
            "PDF ingestion latency in seconds",
        )

    async def ingest(self, source: str) -> IngestResult:
        """
        Ingest PDF from file path.

        Args:
            source: Path to PDF file

        Returns:
            IngestResult with extracted text and metadata

        Raises:
            IngestError: If file not found or PDF is corrupt
        """
        start = time.time()

        try:
            # Run blocking PDF extraction in thread pool
            result = await asyncio.to_thread(self._extract_pdf, source)

            # Record metrics
            self._counter.inc()
            latency = time.time() - start
            self._histogram.observe(latency)

            return result

        except (FileNotFoundError, IsADirectoryError) as e:
            raise IngestError(f"PDF file not found: {source}") from e
        except Exception as e:
            raise IngestError(f"Failed to ingest PDF {source}: {str(e)}") from e

    def _extract_pdf(self, source: str) -> IngestResult:
        """
        Synchronous PDF extraction (runs in thread pool).

        Args:
            source: Path to PDF file

        Returns:
            IngestResult with extracted text and metadata

        Raises:
            FileNotFoundError: If file does not exist
            Exception: If PDF is corrupt
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            # Open PDF document
            doc = pypdfium2.PdfDocument(source)
            page_count = len(doc)

            pages = []
            skipped_pages = []
            total_chars = 0

            # Extract text page by page
            for page_idx in range(page_count):
                page = doc[page_idx]
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()

                # Normalize whitespace if configured
                if self.config.normalize_whitespace:
                    # Collapse multiple spaces/tabs to single space
                    text = re.sub(r"[ \t]+", " ", text)
                    # Collapse multiple newlines to single newline
                    text = re.sub(r"\n+", "\n", text)

                # Strip text
                text = text.strip()

                # Skip pages below min_page_chars
                if len(text) < self.config.min_page_chars:
                    skipped_pages.append(page_idx)
                    continue

                pages.append(text)
                total_chars += len(text)

            # Join pages with separator
            full_text = self.config.page_separator.join(pages)

            # Build metadata
            meta = {
                "filename": path.name,
                "page_count": page_count,
                "char_count": total_chars,
                "skipped_pages": skipped_pages,
            }

            return IngestResult(text=full_text, meta=meta, source=source)

        except Exception as e:
            raise IngestError(f"Corrupt PDF or read error: {str(e)}") from e
