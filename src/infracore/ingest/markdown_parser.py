"""MarkdownParser - Extract text and metadata from Markdown files."""

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from prometheus_client import Counter, Histogram
from pydantic import Field

from infracore.ingest.base import IngestConfig
from infracore.ingest.pdf_parser import BaseIngester, IngestError, IngestResult


class MarkdownConfig(IngestConfig):
    """Markdown ingestion configuration."""

    store_type: str = Field(default="markdown", description="Parser type")
    preserve_headings: bool = Field(default=True, description="Keep heading text")
    strip_code_blocks: bool = Field(default=False, description="Remove code blocks")
    strip_frontmatter: bool = Field(default=True, description="Remove YAML frontmatter")
    min_content_chars: int = Field(default=50, description="Minimum content length")


class MarkdownParser(BaseIngester):
    """Markdown text extraction with metadata parsing."""

    def __init__(self, config: MarkdownConfig):
        super().__init__(config)
        self.config = config

        # Prometheus metrics (unique per parser type)
        self._counter = Counter(
            "markdown_ingest_documents_total",
            "Total Markdown documents ingested",
        )
        self._histogram = Histogram(
            "markdown_ingest_latency_seconds",
            "Markdown ingestion latency in seconds",
        )

    async def ingest(self, source: str) -> IngestResult:
        """
        Ingest Markdown from file path.

        Args:
            source: Path to Markdown file

        Returns:
            IngestResult with extracted text and metadata

        Raises:
            IngestError: If file not found or too small
        """
        start = time.time()

        try:
            # Run blocking file operation in thread pool
            result = await asyncio.to_thread(self._extract_markdown, source)

            # Record metrics
            self._counter.inc()
            latency = time.time() - start
            self._histogram.observe(latency)

            return result

        except (FileNotFoundError, IsADirectoryError) as e:
            raise IngestError(f"Markdown file not found: {source}") from e
        except Exception as e:
            raise IngestError(f"Failed to ingest Markdown {source}: {str(e)}") from e

    def _extract_markdown(self, source: str) -> IngestResult:
        """
        Synchronous Markdown extraction (runs in thread pool).

        Args:
            source: Path to Markdown file

        Returns:
            IngestResult with extracted text and metadata

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise IngestError(f"Failed to read Markdown file: {str(e)}") from e

        # Extract and strip YAML frontmatter
        frontmatter_dict: Optional[Dict[str, str]] = None
        if self.config.strip_frontmatter:
            content, frontmatter_dict = self._extract_frontmatter(content)

        # Count headings
        heading_count = len(re.findall(r"^#+\s+", content, re.MULTILINE))

        # Count code blocks
        code_blocks = re.findall(r"```.*?```", content, re.DOTALL)
        code_block_count = len(code_blocks)

        # Strip code blocks if configured
        if self.config.strip_code_blocks:
            content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

        # Clean up extra whitespace
        text = content.strip()

        # Validate minimum content size
        if len(text) < self.config.min_content_chars:
            raise IngestError(
                f"Content too small ({len(text)} chars < {self.config.min_content_chars})"
            )

        # Build metadata
        meta = {
            "source": source,
            "heading_count": heading_count,
            "code_block_count": code_block_count,
            "char_count": len(text),
            "frontmatter": frontmatter_dict,
        }

        return IngestResult(text=text, meta=meta, source=source)

    def _extract_frontmatter(
        self, content: str
    ) -> tuple[str, Optional[Dict[str, str]]]:
        """
        Extract and parse YAML frontmatter.

        Format:
            ---
            key1: value1
            key2: value2
            ---
            Rest of content

        Args:
            content: Markdown content

        Returns:
            Tuple of (content_without_frontmatter, frontmatter_dict or None)
        """
        # Regex to match frontmatter: --- at start, then content, then --- on own line
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)

        if not match:
            return content, None

        frontmatter_text = match.group(1)
        remaining_content = content[match.end() :]

        # Parse key:value pairs
        frontmatter_dict = {}
        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter_dict[key.strip()] = value.strip()

        return remaining_content, frontmatter_dict if frontmatter_dict else None
