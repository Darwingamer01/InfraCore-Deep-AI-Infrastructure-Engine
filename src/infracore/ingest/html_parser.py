"""HTMLParser - Extract text from HTML files and URLs."""

import asyncio
import html.parser
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import html2text
from prometheus_client import Counter, Histogram
from pydantic import Field

from infracore.ingest.base import IngestConfig
from infracore.ingest.pdf_parser import BaseIngester, IngestError, IngestResult


class TitleExtractor(html.parser.HTMLParser):
    """Extract <title> tag from HTML using stdlib parser."""

    def __init__(self):
        super().__init__()
        self.title = None
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        """Called when encountering opening tag."""
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        """Called when encountering closing tag."""
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        """Called when text data is encountered."""
        if self.in_title and data:
            self.title = data.strip()


class HTMLConfig(IngestConfig):
    """HTML ingestion configuration."""

    store_type: str = Field(default="html", description="Parser type")
    ignore_links: bool = Field(
        default=True, description="Remove links from output"
    )
    ignore_images: bool = Field(default=True, description="Remove images")
    ignore_tables: bool = Field(default=False, description="Remove tables")
    body_width: int = Field(default=0, description="Line width (0 = no wrapping)")
    min_content_chars: int = Field(default=100, description="Minimum content length")
    fetch_timeout: int = Field(default=10, description="URL fetch timeout (seconds)")


class HTMLParser(BaseIngester):
    """HTML text extraction using html2text."""

    def __init__(self, config: HTMLConfig):
        super().__init__(config)
        self.config = config

        # Prometheus metrics (unique per parser type)
        self._counter = Counter(
            "html_ingest_documents_total",
            "Total HTML documents ingested",
        )
        self._histogram = Histogram(
            "html_ingest_latency_seconds",
            "HTML ingestion latency in seconds",
        )

    async def ingest(self, source: str) -> IngestResult:
        """
        Ingest HTML from file path, URL, or raw HTML string.

        Args:
            source: File path (.html/.htm), URL (http/https), or raw HTML string

        Returns:
            IngestResult with extracted text and metadata

        Raises:
            IngestError: If processing fails or content too small
        """
        start = time.time()

        try:
            # Determine source type and extract HTML
            if source.startswith("http://") or source.startswith("https://"):
                # URL - fetch in thread pool
                html_content = await asyncio.to_thread(
                    self._fetch_url, source, self.config.fetch_timeout
                )
                source_identifier = source
            elif source.startswith("<"):
                # Raw HTML string
                html_content = source
                source_identifier = "[raw_html]"
            else:
                # File path
                html_content = await asyncio.to_thread(self._read_file, source)
                source_identifier = source

            # Extract title
            title = self._extract_title(html_content)

            # Convert HTML to text using html2text
            text = await asyncio.to_thread(
                self._convert_to_text, html_content
            )

            # Validate minimum content size
            if len(text.strip()) < self.config.min_content_chars:
                raise IngestError(
                    f"Content too small ({len(text)} chars < {self.config.min_content_chars})"
                )

            # Build metadata
            meta = {
                "url_or_path": source_identifier,
                "title": title,
                "char_count": len(text),
                "fetch_time_ms": int((time.time() - start) * 1000),
            }

            # Record metrics
            self._counter.inc()
            latency = time.time() - start
            self._histogram.observe(latency)

            return IngestResult(text=text, meta=meta, source=source_identifier)

        except IngestError:
            raise
        except Exception as e:
            raise IngestError(f"HTML ingestion failed: {str(e)}") from e

    def _fetch_url(self, url: str, timeout: int) -> str:
        """
        Fetch HTML from URL (blocking, runs in thread pool).

        Args:
            url: HTTP(S) URL
            timeout: Fetch timeout in seconds

        Returns:
            HTML content as string

        Raises:
            IngestError: If URL fetch fails
        """
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read()
                # Decode with fallback
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return content.decode("latin-1", errors="ignore")
        except Exception as e:
            raise IngestError(f"Failed to fetch URL {url}: {str(e)}") from e

    def _read_file(self, file_path: str) -> str:
        """
        Read HTML from file (blocking, runs in thread pool).

        Args:
            file_path: Path to .html/.htm file

        Returns:
            HTML content as string

        Raises:
            IngestError: If file not found or unreadable
        """
        path = Path(file_path)

        if not path.exists():
            raise IngestError(f"HTML file not found: {file_path}")

        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            raise IngestError(f"Failed to read HTML file: {str(e)}") from e

    def _extract_title(self, html_content: str) -> Optional[str]:
        """
        Extract <title> tag content from HTML.

        Args:
            html_content: HTML string

        Returns:
            Title text or None
        """
        try:
            extractor = TitleExtractor()
            extractor.feed(html_content)
            return extractor.title
        except Exception:
            return None

    def _convert_to_text(self, html_content: str) -> str:
        """
        Convert HTML to clean text using html2text (blocking, runs in thread pool).

        Args:
            html_content: HTML string

        Returns:
            Clean markdown-style text
        """
        converter = html2text.HTML2Text()
        converter.ignore_links = self.config.ignore_links
        converter.ignore_images = self.config.ignore_images
        converter.ignore_tables = self.config.ignore_tables
        converter.body_width = self.config.body_width

        text = converter.handle(html_content)

        # Strip excessive blank lines (max 2 consecutive)
        text = re.sub(r"\n\n\n+", "\n\n", text)

        return text.strip()
