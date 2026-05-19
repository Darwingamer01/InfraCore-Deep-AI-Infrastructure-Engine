"""Test suite for PDF, HTML, and Markdown parsers."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open
from dataclasses import dataclass

from src.infracore.ingest.pdf_parser import PDFParser, PDFConfig, IngestError, IngestResult
from src.infracore.ingest.html_parser import HTMLParser, HTMLConfig
from src.infracore.ingest.markdown_parser import MarkdownParser, MarkdownConfig


# ============================================================================
# PDFParser Tests (1-6)
# ============================================================================


@pytest.mark.asyncio
async def test_pdf_ingest_returns_result_with_text():
    """Test 1: ingest() returns IngestResult with non-empty text."""
    config = PDFConfig()
    parser = PDFParser(config)

    # Mock pypdfium2.PdfDocument and its chain
    with patch("src.infracore.ingest.pdf_parser.pypdfium2") as mock_pdf:
        with patch("src.infracore.ingest.pdf_parser.Path") as mock_path_class:
            # Mock Path.exists()
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.name = "test.pdf"
            mock_path_class.return_value = mock_path_inst

            # Setup mock chain: PdfDocument -> page -> textpage -> text
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_textpage = MagicMock()

            mock_pdf.PdfDocument.return_value = mock_doc
            mock_doc.__len__.return_value = 1  # 1 page
            mock_doc.__getitem__.return_value = mock_page
            mock_page.get_textpage.return_value = mock_textpage
            mock_textpage.get_text_bounded.return_value = "Hello, this is a PDF with enough content to pass minimum character threshold."

            result = await parser.ingest("test.pdf")

            assert isinstance(result, IngestResult)
            assert "Hello" in result.text
            assert "enough content" in result.text
            assert result.source == "test.pdf"
            assert result.chunks == []


@pytest.mark.asyncio
async def test_pdf_skips_pages_below_min_chars():
    """Test 2: Pages below min_page_chars are skipped and tracked."""
    config = PDFConfig(min_page_chars=100)
    parser = PDFParser(config)

    with patch("src.infracore.ingest.pdf_parser.pypdfium2") as mock_pdf:
        with patch("src.infracore.ingest.pdf_parser.Path") as mock_path_class:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.name = "test.pdf"
            mock_path_class.return_value = mock_path_inst

            mock_doc = MagicMock()
            mock_pdf.PdfDocument.return_value = mock_doc
            mock_doc.__len__.return_value = 3

            # Setup pages: short, long, short
            pages = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            mock_doc.__getitem__.side_effect = pages

            # Page 0: too short (skipped)
            textpage0 = MagicMock()
            textpage0.get_text_bounded.return_value = "Short"
            pages[0].get_textpage.return_value = textpage0

            # Page 1: long enough (included)
            textpage1 = MagicMock()
            textpage1.get_text_bounded.return_value = "X" * 150
            pages[1].get_textpage.return_value = textpage1

            # Page 2: too short (skipped)
            textpage2 = MagicMock()
            textpage2.get_text_bounded.return_value = "Also short"
            pages[2].get_textpage.return_value = textpage2

            result = await parser.ingest("test.pdf")

            assert result.meta["skipped_pages"] == [0, 2]
            assert result.meta["page_count"] == 3


@pytest.mark.asyncio
async def test_pdf_normalize_whitespace():
    """Test 3: normalize_whitespace collapses spaces."""
    config = PDFConfig(normalize_whitespace=True, min_page_chars=10)
    parser = PDFParser(config)

    with patch("src.infracore.ingest.pdf_parser.pypdfium2") as mock_pdf:
        with patch("src.infracore.ingest.pdf_parser.Path") as mock_path_class:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.name = "test.pdf"
            mock_path_class.return_value = mock_path_inst

            mock_doc = MagicMock()
            mock_pdf.PdfDocument.return_value = mock_doc
            mock_doc.__len__.return_value = 1

            mock_page = MagicMock()
            mock_doc.__getitem__.return_value = mock_page

            mock_textpage = MagicMock()
            mock_textpage.get_text_bounded.return_value = "hello    world  \t  test"
            mock_page.get_textpage.return_value = mock_textpage

            result = await parser.ingest("test.pdf")

            # Whitespace should be normalized
            assert "hello world test" in result.text


@pytest.mark.asyncio
async def test_pdf_metadata_contains_expected_fields():
    """Test 4: metadata contains filename, page_count, char_count."""
    config = PDFConfig()
    parser = PDFParser(config)

    with patch("src.infracore.ingest.pdf_parser.pypdfium2") as mock_pdf:
        with patch("src.infracore.ingest.pdf_parser.Path") as mock_path_class:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_inst.name = "test.pdf"
            mock_path_class.return_value = mock_path_inst

            mock_doc = MagicMock()
            mock_pdf.PdfDocument.return_value = mock_doc
            mock_doc.__len__.return_value = 2

            pages = [MagicMock(), MagicMock()]
            mock_doc.__getitem__.side_effect = pages

            for i, page in enumerate(pages):
                textpage = MagicMock()
                textpage.get_text_bounded.return_value = f"Page {i}" * 20
                page.get_textpage.return_value = textpage

            result = await parser.ingest("test.pdf")

            assert result.meta["filename"] == "test.pdf"
            assert result.meta["page_count"] == 2
            assert result.meta["char_count"] > 0


@pytest.mark.asyncio
async def test_pdf_ingest_error_file_not_found():
    """Test 5: IngestError raised when file does not exist."""
    config = PDFConfig()
    parser = PDFParser(config)

    # Don't patch - let it try to open a non-existent file
    with pytest.raises(IngestError, match="PDF file not found"):
        await parser.ingest("/nonexistent/path/test.pdf")


@pytest.mark.asyncio
async def test_pdf_ingest_error_corrupt_pdf():
    """Test 6: IngestError raised on corrupt PDF."""
    config = PDFConfig()
    parser = PDFParser(config)

    with patch("src.infracore.ingest.pdf_parser.pypdfium2") as mock_pdf:
        with patch("src.infracore.ingest.pdf_parser.Path") as mock_path_class:
            mock_path_inst = MagicMock()
            mock_path_inst.exists.return_value = True
            mock_path_class.return_value = mock_path_inst

            # Mock raises exception when opening
            mock_pdf.PdfDocument.side_effect = Exception("Corrupt PDF")

            with pytest.raises(IngestError, match="Corrupt PDF"):
                await parser.ingest("test.pdf")


# ============================================================================
# HTMLParser Tests (7-11)
# ============================================================================


@pytest.mark.asyncio
async def test_html_raw_string_extraction():
    """Test 7: ingest() with raw HTML string extracts text correctly."""
    config = HTMLConfig(min_content_chars=50)  # Lower threshold for test
    parser = HTMLParser(config)

    html = "<h1>Hello</h1><p>World</p><p>This is a longer HTML document with substantial content to meet minimum requirements.</p>"
    result = await parser.ingest(html)

    assert "Hello" in result.text
    assert "World" in result.text
    assert isinstance(result, IngestResult)


@pytest.mark.asyncio
async def test_html_title_extraction():
    """Test 8: <title> tag is correctly extracted into metadata."""
    config = HTMLConfig(min_content_chars=50)
    parser = HTMLParser(config)

    html = "<html><head><title>My Page Title</title></head><body><p>Content with enough text to meet minimum requirements for ingestion</p></body></html>"
    result = await parser.ingest(html)

    assert result.meta["title"] == "My Page Title"


@pytest.mark.asyncio
async def test_html_ignore_links():
    """Test 9: ignore_links=True removes href links from output."""
    config = HTMLConfig(ignore_links=True, min_content_chars=50)
    parser = HTMLParser(config)

    html = '<p>Check out <a href="http://example.com">this website</a> for more information about our products and services.</p>'
    result = await parser.ingest(html)

    # With ignore_links=True, the link should not appear in output
    # (html2text should not include the URL)
    assert "http://example.com" not in result.text or "this website" in result.text


@pytest.mark.asyncio
async def test_html_url_fetch_with_to_thread():
    """Test 10: URL source triggers asyncio.to_thread fetch."""
    config = HTMLConfig(min_content_chars=50)
    parser = HTMLParser(config)

    with patch("src.infracore.ingest.html_parser.urllib.request.urlopen") as mock_urlopen:
        # Mock URL response with enough content
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html><body>Fetched content with enough text to meet minimum requirements for the test.</body></html>"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)
        mock_urlopen.return_value = mock_response

        result = await parser.ingest("http://example.com/page.html")

        assert "Fetched content" in result.text
        assert "example.com" in result.meta["url_or_path"]
        mock_urlopen.assert_called_once()


@pytest.mark.asyncio
async def test_html_ingest_error_content_too_small():
    """Test 11: IngestError raised when fetched content < min_content_chars."""
    config = HTMLConfig(min_content_chars=1000)
    parser = HTMLParser(config)

    html = "<p>Too small</p>"
    with pytest.raises(IngestError, match="Content too small"):
        await parser.ingest(html)


# ============================================================================
# MarkdownParser Tests (12-15)
# ============================================================================


@pytest.mark.asyncio
async def test_markdown_frontmatter_stripped():
    """Test 12: YAML frontmatter is stripped from output text."""
    config = MarkdownConfig(strip_frontmatter=True)
    parser = MarkdownParser(config)

    content = """---
title: My Doc
author: Test
---
# Introduction

This is the main content with enough text to meet minimum requirements."""

    with patch("src.infracore.ingest.markdown_parser.Path") as mock_path_class:
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_inst.read_text.return_value = content
        mock_path_class.return_value = mock_path_inst

        result = await parser.ingest("test.md")

        # Frontmatter should not be in text
        assert "---" not in result.text
        assert "title: My Doc" not in result.text
        assert "Introduction" in result.text


@pytest.mark.asyncio
async def test_markdown_frontmatter_parsing():
    """Test 13: Frontmatter key:value pairs appear in metadata."""
    config = MarkdownConfig(strip_frontmatter=True)
    parser = MarkdownParser(config)

    content = """---
title: My Document
author: John Doe
---
# Content

Main text here with sufficient content to meet minimum requirements."""

    with patch("src.infracore.ingest.markdown_parser.Path") as mock_path_class:
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_inst.read_text.return_value = content
        mock_path_class.return_value = mock_path_inst

        result = await parser.ingest("test.md")

        assert result.meta["frontmatter"] is not None
        assert result.meta["frontmatter"]["title"] == "My Document"
        assert result.meta["frontmatter"]["author"] == "John Doe"


@pytest.mark.asyncio
async def test_markdown_heading_count():
    """Test 14: heading_count is correct for document with headings."""
    config = MarkdownConfig()
    parser = MarkdownParser(config)

    content = """# Title
## Section 1
Content here with enough text to meet minimum requirements and be properly ingested.
## Section 2
More content
### Subsection
Even more content here"""

    with patch("src.infracore.ingest.markdown_parser.Path") as mock_path_class:
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_inst.read_text.return_value = content
        mock_path_class.return_value = mock_path_inst

        result = await parser.ingest("test.md")

        # Should count 4 headings (1 h1, 2 h2, 1 h3)
        assert result.meta["heading_count"] == 4


@pytest.mark.asyncio
async def test_markdown_strip_code_blocks():
    """Test 15: strip_code_blocks=True removes ```python ... ``` blocks."""
    config = MarkdownConfig(strip_code_blocks=True)
    parser = MarkdownParser(config)

    content = """# Example

Here's some code:

```python
def hello():
    return "world"
```

And text after code with additional content to meet minimum requirements for ingestion."""

    with patch("src.infracore.ingest.markdown_parser.Path") as mock_path_class:
        mock_path_inst = MagicMock()
        mock_path_inst.exists.return_value = True
        mock_path_inst.read_text.return_value = content
        mock_path_class.return_value = mock_path_inst

        result = await parser.ingest("test.md")

        # Code block should be stripped
        assert "def hello" not in result.text
        assert "return" not in result.text
        assert "Example" in result.text
        assert "And text after code" in result.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
