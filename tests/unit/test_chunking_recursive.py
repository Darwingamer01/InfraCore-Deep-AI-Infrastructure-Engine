"""Unit tests for RecursiveChunker"""

import pytest

from src.infracore.chunking.base import ChunkConfig
from src.infracore.chunking.recursive import RecursiveChunker


@pytest.mark.asyncio
async def test_recursive_empty_and_short_text():
    cfg = ChunkConfig(strategy="recursive", max_tokens=50, overlap=2, min_chunk_size=3)
    chunker = RecursiveChunker(cfg)

    assert await chunker.chunk("") == []
    assert await chunker.chunk("   ") == []

    short = "This is short."
    chunks = await chunker.chunk(short)
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_recursive_paragraph_splitting():
    cfg = ChunkConfig(strategy="recursive", max_tokens=100, overlap=1, min_chunk_size=3)
    chunker = RecursiveChunker(cfg)

    text = "Para one." + "\n\n" + "Para two is a bit longer " * 10
    chunks = await chunker.chunk(text)
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_recursive_max_size_enforcement():
    cfg = ChunkConfig(strategy="recursive", max_tokens=5, overlap=1, min_chunk_size=1)
    chunker = RecursiveChunker(cfg)

    # Create a long sentence without sentence boundaries so it must fall back to clause/word splits
    long = "word " * 42
    chunks = await chunker.chunk(long)
    assert all(len(c.text.split()) <= cfg.max_tokens for c in chunks)


@pytest.mark.asyncio
async def test_recursive_overlap_and_determinism():
    cfg = ChunkConfig(strategy="recursive", max_tokens=10, overlap=2, min_chunk_size=1)
    chunker = RecursiveChunker(cfg)

    text = ("This is sentence one. " * 3) + ("This is sentence two. " * 3)
    first = await chunker.chunk(text)
    second = await chunker.chunk(text)

    # deterministic
    assert [c.text for c in first] == [c.text for c in second]

    # overlap reasonableness: consecutive chunks should share some words when overlap>0
    if len(first) > 1:
        last_words = first[0].text.split()[-cfg.overlap :]
        next_words = first[1].text.split()[: cfg.overlap]
        assert last_words == next_words
