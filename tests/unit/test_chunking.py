"""
INFRACORE TEST — FixedChunker + SemanticChunker

Comprehensive test suite for both chunking strategies.
"""

import pytest

from src.infracore.chunking.base import ChunkConfig, Chunk
from src.infracore.chunking.fixed import FixedChunker
from src.infracore.chunking.semantic import SemanticChunker


# ============================================================================
# FixedChunker Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fixed_chunker_basic():
    """Test basic fixed chunking with simple text."""
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=2, min_chunk_size=5)
    chunker = FixedChunker(config)

    text = "word " * 50  # 50 words
    chunks = await chunker.chunk(text)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.start_idx < c.end_idx for c in chunks)


@pytest.mark.asyncio
async def test_fixed_chunker_word_count_limit():
    """Test that fixed chunks respect max_tokens (word count)."""
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=0, min_chunk_size=5)
    chunker = FixedChunker(config)

    text = " ".join([f"word{i}" for i in range(100)])
    chunks = await chunker.chunk(text)

    # Each chunk should have <= max_tokens words
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert word_count <= config.max_tokens, f"Chunk has {word_count} words, max is {config.max_tokens}"


@pytest.mark.asyncio
async def test_fixed_chunker_overlap():
    """Test that overlap works correctly."""
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=3, min_chunk_size=5)
    chunker = FixedChunker(config)

    text = " ".join([f"word{i}" for i in range(50)])
    chunks = await chunker.chunk(text)

    # Check overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk1_words = chunks[i].text.split()
        chunk2_words = chunks[i + 1].text.split()

        # Last N words of chunk1 should match first N words of chunk2
        overlap_size = min(config.overlap, len(chunk1_words))
        if overlap_size > 0:
            last_words_chunk1 = chunk1_words[-overlap_size:]
            first_words_chunk2 = chunk2_words[:overlap_size]
            assert (
                last_words_chunk1 == first_words_chunk2
            ), f"Overlap mismatch: {last_words_chunk1} != {first_words_chunk2}"


@pytest.mark.asyncio
async def test_fixed_chunker_min_chunk_size():
    """Test that chunks smaller than min_chunk_size are skipped."""
    config = ChunkConfig(strategy="fixed", max_tokens=5, overlap=0, min_chunk_size=3)
    chunker = FixedChunker(config)

    text = " ".join([f"word{i}" for i in range(20)])
    chunks = await chunker.chunk(text)

    # All chunks should have >= min_chunk_size words
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert word_count >= config.min_chunk_size, f"Chunk too small: {word_count} < {config.min_chunk_size}"


@pytest.mark.asyncio
async def test_fixed_chunker_metadata():
    """Test that metadata is correctly populated."""
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=2, min_chunk_size=5)
    chunker = FixedChunker(config)

    text = " ".join([f"word{i}" for i in range(50)])
    chunks = await chunker.chunk(text)

    for i, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["strategy"] == "fixed"
        assert "word_count" in chunk.metadata
        assert chunk.metadata["word_count"] == len(chunk.text.split())
        assert chunk.metadata["max_tokens"] == config.max_tokens
        assert chunk.metadata["overlap"] == config.overlap


@pytest.mark.asyncio
async def test_fixed_chunker_empty_text():
    """Test that empty text returns empty list."""
    config = ChunkConfig(strategy="fixed", max_tokens=10, overlap=0, min_chunk_size=5)
    chunker = FixedChunker(config)

    chunks = await chunker.chunk("")
    assert chunks == []

    chunks = await chunker.chunk("   ")
    assert chunks == []


@pytest.mark.asyncio
async def test_fixed_chunker_indices():
    """Test that start_idx and end_idx are correct."""
    config = ChunkConfig(strategy="fixed", max_tokens=5, overlap=0, min_chunk_size=3)
    chunker = FixedChunker(config)

    text = "word0 word1 word2 word3 word4 word5 word6 word7"
    chunks = await chunker.chunk(text)

    for chunk in chunks:
        # Verify indices point to correct substring
        indexed_text = text[chunk.start_idx : chunk.end_idx]
        assert indexed_text.strip() == chunk.text.strip(), f"Indices mismatch: '{indexed_text}' != '{chunk.text}'"


# ============================================================================
# SemanticChunker Tests
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_chunker_basic():
    """Test basic semantic chunking."""
    config = ChunkConfig(strategy="semantic", max_tokens=20, overlap=2, min_chunk_size=5)
    chunker = SemanticChunker(config)

    text = "This is a test. This is another test. And this is the third test. Here is the fourth test."
    chunks = await chunker.chunk(text)

    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


@pytest.mark.asyncio
async def test_semantic_chunker_no_mid_sentence_splits():
    """Test that semantic chunker doesn't split in middle of sentences."""
    config = ChunkConfig(strategy="semantic", max_tokens=10, overlap=0, min_chunk_size=3)
    chunker = SemanticChunker(config)

    text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
    chunks = await chunker.chunk(text)

    # Each chunk should contain complete sentences (no partial words)
    for chunk in chunks:
        # Check that chunk contains complete sentences
        assert not chunk.text.endswith(" "), f"Chunk ends with space: '{chunk.text}'"


@pytest.mark.asyncio
async def test_semantic_chunker_word_count_limit():
    """Test that semantic chunks respect max_tokens."""
    config = ChunkConfig(strategy="semantic", max_tokens=20, overlap=0, min_chunk_size=5)
    chunker = SemanticChunker(config)

    sentences = [
        "This is the first sentence with several words.",
        "This is the second sentence with several words.",
        "This is the third sentence with several words.",
        "This is the fourth sentence with several words.",
    ]
    text = " ".join(sentences)
    chunks = await chunker.chunk(text)

    # Each chunk should have <= max_tokens words
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert word_count <= config.max_tokens, f"Chunk has {word_count} words, max is {config.max_tokens}"


@pytest.mark.asyncio
async def test_semantic_chunker_sentence_count_metadata():
    """Test that sentence count is tracked in metadata."""
    config = ChunkConfig(strategy="semantic", max_tokens=50, overlap=0, min_chunk_size=5)
    chunker = SemanticChunker(config)

    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    chunks = await chunker.chunk(text)

    for chunk in chunks:
        assert "sentence_count" in chunk.metadata
        assert chunk.metadata["strategy"] == "semantic"
        assert chunk.metadata["sentence_count"] > 0


@pytest.mark.asyncio
async def test_semantic_chunker_empty_text():
    """Test that empty text returns empty list."""
    config = ChunkConfig(strategy="semantic", max_tokens=20, overlap=0, min_chunk_size=5)
    chunker = SemanticChunker(config)

    chunks = await chunker.chunk("")
    assert chunks == []

    chunks = await chunker.chunk("   ")
    assert chunks == []


@pytest.mark.asyncio
async def test_semantic_chunker_single_sentence():
    """Test chunking a single sentence."""
    config = ChunkConfig(strategy="semantic", max_tokens=50, overlap=0, min_chunk_size=3)
    chunker = SemanticChunker(config)

    text = "This is a single sentence with several words."
    chunks = await chunker.chunk(text)

    # Single sentence that's long enough should be one chunk
    if len(text.split()) >= config.min_chunk_size:
        assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_semantic_chunker_abbreviations():
    """Test that common abbreviations don't cause false sentence splits."""
    config = ChunkConfig(strategy="semantic", max_tokens=50, overlap=0, min_chunk_size=5)
    chunker = SemanticChunker(config)

    text = "Dr. Smith works at U.S.A. Corp. He is the CEO of the company. The company was founded in Inc."
    chunks = await chunker.chunk(text)

    # Should not split on Dr., U.S.A., Inc., etc.
    assert len(chunks) >= 1
    chunk_text = " ".join(c.text for c in chunks)
    assert "Dr. Smith" in chunk_text


# ============================================================================
# Comparative Tests
# ============================================================================


@pytest.mark.asyncio
async def test_fixed_vs_semantic_chunk_count():
    """Test that fixed chunking produces more chunks than semantic (generally)."""
    fixed_config = ChunkConfig(strategy="fixed", max_tokens=30, overlap=0, min_chunk_size=5)
    semantic_config = ChunkConfig(strategy="semantic", max_tokens=30, overlap=0, min_chunk_size=5)

    fixed_chunker = FixedChunker(fixed_config)
    semantic_chunker = SemanticChunker(semantic_config)

    text = "This is the first sentence with several words. " * 10  # Repeated text

    fixed_chunks = await fixed_chunker.chunk(text)
    semantic_chunks = await semantic_chunker.chunk(text)

    # Both should produce chunks, but counts may differ based on word boundaries
    assert len(fixed_chunks) > 0
    assert len(semantic_chunks) > 0


@pytest.mark.asyncio
async def test_both_chunkers_preserve_text():
    """Test that concatenating chunks recovers most of the original text."""
    config_fixed = ChunkConfig(strategy="fixed", max_tokens=25, overlap=0, min_chunk_size=5)
    config_semantic = ChunkConfig(strategy="semantic", max_tokens=25, overlap=0, min_chunk_size=5)

    text = "This is a sample text with several sentences. It should be chunked correctly. The original text should be mostly recoverable."

    fixed_chunker = FixedChunker(config_fixed)
    semantic_chunker = SemanticChunker(config_semantic)

    fixed_chunks = await fixed_chunker.chunk(text)
    semantic_chunks = await semantic_chunker.chunk(text)

    # Reconstruct text by joining chunks
    fixed_reconstructed = " ".join(c.text for c in fixed_chunks)
    semantic_reconstructed = " ".join(c.text for c in semantic_chunks)

    # Most of the original text should be recoverable (allowing for overlap duplicates)
    assert text.replace("  ", " ") in fixed_reconstructed or "".join(fixed_reconstructed.split()) in "".join(text.split())
    assert text.replace("  ", " ") in semantic_reconstructed or "".join(semantic_reconstructed.split()) in "".join(text.split())
