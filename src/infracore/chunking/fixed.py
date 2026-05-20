"""
INFRACORE — FixedChunker

Fixed-size text chunking by word count with configurable overlap.
No external libraries. Pure async Python.
"""

from typing import List

from pydantic import Field

from src.infracore.chunking.base import BaseChunker, Chunk, ChunkConfig


class FixedChunkConfig(ChunkConfig):
    """Backward-compatible config with a default fixed strategy."""

    strategy: str = Field(default="fixed", description="Chunking strategy: fixed")


class FixedChunker(BaseChunker):
    """
    Fixed-size word-based chunker.
    
    Splits text into chunks of approximately max_tokens words, with optional overlap.
    Skips chunks smaller than min_chunk_size.
    
    Example:
        config = ChunkConfig(strategy="fixed", max_tokens=256, overlap=32, min_chunk_size=64)
        chunker = FixedChunker(config)
        chunks = await chunker.chunk("Your text here...")
    """

    async def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text into fixed-size word-based segments.

        Args:
            text: Raw text to chunk

        Returns:
            List of Chunk objects with metadata
        """
        if not text or not text.strip():
            return []

        # Split into words
        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        current_start_word_idx = 0

        while current_start_word_idx < len(words):
            # Get chunk of max_tokens words
            chunk_end_word_idx = min(
                current_start_word_idx + self.config.max_tokens, len(words)
            )
            chunk_words = words[current_start_word_idx:chunk_end_word_idx]

            # Skip if chunk is too small
            if len(chunk_words) < self.config.min_chunk_size:
                if chunks:
                    break
                # Keep a single short chunk so small documents are still indexed.
                # This preserves the minimum-size guard for later fragments.
                if not chunk_words:
                    break

            # Find byte positions for start_idx and end_idx
            chunk_text = " ".join(chunk_words)

            # Calculate start byte index in original text
            start_text = " ".join(words[:current_start_word_idx])
            if start_text:
                start_idx = len(start_text) + 1  # +1 for space
            else:
                start_idx = 0

            # Calculate end byte index
            end_text = " ".join(words[:chunk_end_word_idx])
            end_idx = len(end_text)

            # Create chunk with metadata
            chunk = Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                metadata={
                    "chunk_index": len(chunks),
                    "strategy": "fixed",
                    "word_count": len(chunk_words),
                    "max_tokens": self.config.max_tokens,
                    "overlap": self.config.overlap,
                },
            )
            chunks.append(chunk)

            # Move to next chunk start (accounting for overlap)
            overlap_words = min(self.config.overlap, len(chunk_words) // 2)
            current_start_word_idx = chunk_end_word_idx - overlap_words

            # Prevent infinite loop if no overlap and small chunks
            if overlap_words == 0 and chunk_end_word_idx >= len(words):
                break

        return chunks
