"""
INFRACORE — SemanticChunker

Semantic sentence-based chunking with word count limits.
No external libraries (no NLTK, spaCy). Pure regex + Python.
"""

import re
import time
from typing import List

import structlog
from prometheus_client import Counter, Summary

from src.infracore.chunking.base import BaseChunker, Chunk, ChunkConfig

logger = structlog.get_logger()

# Metrics
SEMANTIC_CHUNKER_CALLS = Counter("infracore_semantic_chunker_calls_total", "Total SemanticChunker.chunk calls")
SEMANTIC_CHUNKER_CHUNKS = Counter("infracore_semantic_chunker_chunks_total", "Total chunks produced by SemanticChunker")
SEMANTIC_CHUNKER_WORDS = Summary("infracore_semantic_chunker_words", "Words processed per SemanticChunker call")
SEMANTIC_CHUNKER_LATENCY = Summary("infracore_semantic_chunker_latency_seconds", "Latency of SemanticChunker.chunk in seconds")


class SemanticChunker(BaseChunker):
    """
    Semantic sentence-based chunker.
    
    Splits text into sentences using regex, then groups sentences into chunks
    where total word count stays under max_tokens. Skips chunks smaller than min_chunk_size.
    
    Example:
        config = ChunkConfig(strategy="semantic", max_tokens=256, overlap=32, min_chunk_size=64)
        chunker = SemanticChunker(config)
        chunks = await chunker.chunk("Your text here...")
    """

    # Regex to split on sentence boundaries
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$")

    async def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text into semantic sentence-based segments.

        Args:
            text: Raw text to chunk

        Returns:
            List of Chunk objects with metadata
        """
        SEMANTIC_CHUNKER_CALLS.inc()
        start_time = time.perf_counter()

        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_chunk_sentences: List[str] = []
        current_chunk_word_count = 0
        chunk_start_idx = 0

        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            word_count = len(sentence.split())

            # If adding this sentence would exceed max_tokens, save current chunk
            if (
                current_chunk_word_count + word_count > self.config.max_tokens
                and current_chunk_sentences
            ):
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)

                if len(chunk_text.split()) >= self.config.min_chunk_size:
                    chunk = Chunk(
                        text=chunk_text,
                        start_idx=chunk_start_idx,
                        end_idx=chunk_start_idx + len(chunk_text),
                        metadata={
                            "chunk_index": len(chunks),
                            "strategy": "semantic",
                            "sentence_count": len(current_chunk_sentences),
                            "word_count": len(chunk_text.split()),
                        },
                    )
                    chunks.append(chunk)

                    # Apply overlap: use configured overlap interpreted as sentence count
                    overlap_sentences = min(int(self.config.overlap), len(current_chunk_sentences))
                    if overlap_sentences > 0 and len(current_chunk_sentences) > overlap_sentences:
                        current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                        current_chunk_word_count = sum(len(s.split()) for s in current_chunk_sentences)
                    else:
                        current_chunk_sentences = []
                        current_chunk_word_count = 0

                    # Update chunk_start_idx based on how far we've consumed text
                    if current_chunk_sentences:
                        overlap_text = " ".join(current_chunk_sentences)
                        chunk_start_idx = chunk_start_idx + len(chunk_text) - len(overlap_text)
                    else:
                        chunk_start_idx = chunk_start_idx + len(chunk_text)

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_word_count += word_count

        # Save final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text.split()) >= self.config.min_chunk_size:
                chunk = Chunk(
                    text=chunk_text,
                    start_idx=chunk_start_idx,
                    end_idx=chunk_start_idx + len(chunk_text),
                    metadata={
                        "chunk_index": len(chunks),
                        "strategy": "semantic",
                        "sentence_count": len(current_chunk_sentences),
                        "word_count": len(chunk_text.split()),
                    },
                )
                chunks.append(chunk)

        # Metrics + logging
        SEMANTIC_CHUNKER_CHUNKS.inc(len(chunks))
        SEMANTIC_CHUNKER_WORDS.observe(sum(len(s.split()) for s in sentences))
        SEMANTIC_CHUNKER_LATENCY.observe(time.perf_counter() - start_time)
        logger.info("chunking.complete", strategy="semantic", num_chunks=len(chunks), total_sentences=len(sentences))

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Looks for sentence boundaries (., !, ?) followed by space and capital letter.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Handle common abbreviations to avoid false splits
        text = text.replace("Dr. ", "Dr_DOT_ ")
        text = text.replace("Mr. ", "Mr_DOT_ ")
        text = text.replace("Mrs. ", "Mrs_DOT_ ")
        text = text.replace("Ms. ", "Ms_DOT_ ")
        text = text.replace("Jr. ", "Jr_DOT_ ")
        text = text.replace("Sr. ", "Sr_DOT_ ")
        text = text.replace("Ph.D. ", "Ph_D_DOT_ ")
        text = text.replace("U.S.A.", "U_S_A_DOT")
        text = text.replace("U.S.", "U_S_DOT")
        text = text.replace("Inc.", "Inc_DOT")
        text = text.replace("Ltd.", "Ltd_DOT")
        text = text.replace("Co.", "Co_DOT")

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$", text)

        # Restore abbreviations
        sentences = [
            s.replace("Dr_DOT_ ", "Dr. ")
            .replace("Mr_DOT_ ", "Mr. ")
            .replace("Mrs_DOT_ ", "Mrs. ")
            .replace("Ms_DOT_ ", "Ms. ")
            .replace("Jr_DOT_ ", "Jr. ")
            .replace("Sr_DOT_ ", "Sr. ")
            .replace("Ph_D_DOT_ ", "Ph.D. ")
            .replace("U_S_A_DOT", "U.S.A.")
            .replace("U_S_DOT", "U.S.")
            .replace("Inc_DOT", "Inc.")
            .replace("Ltd_DOT", "Ltd.")
            .replace("Co_DOT", "Co.")
            for s in sentences
        ]

        return [s.strip() for s in sentences if s.strip()]
