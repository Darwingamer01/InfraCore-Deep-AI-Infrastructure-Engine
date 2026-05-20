"""
INFRACORE — RecursiveChunker

Hierarchical chunking by paragraph -> sentence -> clause -> word.
Deterministic, supports max_tokens (word count), overlap and min_chunk_size.
"""

import re
import time
from typing import List

import structlog
from prometheus_client import Counter, Summary

from src.infracore.chunking.base import BaseChunker, Chunk, ChunkConfig

logger = structlog.get_logger()

# Metrics
RECURSIVE_CHUNKER_CALLS = Counter("infracore_recursive_chunker_calls_total", "Total RecursiveChunker.chunk calls")
RECURSIVE_CHUNKER_CHUNKS = Counter("infracore_recursive_chunker_chunks_total", "Total chunks produced by RecursiveChunker")
RECURSIVE_CHUNKER_WORDS = Summary("infracore_recursive_chunker_words", "Words processed per RecursiveChunker call")
RECURSIVE_CHUNKER_LATENCY = Summary("infracore_recursive_chunker_latency_seconds", "Latency of RecursiveChunker.chunk in seconds")


class RecursiveChunker(BaseChunker):
    """Recursive hierarchical chunker.

    Strategy:
    1. Split into paragraphs (\n\n).
    2. If paragraph fits, accept.
    3. Else split into sentences and group sentences into chunks under max_tokens.
    4. If a sentence is too large, split into clauses (commas/;).
    5. If a clause is too large, fall back to word-based fixed splits.

    Overlap: interpreted as sentence-count when operating at sentence level,
    as clause-count at clause level, and as words when doing word splits.
    """

    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$")
    CLAUSE_SPLIT_RE = re.compile(r",|;")

    async def chunk(self, text: str) -> List[Chunk]:
        RECURSIVE_CHUNKER_CALLS.inc()
        start_time = time.perf_counter()

        if not text or not text.strip():
            return []

        words_total = len(text.split())

        chunks: List[Chunk] = []

        # Top-level: paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        search_pos = 0

        for para in paragraphs:
            if not para:
                continue

            if len(para.split()) <= self.config.max_tokens:
                # Accept paragraph as a chunk candidate (subject to min size)
                if len(para.split()) >= self.config.min_chunk_size:
                    start_idx = text.find(para, search_pos)
                    if start_idx == -1:
                        start_idx = search_pos
                    end_idx = start_idx + len(para)
                    chunk = Chunk(text=para, start_idx=start_idx, end_idx=end_idx, metadata={
                        "chunk_index": len(chunks),
                        "strategy": "recursive",
                        "level": "paragraph",
                        "word_count": len(para.split()),
                    })
                    chunks.append(chunk)
                    search_pos = end_idx
                else:
                    # Too small: still include to avoid data loss
                    start_idx = text.find(para, search_pos)
                    if start_idx == -1:
                        start_idx = search_pos
                    end_idx = start_idx + len(para)
                    chunk = Chunk(text=para, start_idx=start_idx, end_idx=end_idx, metadata={
                        "chunk_index": len(chunks),
                        "strategy": "recursive",
                        "level": "paragraph",
                        "word_count": len(para.split()),
                    })
                    chunks.append(chunk)
                    search_pos = end_idx
                continue

            # Paragraph too large -> split into sentences
            sentences = [s.strip() for s in re.split(self.SENTENCE_SPLIT_RE, para) if s and s.strip()]

            current_sentences: List[str] = []
            current_word_count = 0

            for sent in sentences:
                sent_word_count = len(sent.split())
                # If sentence itself exceeds max_tokens -> split into clauses
                if sent_word_count > self.config.max_tokens:
                    clauses = [c.strip() for c in self.CLAUSE_SPLIT_RE.split(sent) if c and c.strip()]

                    for clause in clauses:
                        clause_wc = len(clause.split())
                        if clause_wc > self.config.max_tokens:
                            # Fallback: word-based fixed splits
                            sub_chunks = self._word_fixed_splits(clause)
                            for sc in sub_chunks:
                                # finalize current_sentences if any
                                if current_sentences:
                                    self._flush_sentence_group(current_sentences, chunks, text, search_pos)
                                    # adjust search_pos from last appended chunk
                                    if chunks:
                                        search_pos = chunks[-1].end_idx
                                    current_sentences = []
                                    current_word_count = 0

                                # add sub-chunk
                                start_idx = text.find(sc, search_pos)
                                if start_idx == -1:
                                    start_idx = search_pos
                                end_idx = start_idx + len(sc)
                                chunk = Chunk(text=sc, start_idx=start_idx, end_idx=end_idx, metadata={
                                    "chunk_index": len(chunks),
                                    "strategy": "recursive",
                                    "level": "word_split",
                                    "word_count": len(sc.split()),
                                })
                                chunks.append(chunk)
                                search_pos = end_idx
                        else:
                            # treat clause as a sentence-level unit
                            if current_word_count + clause_wc > self.config.max_tokens and current_sentences:
                                # finalize and keep overlap sentences/clauses
                                overlap_sentences = self._finalize_sentence_chunk(current_sentences, chunks, text, search_pos)
                                if chunks:
                                    search_pos = chunks[-1].end_idx
                                current_sentences = overlap_sentences
                                current_word_count = sum(len(s.split()) for s in current_sentences)

                            current_sentences.append(clause)
                            current_word_count += clause_wc
                else:
                    # Normal sentence
                    if current_word_count + sent_word_count > self.config.max_tokens and current_sentences:
                        overlap_sentences = self._finalize_sentence_chunk(current_sentences, chunks, text, search_pos)
                        if chunks:
                            search_pos = chunks[-1].end_idx
                        current_sentences = overlap_sentences
                        current_word_count = sum(len(s.split()) for s in current_sentences)

                    current_sentences.append(sent)
                    current_word_count += sent_word_count

            # Flush remaining sentence group
            if current_sentences:
                _ = self._finalize_sentence_chunk(current_sentences, chunks, text, search_pos)
                if chunks:
                    search_pos = chunks[-1].end_idx

        # Metrics + logging
        RECURSIVE_CHUNKER_CHUNKS.inc(len(chunks))
        RECURSIVE_CHUNKER_WORDS.observe(words_total)
        RECURSIVE_CHUNKER_LATENCY.observe(time.perf_counter() - start_time)
        logger.info("chunking.complete", strategy="recursive", num_chunks=len(chunks), total_words=words_total)

        return chunks

    def _finalize_sentence_chunk(self, sentences: List[str], chunks: List[Chunk], original_text: str, search_pos: int) -> List[str]:
        """Create a chunk from a list of sentences (or clauses) and append to chunks."""
        chunk_text = " ".join(sentences)
        # Enforce min_chunk_size
        if len(chunk_text.split()) < self.config.min_chunk_size:
            # still append to avoid loss
            pass

        start_idx = original_text.find(chunk_text, search_pos)
        if start_idx == -1:
            start_idx = search_pos
        end_idx = start_idx + len(chunk_text)
        chunk = Chunk(text=chunk_text, start_idx=start_idx, end_idx=end_idx, metadata={
            "chunk_index": len(chunks),
            "strategy": "recursive",
            "level": "sentence_group",
            "sentence_count": len(sentences),
            "word_count": len(chunk_text.split()),
        })
        chunks.append(chunk)
        # Compute overlap: interpret config.overlap as word count for deterministic word-level overlap
        try:
            overlap_words = int(self.config.overlap)
        except Exception:
            overlap_words = 0

        if overlap_words > 0:
            words = chunk_text.split()
            if overlap_words < len(words):
                tail = words[-overlap_words:]
                return [" ".join(tail)]

        return []

    def _word_fixed_splits(self, text: str) -> List[str]:
        """Split text into fixed-size word chunks honoring max_tokens and overlap."""
        words = text.split()
        if not words:
            return []

        max_t = int(self.config.max_tokens)
        overlap = int(self.config.overlap)
        chunks: List[str] = []

        idx = 0
        while idx < len(words):
            end = min(idx + max_t, len(words))
            chunk_words = words[idx:end]
            chunks.append(" ".join(chunk_words))
            if end >= len(words):
                break
            step = max_t - min(overlap, max_t // 2)
            if step <= 0:
                step = max_t
            idx += step

        return chunks
