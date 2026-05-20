#!/usr/bin/env python
"""
INFRACORE — Cross-Encoder Reranking Module

Purpose: Rerank retrieved documents using semantic similarity scores from
a cross-encoder model (MS-MARCO MiniLM-L-6-v2).

Cross-encoders: Process query+document pairs directly for higher accuracy
than bi-encoder similarity.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.special import expit
from sentence_transformers import CrossEncoder

from infracore.retrieval.base import RetrievalResult


@dataclass
class RerankedResult:
    """Single reranked document with score."""

    doc_id: str
    text: str
    source: str
    rerank_score: float
    original_score: float = 0.0
    rank: int = field(default=0)


class CrossEncoderReranker:
    """
    Rerank retrieved documents using cross-encoder scoring.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Fine-tuned on MS-MARCO dataset for relevance ranking
    - Fast and accurate (6-layer BERT)
    - Scores 0-1 (higher = more relevant)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder model."""
        self.model = CrossEncoder(model_name)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder scores.

        Args:
            query: Search query
            candidates: List of RetrievalResult from retrieval
            top_k: Return only top-k results (default: all)
            batch_size: Batch size for scoring

        Returns:
            Sorted list of RerankedResult by score descending
        """
        if not candidates:
            return []

        # Prepare query-document pairs
        pairs = [[query, c.text] for c in candidates]

        # Score in batches
        scores = await asyncio.to_thread(
            self._score_batch,
            pairs,
            batch_size,
        )

        # Create results
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            doc_id = candidate.metadata.get("doc_id", f"doc_{i}") if candidate.metadata else f"doc_{i}"
            source = candidate.metadata.get("source", "retrieval") if candidate.metadata else "retrieval"
            results.append(
                RerankedResult(
                    doc_id=str(doc_id),
                    text=candidate.text,
                    source=source,
                    rerank_score=float(score),
                    original_score=candidate.score,
                    rank=i,
                )
            )

        # Sort by rerank score descending
        results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

        # Re-rank indices
        for i, r in enumerate(results):
            r.rank = i

        # Truncate to top_k
        if top_k:
            results = results[:top_k]

        return results

    def _score_batch(self, pairs: List[List[str]], batch_size: int) -> np.ndarray:
        """Score batch of query-document pairs."""
        scores = self.model.predict(pairs, batch_size=batch_size)
        return expit(scores)

    def score_single(self, query: str, text: str) -> float:
        """Score a single query-document pair."""
        score = self.model.predict([[query, text]])[0]
        return float(expit(score))


class FastApproxReranker:
    """
    Lightweight reranker using sentence-transformer embeddings.

    Faster than cross-encoder but less accurate. Useful for quick ranking
    before heavy cross-encoder reranking.
    """

    def __init__(self):
        """Initialize with None for now (uses existing embedder if available)."""
        pass

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        top_k: int | None = None,
    ) -> List[RerankedResult]:
        """
        Rerank using dot product similarity (cosine on normalized embeddings).

        Args:
            query: Search query
            candidates: List of RetrievalResult
            query_embedding: Query embedding vector
            embeddings: Document embeddings
            top_k: Return only top-k

        Returns:
            Sorted list of RerankedResult
        """
        if not candidates:
            return []

        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = np.array([
            e / (np.linalg.norm(e) + 1e-8) for e in embeddings
        ])

        # Cosine similarity
        scores = np.dot(doc_norms, query_norm)

        # Create results
        results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            doc_id = candidate.metadata.get("doc_id", f"doc_{i}") if candidate.metadata else f"doc_{i}"
            source = candidate.metadata.get("source", "retrieval") if candidate.metadata else "retrieval"
            results.append(
                RerankedResult(
                    doc_id=str(doc_id),
                    text=candidate.text,
                    source=source,
                    rerank_score=float(score),
                    original_score=candidate.score,
                    rank=i,
                )
            )

        # Sort descending
        results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

        for i, r in enumerate(results):
            r.rank = i

        if top_k:
            results = results[:top_k]

        return results
