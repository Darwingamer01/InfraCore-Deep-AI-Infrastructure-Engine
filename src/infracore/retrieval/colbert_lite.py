#!/usr/bin/env python
"""
INFRACORE — ColBERT-Lite Reranker

Purpose: Lightweight late-interaction reranking without full ColBERT infrastructure.

ColBERT concept: 
- Query and documents are embedded token-by-token
- Scoring uses MaxSim: for each query token, find max similarity to any document token
- Sum over all query tokens = final relevance score
- Faster than cross-encoder, more accurate than simple embedding dot-product
"""

import asyncio
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from infracore.retrieval.base import RetrievalResult


@dataclass
class TokenEmbedding:
    """Token-level embeddings for a document or query."""

    tokens: List[str]
    embeddings: np.ndarray  # (num_tokens, embedding_dim)


class ColBERTLiteReranker:
    """
    Lightweight ColBERT-style reranker using token-level embeddings.

    Uses a sentence-transformer model that supports token embeddings (e.g., 'all-MiniLM-L6-v2').
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize tokenizer and embedding model."""
        self.model = SentenceTransformer(model_name)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List["RerankedResult"]:
        """
        Rerank using ColBERT MaxSim scoring.

        Args:
            query: Search query
            candidates: List of RetrievalResult
            top_k: Return only top-k

        Returns:
            Sorted list of RerankedResult
        """
        from infracore.retrieval.cross_encoder_reranker import RerankedResult

        if not candidates:
            return []

        # Get query token embeddings
        query_embeddings = await asyncio.to_thread(
            self._get_token_embeddings, query
        )

        # Score each candidate
        scores = []
        for candidate in candidates:
            doc_embeddings = await asyncio.to_thread(
                self._get_token_embeddings, candidate.text
            )

            # MaxSim: max similarity between each query token and all doc tokens
            score = self._max_sim(query_embeddings, doc_embeddings)
            scores.append(score)

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

    def _get_token_embeddings(self, text: str) -> np.ndarray:
        """
        Get token-level embeddings for text.

        Returns: (num_tokens, embedding_dim) array
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Get token embeddings (from model's hidden state)
        with asyncio.to_thread as hidden:
            outputs = self.model(inputs, output_value="token_embeddings")

        # outputs shape: (1, num_tokens, embedding_dim)
        embeddings = outputs[0].cpu().numpy()

        return embeddings

    def _max_sim(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
        """
        MaxSim scoring: sum of max similarities over query tokens.

        Args:
            query_embeddings: (num_query_tokens, dim)
            doc_embeddings: (num_doc_tokens, dim)

        Returns:
            Scalar score (higher = more relevant)
        """
        # Cosine similarity between all query and doc tokens
        # (num_query_tokens, num_doc_tokens)
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(query_norm, doc_norm.T)  # (Q, D)

        # For each query token, get max similarity to any doc token
        max_sims = np.max(similarities, axis=1)  # (Q,)

        # Sum over query tokens
        score = np.sum(max_sims) / len(query_embeddings)  # Normalize by query length

        return float(score)

    def score_single(self, query: str, text: str) -> float:
        """Score a single query-document pair."""
        query_emb = self._get_token_embeddings(query)
        doc_emb = self._get_token_embeddings(text)
        return self._max_sim(query_emb, doc_emb)


# Fallback: Simple in-memory reranker for testing
class MockReranker:
    """Mock reranker for testing (uses original scores)."""

    async def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int | None = None,
    ) -> List["RerankedResult"]:
        """Rerank by returning candidates sorted by original score."""
        from infracore.retrieval.cross_encoder_reranker import RerankedResult

        results = []
        for i, c in enumerate(candidates):
            doc_id = c.metadata.get("doc_id", f"doc_{i}") if c.metadata else f"doc_{i}"
            source = c.metadata.get("source", "retrieval") if c.metadata else "retrieval"
            results.append(
                RerankedResult(
                    doc_id=str(doc_id),
                    text=c.text,
                    source=source,
                    rerank_score=c.score,  # Use original score
                    original_score=c.score,
                    rank=i,
                )
            )

        results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

        for i, r in enumerate(results):
            r.rank = i

        if top_k:
            results = results[:top_k]

        return results
