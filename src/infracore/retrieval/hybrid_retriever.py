"""HybridRetriever - Combines dense (vector) + sparse (BM25) retrieval using RRF."""

import asyncio
import math
import re
import time
from collections import Counter as CollectionsCounter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter as PrometheusCounter, Histogram
from pydantic import ConfigDict, Field

from src.infracore.embedding.base import BaseEmbedder
from src.infracore.retrieval.base import BaseRetriever, RetrieverConfig, RetrievalResult
from src.infracore.vectordb.base import BaseVectorStore, SearchResult


class HybridConfig(RetrieverConfig):
    """Hybrid retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    dense_weight: float = Field(default=0.7, description="Weight for dense retrieval")
    sparse_weight: float = Field(default=0.3, description="Weight for BM25")
    rrf_k: int = Field(default=60, description="RRF constant")
    dense_top_k: int = Field(default=20, description="Fetch from dense before fusion")
    sparse_top_k: int = Field(default=20, description="Fetch from BM25 before fusion")
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")


class BM25Index:
    """BM25 index built in-memory from corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_index: Dict[int, str] = {}  # doc_id -> text
        self.term_doc_freq: Dict[str, int] = defaultdict(int)  # term -> doc count
        self.doc_term_freq: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # doc_id -> (term -> count)
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def build(self, corpus: List[str]) -> None:
        """Build BM25 index from corpus."""
        self.corpus = corpus
        self.num_docs = len(corpus)

        total_length = 0

        for doc_id, doc in enumerate(corpus):
            # Tokenize: lowercase, split on whitespace, keep alphanumeric
            terms = self._tokenize(doc)
            self.doc_index[doc_id] = doc
            self.doc_lengths[doc_id] = len(terms)
            total_length += len(terms)

            # Track term frequencies in this document
            term_counts = CollectionsCounter(terms)
            for term, count in term_counts.items():
                self.doc_term_freq[doc_id][term] = count
                # Track document frequency
                if count > 0:
                    self.term_doc_freq[term] += 1

        self.avg_doc_length = total_length / max(self.num_docs, 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search corpus for query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) sorted by score descending
        """
        if not self.corpus:
            return []

        query_terms = self._tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        # Calculate BM25 score for each document
        for query_term in query_terms:
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self.term_doc_freq.get(query_term, 0)
            idf = math.log(
                (self.num_docs - df + 0.5) / (df + 0.5) + 1
            )

            # Calculate TF contribution for each document
            for doc_id in range(self.num_docs):
                tf = self.doc_term_freq[doc_id].get(query_term, 0)
                if tf == 0:
                    continue

                # TF normalization
                doc_length = self.doc_lengths[doc_id]
                tf_norm = tf * (self.k1 + 1) / (
                    tf
                    + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )

                # Score
                scores[doc_id] += idf * tf_norm

        # Sort by score descending, return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked[:top_k]]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, alphanumeric only."""
        # Convert to lowercase
        text = text.lower()
        # Split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text)
        return tokens


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense + sparse (BM25) search."""

    def __init__(
        self,
        config: HybridConfig,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
    ):
        super().__init__(config)
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder

        # BM25 index
        self.bm25 = BM25Index(k1=config.bm25_k1, b=config.bm25_b)
        self.doc_id_to_text: Dict[str, str] = {}  # Vector store doc_id -> text

        # Prometheus metrics
        self._counter = PrometheusCounter(
            "hybrid_retrieval_queries_total",
            "Total hybrid retrieval queries",
        )
        self._histogram = Histogram(
            "hybrid_retrieval_latency_seconds",
            "Hybrid retrieval latency",
        )

    async def build_index(self, corpus: List[str]) -> None:
        """
        Build BM25 index from corpus and upsert to vector store.

        Args:
            corpus: List of documents to index
        """
        # Build BM25 index
        self.bm25.build(corpus)

        # Embed and upsert to vector store
        embeddings = await self.embedder.embed(corpus)

        # Prepare payloads
        payloads = [{"text": doc, "corpus_idx": i} for i, doc in enumerate(corpus)]

        # Upsert to vector store
        await self.vector_store.upsert(
            vectors=embeddings,
            payloads=payloads,
            ids=[str(i) for i in range(len(corpus))],
        )

        # Store mapping from doc_id to text
        for i, text in enumerate(corpus):
            self.doc_id_to_text[str(i)] = text

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve using RRF fusion of dense + sparse search.

        Args:
            query: Search query
            top_k: Number of results (uses config.top_k if None)

        Returns:
            List of RetrievalResult sorted by fused score
        """
        if top_k is None:
            top_k = self.config.top_k

        start = time.time()

        try:
            # Dense retrieval
            query_embedding = await self.embedder.embed_single(query)
            dense_results = await self.vector_store.search(
                query_vector=query_embedding.reshape(1, -1),
                top_k=self.config.dense_top_k,
            )

            # Sparse retrieval (BM25)
            sparse_results = await asyncio.to_thread(
                self.bm25.search, query, self.config.sparse_top_k
            )

            # Build rank maps
            dense_ranks: Dict[str, int] = {
                result.id: rank for rank, result in enumerate(dense_results)
            }
            sparse_ranks: Dict[str, int] = {
                str(doc_id): rank for rank, (doc_id, _) in enumerate(sparse_results)
            }

            # RRF fusion
            fused_scores: Dict[str, float] = {}

            # Add dense scores
            for doc_id, rank in dense_ranks.items():
                score = self.config.dense_weight / (
                    self.config.rrf_k + rank
                )
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score

            # Add sparse scores
            for doc_id, rank in sparse_ranks.items():
                score = self.config.sparse_weight / (
                    self.config.rrf_k + rank
                )
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score

            # Sort by fused score
            ranked_results = sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Convert to RetrievalResult
            results = []
            for doc_id, score in ranked_results[:top_k]:
                if score < self.config.score_threshold:
                    continue

                text = self.doc_id_to_text.get(
                    doc_id, ""
                )

                results.append(
                    RetrievalResult(
                        text=text,
                        score=score,
                        metadata={"doc_id": doc_id, "retrieval_method": "hybrid"},
                    )
                )

            # Record metrics
            self._counter.inc()
            latency = time.time() - start
            self._histogram.observe(latency)

            return results

        except Exception as e:
            raise Exception(f"Hybrid retrieval failed: {str(e)}") from e

    async def bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        BM25 search (wrapper for async).

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (doc_id, score)
        """
        return await asyncio.to_thread(self.bm25.search, query, top_k)
