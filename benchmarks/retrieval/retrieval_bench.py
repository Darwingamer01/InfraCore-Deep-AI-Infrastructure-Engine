#!/usr/bin/env python
"""
INFRACORE — Retrieval Reranking Benchmark Suite

Purpose: Compare retrieval strategies end-to-end:
  1. Dense embedding only
  2. BM25 only
  3. Hybrid RRF (dense + BM25)
  4. Hybrid + CrossEncoder reranking
  5. Hybrid + ColBERT-lite reranking

Metrics: Recall@10, MRR@10, nDCG@10, p50/p95/p99 latency
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from infracore.retrieval.base import RetrievalResult
from infracore.retrieval.cross_encoder_reranker import CrossEncoderReranker
from infracore.retrieval.colbert_lite import MockReranker


@dataclass
class BenchmarkQA:
    """Single QA pair with ground truth relevance."""

    qid: str
    question: str
    positive_docs: List[str]  # Ground truth relevant doc IDs


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval strategy."""

    strategy: str
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    n_queries: int


@dataclass
class BenchConfig:
    """Benchmark configuration."""

    n_results: int = 10
    n_queries: int = 20
    timeout_seconds: int = 30


# Hardcoded benchmark dataset
BENCHMARK_QAS = [
    BenchmarkQA(
        qid="q1",
        question="What is machine learning?",
        positive_docs=["doc_1", "doc_2"],
    ),
    BenchmarkQA(
        qid="q2",
        question="How do neural networks work?",
        positive_docs=["doc_3", "doc_4"],
    ),
    BenchmarkQA(
        qid="q3",
        question="What is supervised learning?",
        positive_docs=["doc_2", "doc_5"],
    ),
    BenchmarkQA(
        qid="q4",
        question="Explain transformers architecture",
        positive_docs=["doc_6", "doc_7"],
    ),
    BenchmarkQA(
        qid="q5",
        question="What are embeddings?",
        positive_docs=["doc_8", "doc_9"],
    ),
]

BENCHMARK_DOCS = {
    "doc_1": "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    "doc_2": "Supervised learning requires labeled data to train models effectively.",
    "doc_3": "Neural networks are inspired by biological neurons and their interconnections.",
    "doc_4": "Deep learning uses multiple layers of neural networks for complex patterns.",
    "doc_5": "Training data is crucial for supervised learning algorithms.",
    "doc_6": "Transformers use self-attention mechanisms to process sequences in parallel.",
    "doc_7": "The attention mechanism allows models to focus on relevant parts of input.",
    "doc_8": "Embeddings are dense vector representations of words or documents.",
    "doc_9": "Word embeddings capture semantic meaning in vector space.",
    "doc_10": "Recurrent neural networks process sequential data step by step.",
}


def compute_recall_at_k(retrieved_ids: List[str], positive_ids: List[str], k: int) -> float:
    """Compute Recall@k: intersection / total positives."""
    retrieved_at_k = set(retrieved_ids[:k])
    positive_set = set(positive_ids)
    if not positive_set:
        return 0.0
    intersection = len(retrieved_at_k & positive_set)
    return intersection / len(positive_set)


def compute_mrr_at_k(retrieved_ids: List[str], positive_ids: List[str], k: int) -> float:
    """Compute MRR@k: reciprocal of rank of first relevant doc."""
    positive_set = set(positive_ids)
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in positive_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], positive_ids: List[str], k: int) -> float:
    """Compute nDCG@k: normalized discounted cumulative gain."""
    positive_set = set(positive_ids)

    # DCG: sum of relevance / log(rank+1)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        relevance = 1.0 if doc_id in positive_set else 0.0
        dcg += relevance / np.log2(i + 2)

    # iDCG: ideal ordering (all relevant docs first)
    idcg = 0.0
    for i in range(min(len(positive_set), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class RetrievalBench:
    """Retrieval benchmarking harness."""

    def __init__(self, config: BenchConfig):
        self.config = config
        self.cross_encoder = CrossEncoderReranker()
        self.colbert = MockReranker()  # Use MockReranker instead of ColBERTLiteReranker for testing

    def _mock_dense_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """Mock dense retrieval (returns random relevant docs + noise)."""
        # Simulate retrieval by keyword matching
        relevant = [
            RetrievalResult(
                text=text,
                score=np.random.rand(),
                metadata={"doc_id": doc_id, "strategy": "dense"},
            )
            for doc_id, text in BENCHMARK_DOCS.items()
            if any(w in text.lower() for w in query.lower().split())
        ]

        # Add some non-relevant docs
        non_relevant = [
            RetrievalResult(
                text=text,
                score=np.random.rand() * 0.5,
                metadata={"doc_id": doc_id, "strategy": "dense"},
            )
            for doc_id, text in BENCHMARK_DOCS.items()
            if not any(w in text.lower() for w in query.lower().split())
        ]

        # Mix and return top-k
        all_results = sorted(
            relevant + non_relevant,
            key=lambda x: x.score,
            reverse=True,
        )[:k]

        return all_results

    def _mock_bm25_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """Mock BM25 retrieval."""
        # Similar to dense but different scoring
        results = []
        query_words = set(query.lower().split())

        for doc_id, text in BENCHMARK_DOCS.items():
            doc_words = set(text.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / (len(query_words) + 1e-8)

            results.append(
                RetrievalResult(
                    text=text,
                    score=score,
                    metadata={"doc_id": doc_id, "strategy": "bm25"},
                )
            )

        results = sorted(results, key=lambda x: x.score, reverse=True)[:k]
        return results

    async def run_single_query(
        self,
        qa: BenchmarkQA,
        strategy: str,
    ) -> tuple[List[str], float]:
        """
        Run retrieval for a single query and return (retrieved_doc_ids, latency_ms).

        Strategies:
          - dense: Dense retrieval only
          - bm25: BM25 only
          - hybrid_rrf: Dense + BM25 with RRF
          - hybrid_reranker_ce: Hybrid + CrossEncoder
          - hybrid_reranker_colbert: Hybrid + ColBERT-lite
        """
        start = time.perf_counter()

        if strategy == "dense":
            results = self._mock_dense_retrieval(qa.question, self.config.n_results)
        elif strategy == "bm25":
            results = self._mock_bm25_retrieval(qa.question, self.config.n_results)
        elif strategy == "hybrid_rrf":
            # Combine dense and BM25 with reciprocal rank fusion
            dense = self._mock_dense_retrieval(qa.question, self.config.n_results)
            bm25 = self._mock_bm25_retrieval(qa.question, self.config.n_results)

            # RRF: 1/(rank+60)
            scores = {}
            for rank, result in enumerate(dense):
                doc_id = result.metadata.get("doc_id", f"doc_{rank}") if result.metadata else f"doc_{rank}"
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)
            for rank, result in enumerate(bm25):
                doc_id = result.metadata.get("doc_id", f"doc_{rank}") if result.metadata else f"doc_{rank}"
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)

            # Re-rank by fused scores
            results = [
                RetrievalResult(
                    text=BENCHMARK_DOCS.get(doc_id, ""),
                    score=score,
                    metadata={"doc_id": doc_id, "strategy": "hybrid"},
                )
                for doc_id, score in scores.items()
            ]
            results = sorted(results, key=lambda x: x.score, reverse=True)[
                : self.config.n_results
            ]

        elif strategy == "hybrid_reranker_ce":
            # Hybrid + CrossEncoder reranking
            hybrid = await asyncio.to_thread(
                self._hybrid_retrieval,
                qa.question,
                self.config.n_results * 2,  # Retrieve more candidates
            )
            reranked = await self.cross_encoder.rerank(
                qa.question,
                hybrid,
                top_k=self.config.n_results,
            )
            # Convert RerankedResult back to RetrievalResult
            results = [
                RetrievalResult(
                    text=r.text,
                    score=r.rerank_score,
                    metadata={"doc_id": r.doc_id, "source": r.source, "strategy": "hybrid_reranker_ce"},
                )
                for r in reranked
            ]

        elif strategy == "hybrid_reranker_colbert":
            # Hybrid + ColBERT-lite reranking
            hybrid = await asyncio.to_thread(
                self._hybrid_retrieval,
                qa.question,
                self.config.n_results * 2,
            )
            reranked = await self.colbert.rerank(
                qa.question,
                hybrid,
                top_k=self.config.n_results,
            )
            # Convert RerankedResult back to RetrievalResult
            results = [
                RetrievalResult(
                    text=r.text,
                    score=r.rerank_score,
                    metadata={"doc_id": r.doc_id, "source": r.source, "strategy": "hybrid_reranker_colbert"},
                )
                for r in reranked
            ]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        latency_ms = (time.perf_counter() - start) * 1000
        retrieved_ids = [r.metadata.get("doc_id", f"doc_{i}") for i, r in enumerate(results)]

        return retrieved_ids, latency_ms

    def _hybrid_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """Helper: hybrid retrieval (dense + BM25 RRF)."""
        dense = self._mock_dense_retrieval(query, k)
        bm25 = self._mock_bm25_retrieval(query, k)

        scores = {}
        for rank, result in enumerate(dense):
            doc_id = result.metadata.get("doc_id", result.text[:20])
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)
        for rank, result in enumerate(bm25):
            doc_id = result.metadata.get("doc_id", result.text[:20])
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)

        results = [
            RetrievalResult(
                text=BENCHMARK_DOCS.get(doc_id, ""),
                score=score,
                metadata={"doc_id": doc_id, "strategy": "hybrid"},
            )
            for doc_id, score in scores.items()
        ]
        return sorted(results, key=lambda x: x.score, reverse=True)[:k]

    async def run_strategy(self, strategy: str) -> RetrievalMetrics:
        """Run benchmark for a single strategy."""
        print(f"\n⏱️  Benchmarking: {strategy.upper()}")

        latencies = []
        recalls = []
        mrrs = []
        ndcgs = []

        qas = BENCHMARK_QAS[: self.config.n_queries]

        for i, qa in enumerate(qas):
            retrieved_ids, latency = await self.run_single_query(qa, strategy)

            # Compute metrics
            recall = compute_recall_at_k(retrieved_ids, qa.positive_docs, k=10)
            mrr = compute_mrr_at_k(retrieved_ids, qa.positive_docs, k=10)
            ndcg = compute_ndcg_at_k(retrieved_ids, qa.positive_docs, k=10)

            recalls.append(recall)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
            latencies.append(latency)

            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(qas)} queries completed")

        latencies_array = np.array(latencies)

        metrics = RetrievalMetrics(
            strategy=strategy,
            recall_at_10=np.mean(recalls),
            mrr_at_10=np.mean(mrrs),
            ndcg_at_10=np.mean(ndcgs),
            p50_latency_ms=float(np.percentile(latencies_array, 50)),
            p95_latency_ms=float(np.percentile(latencies_array, 95)),
            p99_latency_ms=float(np.percentile(latencies_array, 99)),
            avg_latency_ms=float(np.mean(latencies_array)),
            n_queries=len(qas),
        )

        return metrics

    async def run_all(self) -> List[RetrievalMetrics]:
        """Run all strategies."""
        strategies = [
            "dense",
            "bm25",
            "hybrid_rrf",
            "hybrid_reranker_ce",
            "hybrid_reranker_colbert",
        ]

        print(f"\n{'='*70}")
        print("Retrieval Strategy Benchmark")
        print(f"{'='*70}")
        print(f"Queries: {len(BENCHMARK_QAS)}")
        print(f"Documents: {len(BENCHMARK_DOCS)}")

        results = []
        for strategy in strategies:
            metrics = await self.run_strategy(strategy)
            results.append(metrics)

            print(f"\n✅ {strategy.upper()}")
            print(f"  Recall@10:     {metrics.recall_at_10:.3f}")
            print(f"  MRR@10:        {metrics.mrr_at_10:.3f}")
            print(f"  nDCG@10:       {metrics.ndcg_at_10:.3f}")
            print(f"  Latency p99:   {metrics.p99_latency_ms:.1f} ms")

        return results


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval Reranking Benchmark")
    parser.add_argument("--n-queries", type=int, default=5, help="Number of queries")
    parser.add_argument("--output", type=str, default="eval_reports", help="Output directory")

    args = parser.parse_args()

    config = BenchConfig(n_queries=args.n_queries)
    bench = RetrievalBench(config)

    results = await bench.run_all()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat().replace(":", "-")
    output_file = output_dir / f"retrieval_bench_{timestamp}.json"

    results_dict = {
        "config": asdict(config),
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().isoformat(),
    }

    output_file.write_text(json.dumps(results_dict, indent=2))
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
