#!/usr/bin/env python
"""
INFRACORE — Evaluation Benchmark CLI

Purpose: Run the eval framework on a sample RAG dataset and produce a report
Usage: python benchmarks/eval_bench.py [--samples 20] [--output eval_reports/]

Hardcoded sample dataset covering good, medium, and bad quality levels.
"""

import argparse
import asyncio
from dataclasses import dataclass

from src.infracore.eval.base import EvalSample
from src.infracore.eval.evaluator import EvalConfig, RAGEvaluator


# Sample dataset: good, medium, and bad quality samples
SAMPLE_DATASET = [
    # Good samples (high scores expected)
    EvalSample(
        query="What is a vector database?",
        context="A vector database is a system designed to store, index, and search vector embeddings efficiently. Popular options include Qdrant, Weaviate, and Pinecone. Vector databases use techniques like HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search.",
        answer="A vector database is a system designed to store, index, and search vector embeddings efficiently.",
        ground_truth="A vector database is a system designed to store, index, and search vector embeddings efficiently.",
    ),
    EvalSample(
        query="What is RAG (Retrieval Augmented Generation)?",
        context="RAG stands for Retrieval Augmented Generation. It combines a retriever component that fetches relevant documents with a generator component (usually an LLM) that produces answers. This approach improves factuality and reduces hallucination by grounding answers in retrieved context.",
        answer="RAG is a technique that combines document retrieval with LLM generation to improve factuality.",
        ground_truth="RAG combines retrieval of relevant documents with LLM generation to produce grounded answers.",
    ),
    EvalSample(
        query="What does HNSW stand for?",
        context="HNSW stands for Hierarchical Navigable Small World. It is an algorithm used for approximate nearest neighbor search. HNSW provides fast search with O(log N) complexity while maintaining high recall. It is widely used in vector databases like Qdrant.",
        answer="HNSW stands for Hierarchical Navigable Small World, an algorithm for approximate nearest neighbor search.",
        ground_truth="HNSW stands for Hierarchical Navigable Small World, used for approximate nearest neighbor search.",
    ),
    # Medium samples (partial context match, moderate scores)
    EvalSample(
        query="What is embedding?",
        context="Embeddings are dense vector representations of text learned by neural networks. Common embedding models include BERT, sentence-transformers, and BGE-M3. Embeddings capture semantic meaning and enable similarity-based search.",
        answer="Embeddings are learned vector representations that capture semantic information.",
        ground_truth="Embeddings are dense vector representations of text that capture semantic meaning.",
    ),
    EvalSample(
        query="How does semantic search work?",
        context="Semantic search compares meaning rather than keywords. It converts queries and documents into embeddings, then finds documents with high cosine similarity to the query embedding. This approach works across different phrasings of the same meaning.",
        answer="Semantic search converts text to embeddings and finds similar vectors.",
        ground_truth="Semantic search converts queries and documents to embeddings and finds vectors with high cosine similarity.",
    ),
    EvalSample(
        query="What is the purpose of evaluation metrics?",
        context="Evaluation metrics measure the quality of RAG systems. Metrics like BLEU, ROUGE, and custom faithfulness probes assess whether generated answers are relevant, factual, and grounded in the retrieved context. Metrics help catch regressions in production systems.",
        answer="Metrics help measure system quality and catch performance regressions.",
        ground_truth="Metrics assess whether answers are relevant, factual, and grounded in context.",
    ),
    # Bad samples (hallucinations, context mismatch, low scores expected)
    EvalSample(
        query="What is machine learning?",
        context="Qdrant is a vector database company founded in 2021. The latest version features optimized disk usage and new query APIs.",
        answer="Machine learning is the process of training neural networks on unstructured data.",
        ground_truth="Machine learning is a field of artificial intelligence where algorithms learn patterns from data.",
    ),
    EvalSample(
        query="How many parameters does GPT-4 have?",
        context="The capital of France is Paris. Paris is known for the Eiffel Tower and museums.",
        answer="GPT-4 has 100 billion parameters and can process images.",
        ground_truth="GPT-4 is a multimodal model released by OpenAI in 2023.",
    ),
    EvalSample(
        query="What is deep learning?",
        context="Coffee is made from roasted coffee beans. Different regions produce different flavor profiles.",
        answer="Deep learning involves artificial neurons organized in multiple layers.",
        ground_truth="Deep learning is a subset of machine learning using neural networks with multiple layers.",
    ),
    EvalSample(
        query="What does CPU stand for?",
        context="The weather in Seattle is typically cloudy and rainy. It rains about 150 days per year.",
        answer="CPU stands for Central Processing Unit and is the brain of a computer.",
        ground_truth="CPU stands for Central Processing Unit.",
    ),
]


async def run_benchmark(num_samples: int, output_dir: str) -> None:
    """Run evaluation benchmark."""
    print(f"\n🚀 InfraCore Evaluation Benchmark")
    print(f"   Samples: {num_samples}")
    print(f"   Output: {output_dir}\n")

    # Use first N samples
    samples = SAMPLE_DATASET[:num_samples]

    # Create evaluator
    config = EvalConfig(
        metrics=[
            "answer_relevance",
            "context_recall",
            "context_precision",
            "faithfulness",
            "answer_correctness",
        ],
        output_dir=output_dir,
        report_format="markdown",
        fail_threshold=0.70,
    )
    evaluator = RAGEvaluator(config)

    # Run evaluation
    print(f"⏳ Evaluating {num_samples} samples across 5 metrics...")
    report = await evaluator.evaluate(samples)

    # Print summary
    evaluator.print_summary(report)

    # Save report
    filepath = await evaluator.save_report(report)
    print(f"📄 Report saved to: {filepath}\n")

    # Verify results
    print(f"✅ Benchmark completed successfully")
    print(f"   Samples evaluated: {report.samples_count}")
    print(f"   Duration: {report.duration_ms:.0f}ms")
    print(f"   Status: {'PASSED ✅' if report.passed else 'FAILED ❌'}\n")

    # Assert evaluation ran without errors
    assert report.samples_count == num_samples, "Sample count mismatch"
    assert len(report.per_metric_scores) == 5, "Expected 5 metrics"
    assert all(
        0.0 <= score <= 1.0 for score in report.per_metric_scores.values()
    ), "Scores out of range"

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="InfraCore Evaluation Benchmark")
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (max 10)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_reports", help="Output directory for reports"
    )

    args = parser.parse_args()

    # Clamp samples to max 10
    num_samples = min(args.samples, 10)

    # Run benchmark
    asyncio.run(run_benchmark(num_samples, args.output))


if __name__ == "__main__":
    main()
