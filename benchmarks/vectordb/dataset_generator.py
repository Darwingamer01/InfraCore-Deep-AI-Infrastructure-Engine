#!/usr/bin/env python
"""
INFRACORE — VectorDB Benchmark Dataset Generator

Purpose: Generate reproducible synthetic vector datasets for benchmarking
No external download required — all vectors generated locally

Supports 3 distributions: uniform, clustered, gaussian
Computes ground truth via brute-force cosine similarity
Saves as .npz for efficient loading
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    n_vectors: int
    n_dims: int
    n_queries: int = 100
    seed: int = 42
    distribution: Literal["uniform", "clustered", "gaussian"] = "gaussian"


@dataclass
class DatasetResult:
    """Generated dataset with ground truth."""

    vectors: np.ndarray  # shape (n_vectors, n_dims), float32, L2 normalized
    queries: np.ndarray  # shape (n_queries, n_dims), float32, L2 normalized
    ground_truth: np.ndarray  # shape (n_queries, 10), int32 — top-10 neighbor indices


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def generate_dataset(config: DatasetConfig) -> DatasetResult:
    """
    Generate synthetic vector dataset.

    Args:
        config: DatasetConfig with distribution, size, dims

    Returns:
        DatasetResult with vectors, queries, ground_truth
    """
    np.random.seed(config.seed)

    print(f"🔄 Generating {config.n_vectors:,} vectors ({config.n_dims}D, {config.distribution})")

    # Generate vectors based on distribution
    if config.distribution == "uniform":
        vectors = np.random.uniform(-1, 1, size=(config.n_vectors, config.n_dims))
    elif config.distribution == "gaussian":
        vectors = np.random.randn(config.n_vectors, config.n_dims)
    elif config.distribution == "clustered":
        # 20 cluster centers
        n_clusters = 20
        cluster_centers = np.random.randn(n_clusters, config.n_dims)
        cluster_centers = _l2_normalize(cluster_centers)

        vectors = []
        vectors_per_cluster = config.n_vectors // n_clusters
        for i in range(n_clusters):
            center = cluster_centers[i]
            # Add noise to center
            noise = np.random.randn(vectors_per_cluster, config.n_dims) * 0.1
            cluster_vectors = center + noise
            vectors.append(cluster_vectors)

        vectors = np.vstack(vectors)
        # Pad to exact n_vectors
        if len(vectors) < config.n_vectors:
            padding = np.random.randn(config.n_vectors - len(vectors), config.n_dims)
            vectors = np.vstack([vectors, padding])
        else:
            vectors = vectors[: config.n_vectors]
    else:
        raise ValueError(f"Unknown distribution: {config.distribution}")

    # L2 normalize
    vectors = _l2_normalize(vectors)

    # Generate queries
    print(f"🔄 Generating {config.n_queries} query vectors")
    if config.distribution == "clustered":
        # Queries from cluster centers
        queries = np.random.randn(config.n_queries, config.n_dims)
    else:
        queries = np.random.randn(config.n_queries, config.n_dims)

    queries = _l2_normalize(queries)

    # Compute ground truth via brute-force cosine similarity
    print(f"⏳ Computing ground truth (brute-force top-10)...")
    start_time = time.time()

    ground_truth = np.zeros((config.n_queries, 10), dtype=np.int32)

    for i in tqdm(range(config.n_queries), desc="Ground truth"):
        query = queries[i]
        # Cosine similarity = dot product (since L2 normalized)
        similarities = np.dot(vectors, query)
        # Get top 10 indices
        top_10_indices = np.argsort(similarities)[-10:][::-1]
        ground_truth[i] = top_10_indices

    elapsed = time.time() - start_time
    print(f"✅ Ground truth computed in {elapsed:.2f}s")

    return DatasetResult(vectors=vectors, queries=queries, ground_truth=ground_truth)


def save_dataset(result: DatasetResult, path: str) -> None:
    """
    Save dataset to .npz file.

    Args:
        result: DatasetResult
        path: Output file path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving to {path}")
    np.savez_compressed(
        path,
        vectors=result.vectors,
        queries=result.queries,
        ground_truth=result.ground_truth,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {file_size_mb:.1f} MB")


def load_dataset(path: str) -> DatasetResult:
    """
    Load dataset from .npz file.

    Args:
        path: Path to .npz file

    Returns:
        DatasetResult
    """
    data = np.load(path)
    return DatasetResult(
        vectors=data["vectors"],
        queries=data["queries"],
        ground_truth=data["ground_truth"],
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="InfraCore VectorDB Benchmark Dataset Generator"
    )
    parser.add_argument(
        "--n", type=int, default=10000, help="Number of vectors to generate"
    )
    parser.add_argument("--dims", type=int, default=384, help="Vector dimensionality")
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of query vectors"
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="gaussian",
        choices=["uniform", "clustered", "gaussian"],
        help="Vector distribution",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/bench",
        help="Output directory for dataset files",
    )

    args = parser.parse_args()

    # Generate config
    config = DatasetConfig(
        n_vectors=args.n,
        n_dims=args.dims,
        n_queries=args.queries,
        seed=args.seed,
        distribution=args.dist,
    )

    print(f"\n{'='*70}")
    print(f"InfraCore VectorDB Dataset Generator")
    print(f"{'='*70}")
    print(f"  Vectors: {config.n_vectors:,}")
    print(f"  Dimensions: {config.n_dims}")
    print(f"  Queries: {config.n_queries}")
    print(f"  Distribution: {config.distribution}")
    print(f"{'='*70}\n")

    # Generate dataset
    result = generate_dataset(config)

    # Save dataset
    filename = f"dataset_n{args.n}_d{args.dims}_{args.dist}.npz"
    output_path = str(Path(args.out) / filename)
    save_dataset(result, output_path)

    print(f"\n✅ Dataset generation complete!")
    print(f"   Vectors shape: {result.vectors.shape}")
    print(f"   Queries shape: {result.queries.shape}")
    print(f"   Ground truth shape: {result.ground_truth.shape}")
    print(f"   Saved to: {output_path}\n")


if __name__ == "__main__":
    main()
