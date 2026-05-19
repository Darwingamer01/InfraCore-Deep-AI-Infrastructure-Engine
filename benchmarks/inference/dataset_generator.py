#!/usr/bin/env python
"""
INFRACORE — Inference Benchmark Prompt Dataset Generator

Purpose: Generate 50 prompts across 3 categories (short, medium, long)
for reproducible inference benchmarking.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Prompt:
    """Single benchmark prompt."""

    id: int
    text: str
    category: str  # "short", "medium", "long"
    expected_tokens: int  # Approximate expected response tokens


SHORT_PROMPTS = [
    ("What is 2+2?", 10),
    ("List 3 colors", 15),
    ("Who wrote Hamlet?", 12),
    ("What is water?", 15),
    ("Define photosynthesis", 20),
    ("What is Python?", 15),
    ("Spell 'encyclopedia'", 12),
    ("What is AI?", 18),
    ("Who is Einstein?", 15),
    ("What is HTML?", 18),
    ("Name 5 planets", 20),
    ("What is Bitcoin?", 18),
    ("Who invented electricity?", 15),
    ("What is DNA?", 18),
    ("List 3 programming languages", 20),
]

MEDIUM_PROMPTS = [
    (
        "Explain why the sky is blue in simple terms",
        50,
    ),
    (
        "What are the main causes of climate change? List them.",
        60,
    ),
    (
        "How does photosynthesis work? Explain step by step.",
        70,
    ),
    (
        "Describe the water cycle and its importance",
        65,
    ),
    (
        "What are the benefits of exercise? Provide at least 3",
        60,
    ),
    (
        "Explain how machine learning works at a high level",
        70,
    ),
    (
        "What are the differences between DNA and RNA?",
        55,
    ),
    (
        "How does the internet work? Give a brief overview",
        65,
    ),
    (
        "What is the difference between correlation and causation?",
        60,
    ),
    (
        "Explain quantum computing to someone unfamiliar with it",
        75,
    ),
    (
        "What are the major principles of object-oriented programming?",
        70,
    ),
    (
        "Describe the life cycle of a butterfly",
        50,
    ),
    (
        "What is the difference between weather and climate?",
        55,
    ),
    (
        "How do vaccines work? Explain the process",
        65,
    ),
    (
        "What are the main causes of poverty and how can it be addressed?",
        80,
    ),
]

LONG_PROMPTS = [
    (
        "Explain the theory of evolution in detail. Discuss natural selection, adaptation, and the evidence supporting the theory.",
        200,
    ),
    (
        "Describe the role of artificial intelligence in modern healthcare. Include diagnostics, treatment planning, and patient monitoring.",
        200,
    ),
    (
        "What are the major geopolitical challenges facing the world today? Discuss their causes, impacts, and potential solutions.",
        220,
    ),
    (
        "Explain how neural networks and deep learning have transformed the field of computer vision. Include practical applications.",
        200,
    ),
    (
        "Discuss the environmental impact of renewable energy sources versus fossil fuels. Include economic considerations.",
        210,
    ),
    (
        "Explain the principles of supply chain management and how disruptions affect global trade and economics.",
        200,
    ),
    (
        "What is blockchain technology and how does it work? Discuss applications beyond cryptocurrency.",
        190,
    ),
    (
        "Describe the impact of social media on society, including benefits, risks, and regulatory challenges.",
        210,
    ),
    (
        "Explain the scientific method and why it is fundamental to empirical research. Discuss peer review and scientific consensus.",
        200,
    ),
    (
        "What are the main ethical considerations in artificial intelligence development and deployment?",
        200,
    ),
    (
        "Describe the structure and function of the human immune system. Include innate and adaptive immunity.",
        210,
    ),
    (
        "Explain how cities can transition to sustainable urban development. Include transportation, housing, and energy.",
        220,
    ),
    (
        "Discuss the role of central banks in managing monetary policy and controlling inflation.",
        200,
    ),
    (
        "What are the major milestones in the history of computing and how have they shaped modern technology?",
        200,
    ),
    (
        "Explain the principles of organizational psychology and how they apply to modern workplace management.",
        210,
    ),
]


def generate_dataset(output_file: Path) -> None:
    """Generate and save prompt dataset."""
    prompts = []
    prompt_id = 0

    # Add short prompts
    for text, tokens in SHORT_PROMPTS:
        prompts.append(
            {
                "id": prompt_id,
                "text": text,
                "category": "short",
                "expected_tokens": tokens,
            }
        )
        prompt_id += 1

    # Add medium prompts
    for text, tokens in MEDIUM_PROMPTS:
        prompts.append(
            {
                "id": prompt_id,
                "text": text,
                "category": "medium",
                "expected_tokens": tokens,
            }
        )
        prompt_id += 1

    # Add long prompts
    for text, tokens in LONG_PROMPTS:
        prompts.append(
            {
                "id": prompt_id,
                "text": text,
                "category": "long",
                "expected_tokens": tokens,
            }
        )
        prompt_id += 1

    # Save JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"✅ Generated {len(prompts)} prompts")
    print(f"   - Short: {len(SHORT_PROMPTS)}")
    print(f"   - Medium: {len(MEDIUM_PROMPTS)}")
    print(f"   - Long: {len(LONG_PROMPTS)}")
    print(f"   Saved to: {output_file}")


def load_dataset(dataset_file: Path) -> List[dict]:
    """Load prompt dataset from JSONL."""
    prompts = []
    with open(dataset_file) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


if __name__ == "__main__":
    dataset_path = Path("data/bench/inference_prompts.jsonl")
    generate_dataset(dataset_path)
