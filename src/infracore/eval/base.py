"""
BaseEvaluator – Abstract interface for evaluation.

Handles metrics, faithfulness probes, per-sample scores.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class EvalConfig(BaseModel):
    """Pydantic config for evaluation."""

    model_config = ConfigDict(frozen=True)

    metrics: List[str] = Field(
        default=["bleu", "rouge", "faithfulness"], description="Metrics to compute"
    )
    batch_size: int = Field(default=8, description="Eval batch size")


@dataclass
class EvalSample:
    """Single eval sample."""

    query: str
    context: str
    answer: str
    ground_truth: str


@dataclass
class EvalReport:
    """Evaluation results."""

    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample_scores: List[Dict[str, Any]] = field(default_factory=list)
    total_samples: int = 0


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation.

    Subclasses must implement:
    - evaluate(samples: List[EvalSample]) -> EvalReport
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    @abstractmethod
    async def evaluate(self, samples: List[EvalSample]) -> EvalReport:
        """
        Evaluate a list of samples.

        Args:
            samples: List of EvalSample with query, context, answer, ground_truth

        Returns:
            EvalReport with aggregated metrics + per-sample scores
        """
        pass
