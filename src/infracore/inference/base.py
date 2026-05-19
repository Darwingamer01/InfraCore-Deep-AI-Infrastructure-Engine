"""
BaseInference – Abstract interface for LLM inference.

Handles generation, streaming, token counting. Pure async.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    """Pydantic config for inference."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(..., description="Model ID or path")
    max_tokens: int = Field(default=256, description="Max generation length")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.95, description="Nucleus sampling")


@dataclass
class GenerationResult:
    """Result of inference."""

    text: str
    finish_reason: str  # "length", "stop", "eos"
    tokens_generated: int
    tokens_total: int


class BaseInference(ABC):
    """
    Abstract base class for LLM inference.

    Subclasses must implement:
    - generate(prompt: str) -> GenerationResult
    - generate_stream(prompt: str) -> AsyncGenerator[str, None]
    """

    def __init__(self, config: InferenceConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, stop_sequences: Optional[List[str]] = None) -> GenerationResult:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            stop_sequences: Optional stop tokens

        Returns:
            GenerationResult with text + token counts
        """
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, stop_sequences: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream generation tokens.

        Args:
            prompt: Input prompt
            stop_sequences: Optional stop tokens

        Yields:
            Token strings
        """
        pass
