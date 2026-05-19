"""Base interface for LLM inference backends (vLLM, Ollama)."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InferenceError(Exception):
    """Raised when inference fails."""

    pass


@dataclass
class GenerationResult:
    """Result of text generation."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = "stop"  # "stop" | "length" | "error"


class BackendConfig(BaseModel):
    """Base config for inference backends."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(..., description="Model ID")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=512, description="Max generation tokens")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class BaseInferenceBackend(ABC):
    """Abstract base for inference backends."""

    def __init__(self, config: BackendConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def chat(self, messages: List[dict], **kwargs) -> GenerationResult:
        """Generate text from chat messages."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass
