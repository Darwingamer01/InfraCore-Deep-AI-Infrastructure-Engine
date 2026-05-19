"""OllamaBackend - Local inference via Ollama HTTP API."""

import time
from typing import List, Optional

import httpx
from prometheus_client import Counter as PrometheusCounter, Histogram
from pydantic import ConfigDict, Field

from src.infracore.inference.backend_base import (
    BackendConfig,
    BaseInferenceBackend,
    GenerationResult,
    InferenceError,
)


class OllamaConfig(BackendConfig):
    """Ollama-specific configuration."""

    model_config = ConfigDict(frozen=True)

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    stream: bool = Field(default=False, description="Stream responses")
    context_window: int = Field(default=4096, description="Context window size")


class OllamaBackend(BaseInferenceBackend):
    """Ollama inference backend."""

    def __init__(self, config: OllamaConfig):
        super().__init__(config)
        self.config = config

        # Prometheus metrics
        self._requests_counter = PrometheusCounter(
            "inference_requests_total",
            "Total inference requests",
            labelnames=["backend", "model"],
        )
        self._latency_histogram = Histogram(
            "inference_latency_seconds",
            "Inference latency",
            labelnames=["backend", "model"],
        )
        self._tokens_counter = PrometheusCounter(
            "inference_tokens_total",
            "Total tokens generated",
            labelnames=["backend", "type"],
        )

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text from prompt using Ollama /api/generate endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.base_url}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                )

                if response.status_code != 200:
                    raise InferenceError(f"Ollama error: {response.status_code} {response.text}")

                data = response.json()

                # Parse response
                text = data.get("response", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                finish_reason = data.get("done_reason", "stop")

                latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                self._requests_counter.labels(backend="ollama", model=self.config.model).inc()
                self._latency_histogram.labels(backend="ollama", model=self.config.model).observe(
                    latency_ms / 1000
                )
                self._tokens_counter.labels(backend="ollama", type="prompt").inc(prompt_tokens)
                self._tokens_counter.labels(backend="ollama", type="completion").inc(
                    completion_tokens
                )

                return GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    model=self.config.model,
                    finish_reason=finish_reason,
                )

        except httpx.ConnectError as e:
            raise InferenceError(f"Failed to connect to Ollama: {str(e)}") from e
        except Exception as e:
            raise InferenceError(f"Ollama generation failed: {str(e)}") from e

    async def chat(self, messages: List[dict], **kwargs) -> GenerationResult:
        """Generate text from chat messages using Ollama /api/chat endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.base_url}/api/chat",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                )

                if response.status_code != 200:
                    raise InferenceError(f"Ollama error: {response.status_code} {response.text}")

                data = response.json()

                # Parse response
                message = data.get("message", {})
                text = message.get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                finish_reason = data.get("done_reason", "stop")

                latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                self._requests_counter.labels(backend="ollama", model=self.config.model).inc()
                self._latency_histogram.labels(backend="ollama", model=self.config.model).observe(
                    latency_ms / 1000
                )
                self._tokens_counter.labels(backend="ollama", type="prompt").inc(prompt_tokens)
                self._tokens_counter.labels(backend="ollama", type="completion").inc(
                    completion_tokens
                )

                return GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    model=self.config.model,
                    finish_reason=finish_reason,
                )

        except httpx.ConnectError as e:
            raise InferenceError(f"Failed to connect to Ollama: {str(e)}") from e
        except Exception as e:
            raise InferenceError(f"Ollama chat failed: {str(e)}") from e

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{self.config.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{self.config.base_url}/api/tags")

                if response.status_code != 200:
                    raise InferenceError(f"Failed to list models: {response.status_code}")

                data = response.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models]

        except Exception as e:
            raise InferenceError(f"Failed to list Ollama models: {str(e)}") from e
