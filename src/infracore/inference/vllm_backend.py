"""VLLMBackend - High-throughput inference via vLLM OpenAI-compatible API."""

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


class VLLMConfig(BackendConfig):
    """vLLM-specific configuration."""

    model_config = ConfigDict(frozen=True)

    base_url: str = Field(default="http://localhost:8000", description="vLLM API base URL")
    top_p: float = Field(default=0.95, description="Nucleus sampling parameter")
    api_key: str = Field(default="EMPTY", description="API key for vLLM")


class VLLMBackend(BaseInferenceBackend):
    """vLLM inference backend using OpenAI-compatible API."""

    def __init__(self, config: VLLMConfig):
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
        """Generate text from prompt using vLLM /v1/completions endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.base_url}/v1/completions",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                    },
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )

                if response.status_code != 200:
                    raise InferenceError(f"vLLM error: {response.status_code} {response.text}")

                data = response.json()

                # Parse OpenAI-compatible response
                choices = data.get("choices", [])
                if not choices:
                    raise InferenceError("No choices in vLLM response")

                text = choices[0].get("text", "")
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                self._requests_counter.labels(backend="vllm", model=self.config.model).inc()
                self._latency_histogram.labels(backend="vllm", model=self.config.model).observe(
                    latency_ms / 1000
                )
                self._tokens_counter.labels(backend="vllm", type="prompt").inc(prompt_tokens)
                self._tokens_counter.labels(backend="vllm", type="completion").inc(
                    completion_tokens
                )

                return GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    model=self.config.model,
                    finish_reason="stop",
                )

        except httpx.ConnectError as e:
            raise InferenceError(f"Failed to connect to vLLM: {str(e)}") from e
        except Exception as e:
            raise InferenceError(f"vLLM generation failed: {str(e)}") from e

    async def chat(self, messages: List[dict], **kwargs) -> GenerationResult:
        """Generate text from chat messages using vLLM /v1/chat/completions endpoint."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.base_url}/v1/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                    },
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )

                if response.status_code != 200:
                    raise InferenceError(f"vLLM error: {response.status_code} {response.text}")

                data = response.json()

                # Parse OpenAI-compatible response
                choices = data.get("choices", [])
                if not choices:
                    raise InferenceError("No choices in vLLM response")

                message = choices[0].get("message", {})
                text = message.get("content", "")
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                latency_ms = (time.time() - start_time) * 1000

                # Record metrics
                self._requests_counter.labels(backend="vllm", model=self.config.model).inc()
                self._latency_histogram.labels(backend="vllm", model=self.config.model).observe(
                    latency_ms / 1000
                )
                self._tokens_counter.labels(backend="vllm", type="prompt").inc(prompt_tokens)
                self._tokens_counter.labels(backend="vllm", type="completion").inc(
                    completion_tokens
                )

                return GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    model=self.config.model,
                    finish_reason="stop",
                )

        except httpx.ConnectError as e:
            raise InferenceError(f"Failed to connect to vLLM: {str(e)}") from e
        except Exception as e:
            raise InferenceError(f"vLLM chat failed: {str(e)}") from e

    async def is_available(self) -> bool:
        """Check if vLLM server is available."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(f"{self.config.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List available models on vLLM server."""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.config.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )

                if response.status_code != 200:
                    raise InferenceError(f"Failed to list models: {response.status_code}")

                data = response.json()
                models = data.get("data", [])
                return [m.get("id", "") for m in models]

        except Exception as e:
            raise InferenceError(f"Failed to list vLLM models: {str(e)}") from e
