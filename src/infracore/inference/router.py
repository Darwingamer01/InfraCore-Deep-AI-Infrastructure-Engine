"""InferenceRouter - Smart backend selection with fallback."""

import asyncio
import time
from typing import List, Literal, Optional

from prometheus_client import Gauge
from pydantic import ConfigDict, Field

from src.infracore.inference.backend_base import (
    BackendConfig,
    BaseInferenceBackend,
    GenerationResult,
    InferenceError,
)


class RouterConfig(BackendConfig):
    """Router configuration."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(default="router", description="Router model name")
    primary: Literal["vllm", "ollama"] = Field(default="vllm", description="Primary backend")
    fallback: Optional[Literal["vllm", "ollama"]] = Field(
        default="ollama", description="Fallback backend"
    )
    health_check_timeout: float = Field(default=3.0, description="Health check timeout")


class InferenceRouter(BaseInferenceBackend):
    """Router that selects the best available backend with fallback."""

    def __init__(
        self,
        config: RouterConfig,
        primary_backend: BaseInferenceBackend,
        fallback_backend: Optional[BaseInferenceBackend] = None,
    ):
        super().__init__(config)
        self.config = config
        self.primary = primary_backend
        self.fallback = fallback_backend

        # Availability cache: (timestamp, is_available)
        self._primary_available: Optional[tuple[float, bool]] = None
        self._fallback_available: Optional[tuple[float, bool]] = None
        self._cache_duration = 30.0  # seconds
        self._lock = asyncio.Lock()

        # Prometheus gauge for active backend
        self._active_backend_gauge = Gauge(
            "active_backend",
            "Active inference backend (1=vllm, 0=ollama)",
        )

    async def _check_backend_available(
        self, backend: BaseInferenceBackend, backend_name: str
    ) -> bool:
        """Check if backend is available with timeout."""
        try:
            task = asyncio.create_task(backend.is_available())
            available = await asyncio.wait_for(task, timeout=self.config.health_check_timeout)
            return available
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    async def _get_available_backend(self) -> tuple[BaseInferenceBackend, str]:
        """
        Select the best available backend with fallback.

        Returns:
            (backend, name) tuple
        """
        async with self._lock:
            now = time.time()

            # Check primary
            primary_cached = (
                self._primary_available
                and now - self._primary_available[0] < self._cache_duration
            )
            if not primary_cached:
                primary_is_available = await self._check_backend_available(
                    self.primary, self.config.primary
                )
                self._primary_available = (now, primary_is_available)
            else:
                primary_is_available = self._primary_available[1]

            if primary_is_available:
                self._active_backend_gauge.set(
                    1.0 if self.config.primary == "vllm" else 0.0
                )
                return self.primary, self.config.primary

            # Check fallback
            if not self.fallback:
                raise InferenceError("No backend available (primary down, no fallback)")

            fallback_cached = (
                self._fallback_available
                and now - self._fallback_available[0] < self._cache_duration
            )
            if not fallback_cached:
                fallback_is_available = await self._check_backend_available(
                    self.fallback, self.config.fallback
                )
                self._fallback_available = (now, fallback_is_available)
            else:
                fallback_is_available = self._fallback_available[1]

            if fallback_is_available:
                self._active_backend_gauge.set(
                    1.0 if self.config.fallback == "vllm" else 0.0
                )
                return self.fallback, self.config.fallback

            raise InferenceError("No backend available (both primary and fallback down)")

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using the best available backend."""
        backend, backend_name = await self._get_available_backend()
        return await backend.generate(prompt, **kwargs)

    async def chat(self, messages: List[dict], **kwargs) -> GenerationResult:
        """Chat using the best available backend."""
        backend, backend_name = await self._get_available_backend()
        return await backend.chat(messages, **kwargs)

    async def get_active_backend(self) -> str:
        """Get name of currently active backend."""
        _, backend_name = await self._get_available_backend()
        return backend_name

    async def is_available(self) -> bool:
        """Check if any backend is available."""
        try:
            await self._get_available_backend()
            return True
        except InferenceError:
            return False

    async def list_models(self) -> List[str]:
        """List models from the active backend."""
        backend, _ = await self._get_available_backend()
        return await backend.list_models()
