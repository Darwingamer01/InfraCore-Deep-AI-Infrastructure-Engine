"""Test suite for OllamaBackend, VLLMBackend, and InferenceRouter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from httpx import Response

from src.infracore.inference.backend_base import GenerationResult, InferenceError
from src.infracore.inference.ollama_backend import OllamaBackend, OllamaConfig
from src.infracore.inference.router import InferenceRouter, RouterConfig
from src.infracore.inference.vllm_backend import VLLMBackend, VLLMConfig


# ============================================================================
# OllamaBackend Tests (1-7)
# ============================================================================


@pytest.mark.asyncio
async def test_ollama_generate_returns_result():
    """Test 1: generate() returns GenerationResult with text."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        route = respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "response": "This is a test response.",
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "done_reason": "stop",
                },
            )
        )

        result = await backend.generate("Test prompt")

        assert isinstance(result, GenerationResult)
        assert result.text == "This is a test response."
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15


@pytest.mark.asyncio
async def test_ollama_generate_parses_tokens():
    """Test 2: generate() correctly parses token counts."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "response": "Response text",
                    "prompt_eval_count": 25,
                    "eval_count": 100,
                    "done_reason": "length",
                },
            )
        )

        result = await backend.generate("Prompt")

        assert result.prompt_tokens == 25
        assert result.completion_tokens == 100
        assert result.total_tokens == 125


@pytest.mark.asyncio
async def test_ollama_generate_finish_reason():
    """Test 3: generate() sets finish_reason from done_reason."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "response": "Text",
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                    "done_reason": "stop",
                },
            )
        )

        result = await backend.generate("Prompt")
        assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_ollama_is_available_true():
    """Test 4: is_available() returns True on 200 from /api/tags."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.get("http://localhost:11434/api/tags").mock(return_value=Response(200, json={}))

        available = await backend.is_available()
        assert available is True


@pytest.mark.asyncio
async def test_ollama_is_available_false_on_error():
    """Test 5: is_available() returns False on connection error."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.get("http://localhost:11434/api/tags").mock(side_effect=ConnectionError())

        available = await backend.is_available()
        assert available is False


@pytest.mark.asyncio
async def test_ollama_generate_raises_on_error():
    """Test 6: generate() raises InferenceError on HTTP 500."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        with pytest.raises(InferenceError):
            await backend.generate("Prompt")


@pytest.mark.asyncio
async def test_ollama_latency_ms_positive():
    """Test 7: latency_ms > 0 in GenerationResult."""
    config = OllamaConfig(model="llama3.1:8b")
    backend = OllamaBackend(config)

    with respx.mock:
        respx.post("http://localhost:11434/api/generate").mock(
            return_value=Response(
                200,
                json={
                    "response": "Text",
                    "prompt_eval_count": 5,
                    "eval_count": 3,
                    "done_reason": "stop",
                },
            )
        )

        result = await backend.generate("Prompt")
        assert result.latency_ms > 0.0


# ============================================================================
# VLLMBackend Tests (8-11)
# ============================================================================


@pytest.mark.asyncio
async def test_vllm_generate_parses_openai_response():
    """Test 8: generate() parses OpenAI-format response."""
    config = VLLMConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    backend = VLLMBackend(config)

    with respx.mock:
        respx.post("http://localhost:8000/v1/completions").mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"text": "Generated text here"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                    },
                },
            )
        )

        result = await backend.generate("Prompt")

        assert result.text == "Generated text here"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 8


@pytest.mark.asyncio
async def test_vllm_chat_parses_message():
    """Test 9: chat() parses choices[0].message.content."""
    config = VLLMConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    backend = VLLMBackend(config)

    with respx.mock:
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "Chat response"}}
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 5,
                    },
                },
            )
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = await backend.chat(messages)

        assert result.text == "Chat response"
        assert result.prompt_tokens == 20


@pytest.mark.asyncio
async def test_vllm_is_available_true():
    """Test 10: is_available() returns True on 200 from /health."""
    config = VLLMConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    backend = VLLMBackend(config)

    with respx.mock:
        respx.get("http://localhost:8000/health").mock(return_value=Response(200, json={}))

        available = await backend.is_available()
        assert available is True


@pytest.mark.asyncio
async def test_vllm_list_models():
    """Test 11: list_models() returns list of model names."""
    config = VLLMConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    backend = VLLMBackend(config)

    with respx.mock:
        respx.get("http://localhost:8000/v1/models").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {"id": "llama-3.1-8b"},
                        {"id": "mistral-7b"},
                    ]
                },
            )
        )

        models = await backend.list_models()
        assert "llama-3.1-8b" in models
        assert "mistral-7b" in models


# ============================================================================
# InferenceRouter Tests (12-15)
# ============================================================================


@pytest.mark.asyncio
async def test_router_uses_primary_when_available():
    """Test 12: Uses primary when primary.is_available() returns True."""
    config = RouterConfig(primary="vllm", fallback="ollama")

    primary = AsyncMock()
    primary.is_available.return_value = True
    primary.generate.return_value = GenerationResult(text="From primary")

    fallback = AsyncMock()

    router = InferenceRouter(config, primary, fallback)

    result = await router.generate("Prompt")

    assert result.text == "From primary"
    primary.generate.assert_called_once()
    fallback.generate.assert_not_called()


@pytest.mark.asyncio
async def test_router_falls_back_when_primary_unavailable():
    """Test 13: Falls back when primary unavailable."""
    config = RouterConfig(primary="vllm", fallback="ollama")

    primary = AsyncMock()
    primary.is_available.return_value = False

    fallback = AsyncMock()
    fallback.is_available.return_value = True
    fallback.generate.return_value = GenerationResult(text="From fallback")

    router = InferenceRouter(config, primary, fallback)

    result = await router.generate("Prompt")

    assert result.text == "From fallback"
    fallback.generate.assert_called_once()


@pytest.mark.asyncio
async def test_router_raises_when_both_unavailable():
    """Test 14: Raises InferenceError when both backends unavailable."""
    config = RouterConfig(primary="vllm", fallback="ollama")

    primary = AsyncMock()
    primary.is_available.return_value = False

    fallback = AsyncMock()
    fallback.is_available.return_value = False

    router = InferenceRouter(config, primary, fallback)

    with pytest.raises(InferenceError):
        await router.generate("Prompt")


@pytest.mark.asyncio
async def test_router_caches_availability():
    """Test 15: Caches availability — is_available() not called twice within 30s."""
    config = RouterConfig(primary="vllm", fallback="ollama")

    primary = AsyncMock()
    primary.is_available.return_value = True
    primary.generate.return_value = GenerationResult(text="Response")

    fallback = AsyncMock()

    router = InferenceRouter(config, primary, fallback)

    # First call
    await router.generate("Prompt 1")
    first_call_count = primary.is_available.call_count

    # Second call (should use cache)
    await router.generate("Prompt 2")
    second_call_count = primary.is_available.call_count

    # is_available() should only be called once due to cache
    assert first_call_count == second_call_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
