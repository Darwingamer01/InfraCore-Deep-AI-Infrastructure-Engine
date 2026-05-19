"""
Test module imports and base class structure.
"""

import pytest


@pytest.mark.asyncio
async def test_imports():
    """Verify all modules import correctly."""
    from src.infracore.chunking.base import BaseChunker, ChunkConfig, Chunk
    from src.infracore.embedding.base import BaseEmbedder, EmbedConfig
    from src.infracore.vectordb.base import BaseVectorStore, VectorStoreConfig, SearchResult
    from src.infracore.retrieval.base import BaseRetriever, RetrieverConfig, RetrievalResult
    from src.infracore.agents.base import BaseAgent, AgentConfig, BaseTool
    from src.infracore.eval.base import BaseEvaluator, EvalConfig, EvalSample
    from src.infracore.ingest.base import BaseIngester, IngestConfig, IngestedDocument
    from src.infracore.inference.base import BaseInference, InferenceConfig, GenerationResult

    assert BaseChunker is not None
    assert BaseEmbedder is not None
    assert BaseVectorStore is not None
    assert BaseRetriever is not None
    assert BaseAgent is not None
    assert BaseEvaluator is not None
    assert BaseIngester is not None
    assert BaseInference is not None


def test_pydantic_configs():
    """Verify Pydantic configs are valid."""
    from src.infracore.chunking.base import ChunkConfig
    from src.infracore.embedding.base import EmbedConfig

    config1 = ChunkConfig(strategy="fixed")
    assert config1.strategy == "fixed"

    config2 = EmbedConfig(model_name="test-model")
    assert config2.model_name == "test-model"
