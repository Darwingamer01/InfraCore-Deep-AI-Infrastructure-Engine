"""
INFRACORE – Implementation Template

Copy this structure for every new concrete implementation.
Pattern:
  - src/infracore/{module}/{name}.py
  - Inherit from Base{Class}
  - Implement all abstract methods
  - Emit Prometheus metrics
  - Full async, typed, documented
"""

from typing import List
import structlog

from src.infracore.chunking.base import BaseChunker, ChunkConfig, Chunk

logger = structlog.get_logger()


class ExampleChunker(BaseChunker):
    """
    Example chunker implementation.
    
    Demonstrates the required pattern:
    1. Inherit from ABC (BaseChunker)
    2. Store config in __init__
    3. Implement all abstract methods with async
    4. Return typed result objects (List[Chunk])
    5. Emit metrics (see structlog.context below)
    """

    def __init__(self, config: ChunkConfig):
        super().__init__(config)
        # Optional: Initialize resources (models, connections)

    async def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk implementation.
        
        Args:
            text: Raw text to chunk
            
        Returns:
            List[Chunk] with metadata
        """
        if not text:
            return []
        
        # Your implementation here
        chunks: List[Chunk] = []
        
        # Example: emit metrics via structlog
        with structlog.context.bind(
            strategy=self.config.strategy,
            num_chunks=len(chunks),
            total_chars=len(text),
        ):
            logger.info("chunking.complete")
        
        return chunks


# Test template (src/infracore/chunking/test_{name}.py):
"""
import pytest
from src.infracore.chunking.{name} import ExampleChunker
from src.infracore.chunking.base import ChunkConfig


@pytest.mark.asyncio
async def test_example_chunker_basic():
    config = ChunkConfig(strategy="{name}", max_tokens=512)
    chunker = ExampleChunker(config)
    
    text = "Your test text here"
    chunks = await chunker.chunk(text)
    
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.start_idx < c.end_idx for c in chunks)


@pytest.mark.asyncio
async def test_example_chunker_edge_cases():
    config = ChunkConfig(strategy="{name}")
    chunker = ExampleChunker(config)
    
    # Empty text
    assert await chunker.chunk("") == []
    
    # Very short text
    short = await chunker.chunk("hi")
    assert len(short) >= 1
"""


# Benchmark template (benchmarks/{name}_bench.py):
"""
import asyncio
import time
from src.infracore.chunking.{name} import ExampleChunker
from src.infracore.chunking.base import ChunkConfig


async def benchmark_throughput(num_iterations: int = 100):
    config = ChunkConfig(strategy="{name}")
    chunker = ExampleChunker(config)
    
    sample_text = "Sample text " * 100  # ~1000 chars
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        await chunker.chunk(sample_text)
    elapsed = time.perf_counter() - start
    
    throughput = num_iterations / elapsed
    print(f"Throughput: {throughput:.1f} chunks/sec")
    
    return {"throughput": throughput, "num_iterations": num_iterations}


if __name__ == "__main__":
    result = asyncio.run(benchmark_throughput())
    print(result)
"""
