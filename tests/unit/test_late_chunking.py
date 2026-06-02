"""Unit tests for LateChunkingEmbedder using mock tokenizer and model."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys

try:
    import torch
except ImportError:
    torch = None

from src.infracore.embedding.late_chunking import LateChunkingConfig, LateChunkingEmbedder


def test_late_chunking_config():
    """Test LateChunkingConfig validation and default values."""
    config = LateChunkingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        max_doc_tokens=1024,
        pooling_strategy="mean",
        batch_size=4
    )
    assert config.model_name == "BAAI/bge-m3"
    assert config.device == "cpu"
    assert config.max_doc_tokens == 1024
    assert config.pooling_strategy == "mean"
    assert config.batch_size == 4


@pytest.mark.asyncio
@patch("src.infracore.embedding.late_chunking.AutoTokenizer")
@patch("src.infracore.embedding.late_chunking.AutoModel")
async def test_late_chunking_embed_basic(mock_auto_model, mock_auto_tokenizer):
    """Test LateChunkingEmbedder basic document chunk embedding flow."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    # Tokenizer output mock
    # doc_text = "Hello world! This is a test."
    # Chunks = ["Hello world!", "This is a test."]
    # Let's say we have 7 tokens (CLS, Hello, world!, This, is, a, test., SEP)
    # Offsets: CLS (0,0), Hello (0,5), world! (6,12), This (13,17), is (18,20), a (21,22), test. (23,28), SEP (0,0)
    mock_input_ids = MagicMock()
    mock_attention_mask = MagicMock()
    mock_offset_mapping = MagicMock()
    
    # Simulating tokenizer dictionary response
    mock_tokenizer.return_value = {
        "input_ids": mock_input_ids,
        "attention_mask": mock_attention_mask,
        "offset_mapping": mock_offset_mapping
    }
    
    # offset_mapping returned as tensor or list that we convert
    mock_offset_mapping.__getitem__.return_value.tolist.return_value = [
        [0, 0],   # CLS
        [0, 12],  # "Hello world!"
        [13, 28], # "This is a test."
        [0, 0]    # SEP
    ]
    
    # Mock model
    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_model.config.hidden_size = 128
    
    # Mock model forward output last_hidden_state
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs
    
    # seq_len = 4, hidden_size = 128
    mock_hidden_state = MagicMock()
    mock_outputs.last_hidden_state = mock_hidden_state
    
    # mock_hidden_state[0].cpu().numpy() returns array of shape (4, 128)
    dummy_hidden_states = np.random.randn(4, 128).astype(np.float32)
    mock_hidden_state.__getitem__.return_value.cpu.return_value.numpy.return_value = dummy_hidden_states

    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cpu")
    embedder = LateChunkingEmbedder(config)
    
    chunks = ["Hello world!", "This is a test."]
    doc_text = "Hello world! This is a test."
    
    embeddings = await embedder.embed(chunks, doc_text)
    
    # Check shape: (len(chunks), hidden_size) = (2, 128)
    assert embeddings.shape == (2, 128)
    assert embeddings.dtype == np.float32
    
    # Check L2 normalization: norms should be close to 1.0
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5


@pytest.mark.asyncio
@patch("src.infracore.embedding.late_chunking.AutoTokenizer")
@patch("src.infracore.embedding.late_chunking.AutoModel")
async def test_late_chunking_embed_independent(mock_auto_model, mock_auto_tokenizer):
    """Test LateChunkingEmbedder fallback to independent embedding when no doc_text is provided."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    }
    
    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_model.config.hidden_size = 64
    
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs
    
    # 3 tokens, 64 hidden size
    dummy_hidden_states = torch.randn(3, 64)
    mock_outputs.last_hidden_state.__getitem__.return_value = dummy_hidden_states
    
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cpu")
    embedder = LateChunkingEmbedder(config)
    
    chunks = ["chunk one", "chunk two"]
    embeddings = await embedder.embed(chunks, doc_text="")
    
    assert embeddings.shape == (2, 64)
    # Norms should be 1.0
    for emb in embeddings:
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


@pytest.mark.asyncio
@patch("src.infracore.embedding.late_chunking.AutoTokenizer")
@patch("src.infracore.embedding.late_chunking.AutoModel")
async def test_late_chunking_embed_single(mock_auto_model, mock_auto_tokenizer):
    """Test LateChunkingEmbedder embed_single returns 1D vector."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    mock_tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    }
    
    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_model.config.hidden_size = 256
    
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs
    
    dummy_hidden_states = torch.randn(5, 256)
    mock_outputs.last_hidden_state.__getitem__.return_value = dummy_hidden_states
    
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cpu")
    embedder = LateChunkingEmbedder(config)
    
    embedding = await embedder.embed_single("query text")
    
    assert embedding.shape == (256,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_late_chunking_empty_input():
    """Test LateChunkingEmbedder handles empty chunks list."""
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cpu")
    embedder = LateChunkingEmbedder(config)
    embedder.embedding_dim = 128
    
    embeddings = await embedder.embed([], doc_text="Some document text")
    assert embeddings.shape == (0, 128)


def test_late_chunking_device_detection():
    """Test default device detection mapping logic."""
    # When device is specified explicitly
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cuda")
    embedder = LateChunkingEmbedder(config)
    assert embedder.device == "cuda"

    # When device is auto
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="auto")
    with patch("src.infracore.embedding.late_chunking.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        embedder = LateChunkingEmbedder(config)
        assert embedder.device == "cuda"

    with patch("src.infracore.embedding.late_chunking.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        embedder = LateChunkingEmbedder(config)
        assert embedder.device == "mps"

    with patch("src.infracore.embedding.late_chunking.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        embedder = LateChunkingEmbedder(config)
        assert embedder.device == "cpu"


@pytest.mark.asyncio
async def test_late_chunking_missing_libraries():
    """Test LateChunkingEmbedder raises ImportError when torch/transformers is missing."""
    config = LateChunkingConfig(model_name="BAAI/bge-m3", device="cpu")
    embedder = LateChunkingEmbedder(config)
    
    # Simulate missing dependencies
    with patch("src.infracore.embedding.late_chunking.torch", None):
        with pytest.raises(ImportError) as exc_info:
            await embedder.embed(["test"], "doc")
        assert "transformers libraries are required" in str(exc_info.value)
