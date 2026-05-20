"""Integration smoke test for the full multimodal pipeline.

Tests end-to-end flow:
  image -> OCR -> retrieval/context -> VLMDocumentQA -> answer + provenance
"""

import asyncio
import io
import pytest
from PIL import Image, ImageDraw, ImageFont

from infracore.multimodal import OCRPipeline, VLMDocumentQA, AnswerResult, Source


def _create_synthetic_invoice_image() -> bytes:
    """Create a synthetic invoice image with readable text."""
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    # Draw invoice-like text
    text_lines = [
        "INVOICE #2024-001",
        "Company ABC Corp",
        "Total Due: $1,250.00",
        "Due Date: 2026-06-20",
    ]
    
    y_pos = 30
    for line in text_lines:
        draw.text((30, y_pos), line, fill=(0, 0, 0), font=font)
        y_pos += 50
    
    # Serialize to bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_multimodal_pipeline_end_to_end():
    """Test full pipeline: image -> OCR -> VLM QA with provenance."""
    
    # Check OCR dependencies are available
    try:
        import pytesseract  # noqa: F401
        import shutil
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found")
    except Exception:
        pytest.skip("pytesseract or tesseract not available; skipping multimodal integration test")
    
    # Setup components
    ocr_pipeline = OCRPipeline()
    vlm = VLMDocumentQA()
    
    # Step 1: Create synthetic invoice image
    img_bytes = _create_synthetic_invoice_image()
    assert img_bytes, "Failed to create synthetic image"
    
    # Step 2: Run OCR
    ocr_results = await ocr_pipeline.ocr([img_bytes])
    assert len(ocr_results) == 1, "OCR should return 1 result"
    
    ocr_result = ocr_results[0]
    assert ocr_result.success, f"OCR failed: {ocr_result.error}"
    assert ocr_result.text, "OCR should extract text"
    assert "invoice" in ocr_result.text.lower(), "Invoice text not found"
    assert ocr_result.words_count > 0, "Words should be detected"
    
    # Step 3: Feed OCR to VLM QA (without retrieval context for simplicity)
    answer = await vlm.answer(
        question="What is the total due amount?",
        ocr_results=[ocr_result],
        retrieved=[],
    )
    
    # Step 4: Verify answer structure
    assert isinstance(answer, AnswerResult), "Should return AnswerResult"
    assert hasattr(answer, "text"), "AnswerResult should have text"
    assert hasattr(answer, "sources"), "AnswerResult should have sources"
    assert hasattr(answer, "confidence"), "AnswerResult should have confidence"
    
    # Step 5: Verify provenance is present
    assert isinstance(answer.sources, list), "sources should be a list"
    assert len(answer.sources) > 0, "Should have at least one source"
    
    # Step 6: Verify source structure (full provenance)
    source = answer.sources[0]
    assert isinstance(source, Source), "Source should be a Source object"
    assert source.source_type == "ocr", f"Expected source_type='ocr', got {source.source_type}"
    assert source.source_id.startswith("ocr:"), f"source_id should start with 'ocr:', got {source.source_id}"
    assert source.snippet, "Source should have a snippet"
    
    # Step 7: Log the full output for debugging/validation
    print(f"\n--- Multimodal Pipeline Integration Test ---")
    print(f"OCR extracted {ocr_result.words_count} words:")
    print(f"  {ocr_result.text[:200]}")
    print(f"\nVLM Answer (confidence={answer.confidence:.2f}):")
    print(f"  {answer.text}")
    print(f"\nProvenance:")
    print(f"  Source Type: {source.source_type}")
    print(f"  Source ID: {source.source_id}")
    print(f"  Snippet: {source.snippet[:100]}")
    if source.coordinates:
        print(f"  OCR Coordinates: {len(source.coordinates)} word boxes")
    if source.confidence is not None:
        print(f"  OCR Confidence: {source.confidence:.2f}")


@pytest.mark.asyncio
async def test_multimodal_pipeline_with_retrieval():
    """Test pipeline with both OCR and retrieved documents."""
    
    # Check OCR dependencies
    try:
        import pytesseract  # noqa: F401
        import shutil
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found")
    except Exception:
        pytest.skip("pytesseract or tesseract not available; skipping multimodal integration test")
    
    ocr_pipeline = OCRPipeline()
    vlm = VLMDocumentQA()
    
    # Step 1: OCR a synthetic image
    img_bytes = _create_synthetic_invoice_image()
    ocr_results = await ocr_pipeline.ocr([img_bytes])
    assert ocr_results[0].success, "OCR should succeed"
    
    # Step 2: Simulate retrieved documents (would come from retriever in real pipeline)
    retrieved_docs = [
        {
            "id": "doc:policy-2024",
            "text": "Payment must be made within 30 days of invoice. Late payments incur 5% penalty.",
            "page": 1,
            "confidence": 0.98,
        }
    ]
    
    # Step 3: Query with both OCR + retrieved context
    answer = await vlm.answer(
        question="What are the payment terms?",
        ocr_results=ocr_results,
        retrieved=retrieved_docs,
    )
    
    # Step 4: Verify answer has provenance from mixed sources
    assert isinstance(answer, AnswerResult)
    assert len(answer.sources) >= 1, "Should have at least one source"
    
    source = answer.sources[0]
    assert source.source_type in ("ocr", "retrieved"), f"Unexpected source_type: {source.source_type}"
    assert source.source_id, "Source ID should be present"
    assert source.snippet, "Snippet should be present"
    
    # Step 5: Verify we can distinguish which source the answer came from
    if source.source_type == "retrieved":
        assert source.source_id == "doc:policy-2024", "Should reference retrieved doc"
        assert source.page == 1, "Should preserve page metadata from retrieved doc"
        assert source.confidence == 0.98, "Should preserve confidence from retrieved doc"
    
    print(f"\n--- Mixed Provenance Integration Test ---")
    print(f"Answer (confidence={answer.confidence:.2f}):")
    print(f"  {answer.text}")
    print(f"Source: {source.source_type}:{source.source_id}")


@pytest.mark.asyncio
async def test_multimodal_pipeline_graceful_fallback():
    """Test that pipeline handles empty/missing contexts gracefully."""
    
    vlm = VLMDocumentQA()
    
    # Query with no OCR and no retrieved docs
    answer = await vlm.answer(
        question="What is the total?",
        ocr_results=[],
        retrieved=[],
    )
    
    assert isinstance(answer, AnswerResult)
    assert answer.text == "", "Should return empty text for empty context"
    assert answer.confidence == 0.0, "Should return 0 confidence"
    assert len(answer.sources) == 0, "Should have no sources"
    
    print(f"\n--- Graceful Fallback Test ---")
    print(f"Empty context handled correctly: answer='{answer.text}', confidence={answer.confidence}")


@pytest.mark.asyncio
async def test_blip_backend_optional():
    """Optional: Test BLIP backend if transformers library is available.
    
    This test skips gracefully if transformers/BLIP is not installed.
    Requires: pip install transformers torch
    Run with: PYTHONPATH=src pytest tests/integration/test_multimodal_pipeline.py::test_blip_backend_optional -q -vv -s
    """
    
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering  # noqa: F401
    except ImportError:
        pytest.skip("transformers library not available; skipping BLIP backend test")
    
    # Initialize VLMDocumentQA with BLIP backend
    vlm = VLMDocumentQA(backend="blip")
    assert vlm.backend_name == "blip", "Should use BLIP backend"
    
    # Test with simple text-based context (BLIP VQA fallback)
    retrieved_docs = [
        {
            "id": "doc:invoice-001",
            "text": "Invoice Total: $5,000.00. Payment due within 30 days.",
            "page": 1,
            "confidence": 0.95,
        }
    ]
    
    # Query using BLIP backend (currently falls back to rule-based since we don't have images)
    answer = await vlm.answer(
        question="What is the invoice total?",
        ocr_results=[],
        retrieved=retrieved_docs,
    )
    
    # Verify structure
    assert isinstance(answer, AnswerResult), "Should return AnswerResult"
    assert answer.text, "Should produce an answer"
    assert len(answer.sources) > 0, "Should have sources"
    
    source = answer.sources[0]
    assert isinstance(source, Source), "Should be a Source object"
    assert source.source_type == "retrieved", "Source should be from retrieval"
    
    print(f"\n--- BLIP Backend Integration Test ---")
    print(f"Backend: {vlm.backend_name}")
    print(f"Answer (confidence={answer.confidence:.2f}): {answer.text}")
    print(f"Source: {source.source_type}:{source.source_id}")

