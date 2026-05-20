import asyncio
import io
import numpy as np
import pytest
from PIL import Image

from infracore.multimodal import get_image_embedder, OCRPipeline, ImageTextRetriever, VLMDocumentQA


def _small_image_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color=(123, 222, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clip_embedder_and_retriever_smoke():
    # Skip the smoke test unless a local CLIP model is already cached to
    # avoid long downloads during CI or local quick runs.
    import os

    model_cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32"
    )
    # Only run the heavy CLIP smoke test when explicitly enabled and cached
    if os.environ.get("RUN_CLIP_SMOKE", "0") != "1" or not os.path.exists(model_cache_dir):
        pytest.skip("Skipping CLIP smoke test (set RUN_CLIP_SMOKE=1 and pre-cache model to run)")

    # Try to instantiate a real CLIP embedder; skip test if libs unavailable
    try:
        emb = get_image_embedder(prefer="clip")
    except Exception:
        pytest.skip("CLIP model or dependencies not available")

    retriever = ImageTextRetriever(embedder=emb)
    img = _small_image_bytes()

    async def run():
        # index two documents
        await retriever.index("doc1", img, text="a cat on a mat")
        await retriever.index("doc2", img, text="a dog in a yard")

        # image embedding
        img_emb = await emb.embed_images([img])
        assert isinstance(img_emb, np.ndarray)
        assert img_emb.shape[0] == 1
        # normalized
        assert np.allclose(np.linalg.norm(img_emb, axis=1), 1.0, atol=1e-3)

        # text embedding
        txt_emb = await emb.embed_texts(["a cat on a mat"])
        assert isinstance(txt_emb, np.ndarray)
        assert txt_emb.shape[0] == 1
        assert np.allclose(np.linalg.norm(txt_emb, axis=1), 1.0, atol=1e-3)

        # deterministic-ish: repeat embedding and compare
        txt_emb2 = await emb.embed_texts(["a cat on a mat"])
        assert np.allclose(txt_emb, txt_emb2, atol=1e-6)

        results = await retriever.search_by_text("cat")
        assert isinstance(results, list)
        assert len(results) > 0

    asyncio.get_event_loop().run_until_complete(run())


def test_clip_configurable_lazy_init():
    # Verify the CLIP embedder accepts a configurable model id and does not
    # load the model at construction time (lazy load).
    from infracore.multimodal.clip_embedder import CLIPEmbedder, get_image_embedder

    emb = CLIPEmbedder(model_name="my/custom-model")
    assert emb.model_name == "my/custom-model"
    assert getattr(emb, "_model") is None

    # env var override via factory
    import os

    os.environ["CLIP_MODEL_ID"] = "env/specified-model"
    try:
        emb2 = get_image_embedder(prefer="clip")
        assert emb2.model_name == "env/specified-model"
    finally:
        del os.environ["CLIP_MODEL_ID"]


def test_vlm_stub_answer():
    vlm = VLMDocumentQA()
    ans = asyncio.get_event_loop().run_until_complete(vlm.answer("what is this?", ocr_results=[], retrieved=[]))
    # Verify AnswerResult structure
    assert hasattr(ans, "text")
    assert hasattr(ans, "sources")
    assert hasattr(ans, "confidence")
    assert isinstance(ans.sources, list)
    # Empty context yields empty answer
    assert ans.text == ""
    assert len(ans.sources) == 0


def test_ocr_integration_smoke():
    # Run a lightweight OCR smoke test using a synthetic image. Skip if
    # pytesseract or the tesseract binary are not available.
    try:
        import pytesseract  # noqa: F401
        import shutil
        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found")
    except Exception:
        pytest.skip("pytesseract or tesseract not available; skipping OCR smoke test")

    from infracore.multimodal.ocr import OCRPipeline
    from PIL import Image, ImageDraw, ImageFont
    import io

    # create a small image with text
    img = Image.new("RGB", (200, 60), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((10, 10), "Hello 123", fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    ocr = OCRPipeline()

    async def run():
        res = await ocr.ocr([img_bytes])
        assert isinstance(res, list)
        assert len(res) == 1
        r = res[0]
        assert r.success is True
        # Expect at least one word recognized
        assert r.words_count >= 1

    asyncio.get_event_loop().run_until_complete(run())


def test_vlm_document_qa_basic():
    from infracore.multimodal.vlm import VLMDocumentQA, Source
    from infracore.multimodal.ocr import OCRResult

    qa = VLMDocumentQA()

    # OCR results with some text
    ocr = [OCRResult(text="This is a scanned invoice for Acme Corp. Total due: $123.45", n_lines=1, words_count=8, success=True)]

    # Retrieved documents
    retrieved = [{"id": "doc1", "text": "The invoice from Acme Corp shows total $123.45 due on 2026-05-20."}]

    ans = asyncio.get_event_loop().run_until_complete(qa.answer("What is the total due?", ocr_results=ocr, retrieved=retrieved))
    assert ans.confidence > 0
    assert isinstance(ans.text, str)
    assert len(ans.sources) >= 1
    
    # Verify source provenance structure
    src = ans.sources[0]
    assert isinstance(src, Source)
    assert src.source_type in ("ocr", "retrieved")
    assert src.source_id  # should have an id
    assert src.snippet  # should have supporting snippet text
    assert src.confidence is None or isinstance(src.confidence, float)


def test_vlm_empty_context():
    from infracore.multimodal.vlm import VLMDocumentQA

    qa = VLMDocumentQA()
    ans = asyncio.get_event_loop().run_until_complete(qa.answer("Who wrote this?", ocr_results=[], retrieved=[]))
    assert ans.text == ""
    assert ans.confidence == 0.0


def test_vlm_ocr_provenance():
    """Verify OCR metadata (page, coordinates, confidence) is propagated to sources."""
    from infracore.multimodal.vlm import VLMDocumentQA, Source
    from infracore.multimodal.ocr import OCRResult

    qa = VLMDocumentQA()

    # OCR result with synthetic raw data (simulating pytesseract output)
    ocr_raw = {
        "left": [10, 30, 60],
        "top": [10, 10, 10],
        "width": [15, 25, 20],
        "height": [20, 20, 20],
        "conf": [95, 90, 88],  # confidence scores per word
        "text": ["Total", "amount", "paid"],
    }
    ocr = [
        OCRResult(
            text="Total amount paid $500.00",
            n_lines=1,
            words_count=4,
            raw=ocr_raw,
            success=True,
        )
    ]

    retrieved = []
    ans = asyncio.get_event_loop().run_until_complete(qa.answer("How much was paid?", ocr_results=ocr, retrieved=retrieved))
    
    assert ans.confidence > 0
    assert len(ans.sources) >= 1
    
    src = ans.sources[0]
    assert isinstance(src, Source)
    assert src.source_type == "ocr"
    assert src.source_id.startswith("ocr:")
    # OCR metadata should be extracted
    assert src.coordinates is not None
    assert isinstance(src.coordinates, list)
    assert len(src.coordinates) > 0
    # Verify coordinate structure
    coord = src.coordinates[0]
    assert "left" in coord
    assert "top" in coord
    assert "confidence" in coord
    # Confidence should be averaged from OCR word confidences
    assert src.confidence is not None and src.confidence > 0


def test_vlm_retrieved_provenance():
    """Verify retrieved document metadata is propagated to sources."""
    from infracore.multimodal.vlm import VLMDocumentQA, Source
    from infracore.multimodal.ocr import OCRResult

    qa = VLMDocumentQA()

    ocr = []
    retrieved = [
        {
            "id": "doc1",
            "text": "The invoice amount is $500.00 due immediately.",
            "page": 1,
            "confidence": 0.95,
        }
    ]

    ans = asyncio.get_event_loop().run_until_complete(qa.answer("What is the invoice amount?", ocr_results=ocr, retrieved=retrieved))

    assert ans.confidence > 0
    assert len(ans.sources) >= 1

    src = ans.sources[0]
    assert isinstance(src, Source)
    assert src.source_type == "retrieved"
    assert src.source_id == "doc1"
    assert src.page == 1
    assert src.confidence == 0.95


def test_vlm_mixed_sources():
    """Verify provenance when mixing OCR and retrieved documents."""
    from infracore.multimodal.vlm import VLMDocumentQA, Source
    from infracore.multimodal.ocr import OCRResult

    qa = VLMDocumentQA()

    ocr = [
        OCRResult(
            text="Invoice from Company A",
            n_lines=1,
            words_count=3,
            success=True,
        )
    ]

    retrieved = [
        {
            "id": "ref1",
            "text": "Company A invoice total $1000.00",
            "page": 1,
        }
    ]

    ans = asyncio.get_event_loop().run_until_complete(qa.answer("What is the invoice total?", ocr_results=ocr, retrieved=retrieved))

    assert len(ans.sources) >= 1
    src = ans.sources[0]
    assert isinstance(src, Source)
    # The answer should come from one of the sources
    assert src.source_type in ("ocr", "retrieved")
