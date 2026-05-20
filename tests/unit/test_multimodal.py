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
    ans = asyncio.get_event_loop().run_until_complete(vlm.answer(b"img", "what is this?"))
    assert "VLM-ANSWER-STUB" in ans
