from __future__ import annotations
from typing import List, Sequence
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseImageEmbedder:
    """Minimal interface for image+text embedders."""

    async def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        raise NotImplementedError()

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError()


class CLIPEmbedder(BaseImageEmbedder):
    """CLIP-based embedder with lazy model loading and device-safe behavior.

    Uses `transformers`' `CLIPModel` + `CLIPProcessor` to produce image and
    text embeddings. Model loads on first call to either embedding method.
    """

    def __init__(self, model_name: str | None = None) -> None:
        # Resolve model id from constructor param or environment variable.
        # Default to a smaller development-friendly model to avoid large
        # downloads in casual runs; allow production override to openai/clip-vit-base-patch32.
        env_model = os.environ.get("CLIP_MODEL_ID")
        default_model = "laion/CLIP-ViT-B-32"
        self.model_name = model_name or env_model or default_model
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - environment specific
            logger.exception("transformers or torch not available: falling back")
            raise

        # choose device: cuda > mps > cpu
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger.info("Loading CLIP model %s on device %s", self.model_name, device)
        model = CLIPModel.from_pretrained(self.model_name)
        processor = CLIPProcessor.from_pretrained(self.model_name)

        model.to(device)
        model.eval()

        self._model = model
        self._processor = processor
        self._device = device

    async def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        """Embed images (bytes) and return L2-normalized numpy vectors.

        Returns array shape (N, D) as float32.
        """
        self._ensure_loaded()
        # Lazy import to keep module-level imports light
        import torch
        import io
        from PIL import Image

        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        inputs = self._processor(images=imgs, return_tensors="pt")
        # move tensors to model device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)

        # transformers may return either a Tensor or a ModelOutput wrapper.
        tensor = None
        if hasattr(out, "cpu"):
            tensor = out
        else:
            # try common fields
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                tensor = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                # pool across sequence dim as a fallback
                lh = out.last_hidden_state
                if hasattr(lh, "mean"):
                    tensor = lh.mean(dim=1)
                else:
                    tensor = lh
        if tensor is None:
            raise RuntimeError("Unable to obtain tensor from CLIP image model output")

        arr = tensor.cpu().numpy().astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-12)
        return arr

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts and return L2-normalized numpy vectors."""
        self._ensure_loaded()
        import torch

        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)

        tensor = None
        if hasattr(out, "cpu"):
            tensor = out
        else:
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                tensor = out.pooler_output
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                lh = out.last_hidden_state
                if hasattr(lh, "mean"):
                    tensor = lh.mean(dim=1)
                else:
                    tensor = lh
        if tensor is None:
            raise RuntimeError("Unable to obtain tensor from CLIP text model output")

        arr = tensor.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-12)
        return arr


class DummyImageEmbedder(BaseImageEmbedder):
    """Backward-compatible dummy embedder returning deterministic vectors.

    This is used as a fallback if `transformers` or `torch` are not available
    or the environment requests no external downloads.
    """

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    async def embed_images(self, images: Sequence[bytes]) -> np.ndarray:
        import hashlib

        vecs = []
        for b in images:
            h = hashlib.blake2b(b, digest_size=32).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            out = np.repeat(arr, int(np.ceil(self.dim / arr.size)))[: self.dim]
            out = out / (np.linalg.norm(out) + 1e-12)
            vecs.append(out)
        return np.stack(vecs, axis=0)

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        # text -> bytes -> same deterministic process
        return await self.embed_images([t.encode("utf-8") for t in texts])


def get_image_embedder(prefer: str = "clip", model_name: str = "openai/clip-vit-base-patch32") -> BaseImageEmbedder:
    """Factory returning a CLIP embedder or a Dummy fallback.

    `prefer` may be extended in the future.
    """
    if prefer == "clip":
        # Allow env var override of the model id
        model_name = os.environ.get("CLIP_MODEL_ID", model_name)
        try:
            emb = CLIPEmbedder(model_name=model_name)
            logger.info("Using CLIP embedder with model: %s", emb.model_name)
            return emb
        except Exception:
            logger.exception("Failed to create CLIPEmbedder; falling back to DummyImageEmbedder")
            return DummyImageEmbedder()
    return DummyImageEmbedder()
