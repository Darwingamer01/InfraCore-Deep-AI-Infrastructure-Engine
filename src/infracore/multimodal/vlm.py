from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, List, Optional, Dict
import logging
import os

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from transformers import BlipForQuestionAnswering, BlipProcessor
except Exception:  # pragma: no cover - optional dependency
    BlipProcessor = None
    BlipForQuestionAnswering = None

from .ocr import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source metadata for an answer's supporting evidence."""
    source_type: str  # "ocr" or "retrieved"
    source_id: str  # doc id, ocr index, or filename
    snippet: str  # text excerpt supporting the answer
    page: Optional[int] = None  # page number if multi-page doc
    bounding_box: Optional[Dict[str, Any]] = None  # bbox coords from OCR if available
    coordinates: Optional[List[Dict[str, float]]] = None  # word-level coordinates from OCR raw
    confidence: Optional[float] = None  # OCR/retrieval confidence score if available


@dataclass
class AnswerResult:
    text: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0


class Backend(ABC):
    """Abstract backend for VLM document QA."""

    @abstractmethod
    async def answer(self, question: str, contexts: List[Dict[str, Any]]) -> AnswerResult:
        """Produce an answer given a question and contexts."""
        pass


class RuleBasedBackend(Backend):
    """Rule-based backend: deterministic keyword matching across contexts."""

    def _sentences(self, text: str) -> List[str]:
        # very small sentence splitter
        import re

        s = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return s

    async def answer(self, question: str, contexts: List[Dict[str, Any]]) -> AnswerResult:
        """Score contexts by keyword overlap and pick best sentence."""
        q_words = [w.lower() for w in question.split() if len(w) > 2]
        best = None
        best_score = 0.0
        best_source_meta = None

        for ctx in contexts:
            text = ctx.get("text", "")
            for sent in self._sentences(text):
                s_low = sent.lower()
                score = sum(1 for w in q_words if w in s_low)
                if score > best_score:
                    best_score = float(score)
                    best = sent
                    best_source_meta = ctx

        # Build sources list based on matched context
        sources: List[Source] = []
        
        if best is not None and best_source_meta is not None:
            source = Source(
                source_type=best_source_meta.get("source_type", "retrieved"),
                source_id=best_source_meta.get("source_id", ""),
                snippet=best,
                page=best_source_meta.get("page"),
                bounding_box=best_source_meta.get("bounding_box"),
                coordinates=best_source_meta.get("coordinates"),
                confidence=best_source_meta.get("confidence"),
            )
            sources.append(source)
            denom = max(1, len(q_words))
            confidence = min(1.0, best_score / denom)
            return AnswerResult(text=best, sources=sources, confidence=confidence)

        # fallback: return tiny summary of first available context
        if contexts:
            first = contexts[0]
            snippet = (first.get("text", "")[:200]).strip()
            source = Source(
                source_type=first.get("source_type", "retrieved"),
                source_id=first.get("source_id", ""),
                snippet=snippet,
                page=first.get("page"),
                bounding_box=first.get("bounding_box"),
                coordinates=first.get("coordinates"),
                confidence=first.get("confidence", 0.1),
            )
            sources.append(source)
            return AnswerResult(text=snippet or "", sources=sources, confidence=0.1)

        return AnswerResult(text="", sources=[], confidence=0.0)


class BlipDocumentQABackend(Backend):
    """BLIP-based VQA backend: visual question answering with learned model.

    This backend uses Salesforce/blip-vqa-base for improved answer quality.
    Falls back to rule-based matching if the model, image payload, or runtime
    dependencies are unavailable.
    """

    def __init__(self, model_id: str = "Salesforce/blip-vqa-base"):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._rule_fallback = RuleBasedBackend()
        self._device = "cpu"

    def _load_model(self):
        """Lazy load BLIP model and processor."""
        if self._model is not None or self._processor is not None:
            return
        if BlipProcessor is None or BlipForQuestionAnswering is None or torch is None:
            logger.warning("transformers/torch not available for BLIP backend; falling back to rule-based")
            self._model = "unavailable"
            return

        try:
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForQuestionAnswering.from_pretrained(self.model_id)

            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            self._model = self._model.to(self._device)
            logger.info("Loaded BLIP model %s on device %s", self.model_id, self._device)
        except Exception as e:
            logger.warning("Failed to load BLIP model: %s; falling back to rule-based", e)
            self._model = "unavailable"

    def _extract_image(self, context: Dict[str, Any]) -> Any | None:
        image = context.get("image") or context.get("image_bytes") or context.get("pil_image")
        if image is None:
            return None
        if Image is None:
            return None
        if hasattr(image, "convert"):
            return image.convert("RGB")
        if isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return None

    async def _answer_with_blip(self, question: str, context: Dict[str, Any]) -> str:
        image = self._extract_image(context)
        if image is None or self._model == "unavailable" or self._processor is None:
            return ""

        inputs = self._processor(images=image, text=question, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        elif isinstance(inputs, dict):
            inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}

        generated_ids = self._model.generate(**inputs)
        if hasattr(self._processor, "tokenizer") and self._processor.tokenizer is not None:
            return self._processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if hasattr(self._processor, "decode"):
            return self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return ""

    async def answer(self, question: str, contexts: List[Dict[str, Any]]) -> AnswerResult:
        """Use BLIP VQA to answer, falling back to rule-based if unavailable."""
        self._load_model()

        fallback = await self._rule_fallback.answer(question, contexts)
        if self._model == "unavailable" or self._processor is None:
            return fallback

        for context in contexts:
            blip_text = await self._answer_with_blip(question, context)
            if blip_text:
                return AnswerResult(text=blip_text, sources=fallback.sources, confidence=fallback.confidence)

        return fallback


BlipBackend = BlipDocumentQABackend


class VLMDocumentQA:
    """Document QA facade that delegates to pluggable backends.

    Supports multiple backends for answer generation:
    - "rule": deterministic keyword-based matching (default, fast, reproducible)
    - "blip": Salesforce BLIP VQA model (learned, better quality, requires transformers)

    The interface is intentionally simple and unchanged from prior iteration,
    allowing backend swaps without modifying callers. All backends return
    structured AnswerResult with full source provenance.

    Environment variables:
    - VLM_BACKEND: "rule" or "blip" (default: "rule")
    - BLIP_MODEL_ID: custom BLIP model ID (default: "Salesforce/blip-vqa-base")
    """

    def __init__(self, backend: str | None = None, model: Any | None = None) -> None:
        """Initialize VLMDocumentQA with a backend strategy.

        Args:
            backend: "rule" or "blip". Defaults to env VLM_BACKEND or "rule".
            model: Optional pre-loaded model (for testing/custom models).
        """
        if backend is None:
            backend = os.environ.get("VLM_BACKEND", "rule")

        self.backend_name = backend
        self.model = model

        if backend == "blip":
            model_id = os.environ.get("BLIP_MODEL_ID", "Salesforce/blip-vqa-base")
            self.backend = BlipDocumentQABackend(model_id=model_id)
        else:  # default to rule
            self.backend = RuleBasedBackend()

        logger.info("Initialized VLMDocumentQA with backend=%s", self.backend_name)

    def _extract_ocr_metadata(self, ocr_result: OCRResult, ocr_index: int) -> Dict[str, Any]:
        """Extract provenance metadata from OCRResult.raw."""
        meta: Dict[str, Any] = {
            "source_type": "ocr",
            "source_id": f"ocr:{ocr_index}",
        }
        
        # If raw pytesseract data is available, extract word-level coordinates
        if ocr_result.raw and isinstance(ocr_result.raw, dict):
            # pytesseract.Output.DICT returns keys like:
            # level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text
            try:
                coords = []
                confidences = []
                
                # Collect word-level bounding boxes and confidences
                if "left" in ocr_result.raw and "top" in ocr_result.raw:
                    lefts = ocr_result.raw.get("left", [])
                    tops = ocr_result.raw.get("top", [])
                    widths = ocr_result.raw.get("width", [])
                    heights = ocr_result.raw.get("height", [])
                    confs = ocr_result.raw.get("conf", [])
                    
                    for i, (l, t, w, h, c) in enumerate(zip(lefts, tops, widths, heights, confs)):
                        if c > 0:  # only include detected words (conf > 0)
                            coords.append({
                                "left": l,
                                "top": t,
                                "width": w,
                                "height": h,
                                "confidence": float(c),
                            })
                            confidences.append(float(c))
                
                if coords:
                    meta["coordinates"] = coords
                    # Average OCR confidence across all words
                    if confidences:
                        meta["confidence"] = sum(confidences) / len(confidences)
            except Exception as e:
                logger.debug("Failed to extract OCR metadata from raw: %s", e)
        
        return meta

    async def answer(self, question: str, ocr_results: Optional[List[OCRResult]] = None, retrieved: Optional[List[Dict[str, Any]]] = None) -> AnswerResult:
        """Produce an answer given a question, OCR results, and retrieved docs.

        - `question`: question string
        - `ocr_results`: list of `OCRResult` objects
        - `retrieved`: list of dicts with keys `id`, `text`, and optional `meta`

        Returns AnswerResult with text, sources (with full provenance), and confidence.
        """
        ocr_results = ocr_results or []
        retrieved = retrieved or []

        logger.info("VLMDocumentQA.answer called — backend=%s, question=%s, ocr_items=%d, retrieved_items=%d", 
                    self.backend_name, question, len(ocr_results), len(retrieved))

        contexts: List[Dict[str, Any]] = []
        
        # Build context from OCR results with metadata
        for i, o in enumerate(ocr_results):
            ocr_meta = self._extract_ocr_metadata(o, i)
            contexts.append({
                "source_type": ocr_meta["source_type"],
                "source_id": ocr_meta["source_id"],
                "text": o.text,
                "snippet": (o.text or "")[:200],
                "image": getattr(o, "image", None),
                "page": ocr_meta.get("page"),
                "bounding_box": ocr_meta.get("bounding_box"),
                "coordinates": ocr_meta.get("coordinates"),
                "confidence": ocr_meta.get("confidence"),
            })

        # Build context from retrieved documents
        for doc in retrieved:
            text = doc.get("text") or doc.get("snippet") or ""
            contexts.append({
                "source_type": "retrieved",
                "source_id": doc.get("id", ""),
                "text": text,
                "snippet": (text or "")[:200],
                "image": doc.get("image") or doc.get("image_bytes") or doc.get("pil_image"),
                "page": doc.get("page"),
                "bounding_box": doc.get("bounding_box"),
                "coordinates": doc.get("coordinates"),
                "confidence": doc.get("confidence"),
            })

        # Delegate to backend
        answer = await self.backend.answer(question, contexts)
        logger.info("VLMDocumentQA produced answer (backend=%s, confidence=%.2f, sources=%d): %s", 
                    self.backend_name, answer.confidence, len(answer.sources), answer.text[:120])
        return answer
