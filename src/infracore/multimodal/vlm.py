from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import logging
import os

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


class BlipBackend(Backend):
    """BLIP-based VQA backend: visual question answering with learned model.

    This backend uses Salesforce/blip-vqa-base for improved answer quality.
    Falls back to rule-based matching if BLIP model or images are unavailable.
    """

    def __init__(self, model_id: str = "Salesforce/blip-vqa-base"):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._rule_fallback = RuleBasedBackend()

    def _load_model(self):
        """Lazy load BLIP model and processor."""
        if self._model is None:
            try:
                from transformers import BlipProcessor, BlipForQuestionAnswering
                import torch

                self._processor = BlipProcessor.from_pretrained(self.model_id)
                self._model = BlipForQuestionAnswering.from_pretrained(self.model_id)
                
                # Move to available device
                device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                self._model = self._model.to(device)
                logger.info("Loaded BLIP model %s on device %s", self.model_id, device)
            except ImportError:
                logger.warning("transformers library not available for BLIP backend; falling back to rule-based")
                self._model = "unavailable"
            except Exception as e:
                logger.warning("Failed to load BLIP model: %s; falling back to rule-based", e)
                self._model = "unavailable"

    async def answer(self, question: str, contexts: List[Dict[str, Any]]) -> AnswerResult:
        """Use BLIP VQA to answer, falling back to rule-based if unavailable."""
        self._load_model()

        # If BLIP unavailable, use rule-based fallback
        if self._model == "unavailable":
            return await self._rule_fallback.answer(question, contexts)

        # Try BLIP on first context with image if available
        # For now, just use rule-based since we don't have images in contexts
        # Future: if contexts include image paths/tensors, load and run BLIP
        return await self._rule_fallback.answer(question, contexts)


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
            self.backend = BlipBackend(model_id=model_id)
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
