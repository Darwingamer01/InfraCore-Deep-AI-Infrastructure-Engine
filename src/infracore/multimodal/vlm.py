from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import logging

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


class VLMDocumentQA:
    """Lightweight document QA that combines OCR + retrieved docs.

    This first iteration is rule-based and deterministic: it searches OCR
    and retrieved document text for keywords from the question and returns
    the best matching sentence along with source metadata and a confidence
    score. The interface is intentionally simple so it can later be swapped
    to a learned VLM (BLIP/LLaVA) without changing callers.
    """

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def _sentences(self, text: str) -> List[str]:
        # very small sentence splitter
        import re

        s = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return s

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

    def _score_and_pick(self, question: str, contexts: List[Dict[str, Any]]) -> AnswerResult:
        q_words = [w.lower() for w in question.split() if len(w) > 2]
        best = None
        best_score = 0.0
        best_source_meta = None
        best_source_idx = None

        for ctx_idx, ctx in enumerate(contexts):
            text = ctx.get("text", "")
            for sent in self._sentences(text):
                s_low = sent.lower()
                score = sum(1 for w in q_words if w in s_low)
                if score > best_score:
                    best_score = float(score)
                    best = sent
                    best_source_meta = ctx
                    best_source_idx = ctx_idx

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

    async def answer(self, question: str, ocr_results: Optional[List[OCRResult]] = None, retrieved: Optional[List[Dict[str, Any]]] = None) -> AnswerResult:
        """Produce an answer given a question, OCR results, and retrieved docs.

        - `ocr_results`: list of `OCRResult` objects
        - `retrieved`: list of dicts with keys `id`, `text`, and optional `meta`
        """
        ocr_results = ocr_results or []
        retrieved = retrieved or []

        logger.info("VLMDocumentQA.answer called — question=%s, ocr_items=%d, retrieved_items=%d", question, len(ocr_results), len(retrieved))

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

        answer = self._score_and_pick(question, contexts)
        logger.info("VLMDocumentQA produced answer (confidence=%.2f, sources=%d): %s", answer.confidence, len(answer.sources), answer.text[:120])
        return answer
