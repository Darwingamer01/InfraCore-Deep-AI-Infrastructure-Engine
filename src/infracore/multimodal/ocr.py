from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional
import io
import logging
import shutil

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    n_lines: int
    words_count: int
    raw: Optional[dict] = None
    success: bool = True
    error: Optional[str] = None


class OCRPipeline:
    """OCR pipeline with pytesseract backend and graceful fallback.

    Accepts inputs as bytes, file paths, or PIL.Image objects and returns a
    list of `OCRResult` objects containing extracted text and simple metadata.
    """

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore

            self._pytesseract = pytesseract
        except Exception:
            self._pytesseract = None

    async def ocr(self, images: Sequence[Union[bytes, str, Image.Image]]) -> List[OCRResult]:
        results: List[OCRResult] = []

        # If pytesseract or the tesseract binary is not available, return
        # empty results with a clear failure flag rather than throwing.
        if self._pytesseract is None or shutil.which("tesseract") is None:
            logger.warning("pytesseract or tesseract binary not available; OCR will be a no-op")
            for _ in images:
                results.append(OCRResult(text="", n_lines=0, words_count=0, success=False, error="pytesseract/tesseract unavailable"))
            return results

        for item in images:
            try:
                if isinstance(item, Image.Image):
                    img = item
                elif isinstance(item, bytes):
                    img = Image.open(io.BytesIO(item)).convert("RGB")
                elif isinstance(item, str):
                    img = Image.open(item).convert("RGB")
                else:
                    img = Image.open(io.BytesIO(item)).convert("RGB")

                text = self._pytesseract.image_to_string(img) or ""

                # attempt to get word-level data for metadata
                raw = None
                try:
                    raw = self._pytesseract.image_to_data(img, output_type=self._pytesseract.Output.DICT)
                except Exception:
                    raw = None

                n_lines = len([l for l in text.splitlines() if l.strip()])
                words_count = len(text.split())
                results.append(OCRResult(text=text, n_lines=n_lines, words_count=words_count, raw=raw, success=True))
            except Exception as exc:
                logger.exception("OCR failed for an item")
                results.append(OCRResult(text="", n_lines=0, words_count=0, success=False, error=str(exc)))

        return results
