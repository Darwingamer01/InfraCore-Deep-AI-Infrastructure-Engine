from __future__ import annotations
from typing import List, Sequence


class OCRPipeline:
    """Simple OCR pipeline wrapper.

    This class attempts to use `pytesseract` if available; otherwise it falls
    back to a deterministic no-op that returns empty text for images. The
    interface is intentionally small for easy replacement with production OCR.
    """

    def __init__(self):
        try:
            import pytesseract  # type: ignore

            self._pytesseract = pytesseract
        except Exception:
            self._pytesseract = None

    async def ocr(self, images: Sequence[bytes]) -> List[str]:
        results = []
        if self._pytesseract is None:
            # graceful fallback for tests
            return ["" for _ in images]

        from PIL import Image
        import io

        for b in images:
            img = Image.open(io.BytesIO(b))
            text = self._pytesseract.image_to_string(img)
            results.append(text or "")
        return results
