from __future__ import annotations
from typing import Any

class VLMDocumentQA:
    """Very small VLM QA scaffold.

    This class provides a simple interface `answer(image, question)` which
    currently returns a placeholder. In future iterations it will call an
    integrated VLM (e.g., BLIP2, LLaVA) to generate grounded answers.
    """

    def __init__(self, model: Any | None = None):
        self.model = model

    async def answer(self, image: bytes, question: str) -> str:
        # placeholder implementation to keep tests and scaffolding simple
        return "[VLM-ANSWER-STUB] The VLM pipeline is not yet configured."
