import html
import json
from typing import Any

from PIL import Image

from .base import BaseVLMBackend


class SuryaPackageBackend(BaseVLMBackend):
    def __init__(self, **_: Any):
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ImportError(
                "surya-package backend requires `pip install surya-ocr` and a configured Surya inference backend"
            ) from exc
        self.manager = SuryaInferenceManager()
        self.recognition_predictor = RecognitionPredictor(self.manager)

    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        predictions = self.recognition_predictor([image])
        return self._prediction_to_text(predictions[0] if predictions else None, category)

    def generate_ocr_pages(self, images: list[Image.Image], category: str = "plain") -> list[str]:
        rgb_images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
        predictions = self.recognition_predictor(rgb_images)
        return [self._prediction_to_text(prediction, category) for prediction in predictions]

    def _prediction_to_text(self, prediction: Any, category: str) -> str:
        if prediction is None:
            return ""
        if isinstance(prediction, dict):
            blocks = prediction.get("blocks")
        else:
            blocks = getattr(prediction, "blocks", None)
        if not blocks:
            return str(prediction)
        html_parts = []
        text_parts = []
        for block in blocks:
            if isinstance(block, dict):
                block_html = block.get("html") or ""
            else:
                block_html = getattr(block, "html", "") or ""
            if block_html:
                html_parts.append(str(block_html))
                text_parts.append(html.unescape(str(block_html)))
        if category.strip().lower() in {"html", "table"}:
            return "\n".join(html_parts)
        return "\n".join(text_parts)

    def docqa_pages(
        self,
        images: list[Image.Image],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        page_texts = self.generate_ocr_pages(images, category="plain")
        return json.dumps(
            {
                "answer": "\n\n".join(f"Page {i + 1}: {text}" for i, text in enumerate(page_texts)),
                "evidence_pages": list(range(1, len(page_texts) + 1)),
            },
            ensure_ascii=False,
        )
