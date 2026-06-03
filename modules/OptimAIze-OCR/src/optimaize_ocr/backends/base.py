# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Base class for VLM backends.

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

class BaseVLMBackend(ABC):
    """Abstract interface for all Vision-Language Model OCR backends."""

    @abstractmethod
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        """Run OCR on a PIL image using a specific prompt category.

        Args:
            image: A crop or a full PIL image.
            category: Prompt category (e.g. 'plain', 'table', 'formula', 'caption').

        Returns:
            The recognized/transcribed text output.
        """
        pass

    def supports_multi_image_single_call(self) -> bool:
        return False

    def generate_ocr_pages(self, images: list[Image.Image], category: str = "plain") -> list[str]:
        return [self.generate_ocr(image, category=category) for image in images]

    def docqa_pages(
        self,
        images: list[Image.Image],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support multi-page DocQA")

    def supports_visual_token_cache(self) -> bool:
        return False

    def build_visual_cache(self, image: Image.Image, category: str = "plain") -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support visual token cache")

    def generate_ocr_from_visual_cache(self, cache: dict[str, Any], category: str = "plain") -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support visual token cache")
