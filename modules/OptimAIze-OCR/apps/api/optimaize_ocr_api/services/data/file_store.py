from __future__ import annotations

import time
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from optimaize_ocr.storage.history_db import safe_name
from optimaize_ocr_api.core.errors import UploadValidationError

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"}


class FileStore:
    def __init__(self, upload_dir: Path, max_upload_bytes: int):
        self.upload_dir = upload_dir
        self.max_upload_bytes = max_upload_bytes

    async def save_upload(self, upload: UploadFile, prefix: str = "single") -> tuple[Path, Image.Image]:
        if upload.content_type not in ALLOWED_IMAGE_TYPES:
            raise UploadValidationError(f"Unsupported image type: {upload.content_type}")
        data = await upload.read()
        if not data:
            raise UploadValidationError("Uploaded image is empty.")
        if len(data) > self.max_upload_bytes:
            raise UploadValidationError("Uploaded image is too large.")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        original = safe_name(Path(upload.filename or "upload").stem)
        path = self.upload_dir / f"{prefix}_{int(time.time() * 1000)}_{original}.png"
        try:
            image = Image.open(__import__("io").BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise UploadValidationError("Uploaded file is not a valid image.") from exc
        image.save(path)
        return path, image
