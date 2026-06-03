from __future__ import annotations


class OptimAIzeOCRError(Exception):
    status_code = 400


class UploadValidationError(OptimAIzeOCRError):
    status_code = 422


class NotFoundError(OptimAIzeOCRError):
    status_code = 404
