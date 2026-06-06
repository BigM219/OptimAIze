from __future__ import annotations

# Optional API-key auth. When OPTIMAIZE_API_KEY is unset the dependency is a
# no-op so local/dev use needs no setup; when set, protected routes require a
# matching X-API-Key header.

import hmac
import os

from fastapi import Header

from optimaize_parent_api.core.errors import AuthError

API_KEY_HEADER = "X-API-Key"
_API_KEY_ENV = "OPTIMAIZE_API_KEY"


def _expected_key() -> str | None:
    key = os.getenv(_API_KEY_ENV)
    return key.strip() if key and key.strip() else None


def require_api_key(x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> None:
    """FastAPI dependency enforcing the X-API-Key header when a key is configured.

    No key configured → auth disabled (returns immediately). Uses a constant-time
    compare to avoid leaking the key through timing.
    """
    expected = _expected_key()
    if expected is None:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise AuthError("Missing or invalid API key.")
