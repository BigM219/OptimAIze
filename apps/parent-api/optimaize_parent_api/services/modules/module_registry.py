from __future__ import annotations

from optimaize_parent_api.services.modules.ocr_module_service import ocr_status
from optimaize_parent_api.services.modules.work_module_service import work_status


def list_modules() -> list[dict[str, object]]:
    return [ocr_status(), work_status()]
