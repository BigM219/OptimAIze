from __future__ import annotations

from fastapi import APIRouter

from optimaize_parent_api.api.v1.schemas import LaunchOCRUIRequest, LaunchResultResponse, ModulesResponse, ModuleStatusResponse
from optimaize_parent_api.services.modules.module_registry import list_modules
from optimaize_parent_api.services.modules.ocr_module_service import launch_legacy_ui, ocr_status

router = APIRouter(prefix="/modules", tags=["modules"])


@router.get("", response_model=ModulesResponse)
def modules() -> ModulesResponse:
    return ModulesResponse(modules=[ModuleStatusResponse(**module) for module in list_modules()])


@router.get("/ocr/status", response_model=ModuleStatusResponse)
def ocr_module_status() -> ModuleStatusResponse:
    return ModuleStatusResponse(**ocr_status())


@router.post("/ocr/launch-ui", response_model=LaunchResultResponse)
def launch_ocr_ui(request: LaunchOCRUIRequest) -> LaunchResultResponse:
    result = launch_legacy_ui(request.server_port, request.share, request.inbrowser)
    return LaunchResultResponse(**result)
