from __future__ import annotations

import os

from fastapi import APIRouter, File, Form, UploadFile

from optimaize_ocr_api.api.v1.schemas import ModelListResponse, RuntimeConfigResponse, RuntimeRange, SingleImageOCRResponse
from optimaize_ocr_api.core.config import get_settings
from optimaize_ocr_api.services.ai.ocr_service import MODELS, OCRService
from optimaize_ocr_api.services.data.file_store import FileStore

router = APIRouter(prefix="/ocr", tags=["ocr"])


def _effective_cpu_count() -> int:
    host_cpus = os.cpu_count() or 1
    cpu_max = "/sys/fs/cgroup/cpu.max"
    try:
        quota_text, period_text = open(cpu_max, encoding="utf-8").read().strip().split()[:2]
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                return max(1, min(host_cpus, round(quota / period)))
    except OSError:
        pass
    except (ValueError, IndexError):
        pass
    return host_cpus


@router.get("/models", response_model=ModelListResponse)
def models() -> ModelListResponse:
    return ModelListResponse(models=MODELS, default_model=MODELS[0])


@router.get("/runtime-config", response_model=RuntimeConfigResponse)
def runtime_config() -> RuntimeConfigResponse:
    logical_cpus = _effective_cpu_count()
    return RuntimeConfigResponse(
        logical_cpus=logical_cpus,
        threads=RuntimeRange(default=None, min=1, max=logical_cpus, recommended=min(logical_cpus, 4)),
        cpu_percent=RuntimeRange(default=None, min=1, max=100, recommended=None),
        labels={"default": "Backend default"},
    )


@router.post("/single-image", response_model=SingleImageOCRResponse)
async def single_image_ocr(
    image: UploadFile = File(...),
    model_type: str = Form("falcon-ocr"),
    threads: int | None = Form(None),
    cpu_percent: float | None = Form(None),
    layout_model: str = Form("PaddlePaddle/PP-DocLayoutV3_safetensors"),
    layout_threshold: float = Form(0.3),
    skip_layout: bool = Form(False),
    full_page_mode: str = Form("layout"),
    quantize_int8: bool | None = Form(None),
    quantize_mode: str = Form("selective"),
    auto_runtime: str = Form("off"),
    use_optimized_dots: bool = Form(True),
    dots_fuse_mlp_swiglu: bool = Form(True),
    dots_int8_lm_head: bool = Form(True),
    paddle_table_prompt: str = Form("fast"),
    save_crops: bool = Form(False),
) -> SingleImageOCRResponse:
    settings = get_settings()
    image_path, pil_image = await FileStore(settings.upload_dir, settings.max_upload_bytes).save_upload(image)
    result = OCRService(settings.output_dir).run_single_image(
        image_path=image_path,
        image=pil_image,
        model_type=model_type,
        threads=threads,
        cpu_percent=cpu_percent,
        layout_model=layout_model,
        layout_threshold=layout_threshold,
        skip_layout=skip_layout,
        full_page_mode=full_page_mode,
        quantize_int8=quantize_int8,
        quantize_mode=quantize_mode,
        auto_runtime=auto_runtime,
        use_optimized_dots=use_optimized_dots,
        dots_fuse_mlp_swiglu=dots_fuse_mlp_swiglu,
        dots_int8_lm_head=dots_int8_lm_head,
        paddle_table_prompt=paddle_table_prompt,
        save_crops=save_crops,
    )
    return SingleImageOCRResponse(**result)
