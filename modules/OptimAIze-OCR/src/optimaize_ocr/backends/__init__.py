# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Export modular CPU Vision-Language Model backends.

from .base import BaseVLMBackend
from .falcon_ocr import FalconOCRBackend
from .lighton_ocr import LightOnOCRBackend
from .dots_mocr import DotsMOCRBackend
from .dots_mocr.custom_backend import OptimizedDotsMOCRBackend
from .paddleocr_vl import PaddleOCRVLBackend
from .surya_ocr import SuryaOCRBackend
from .surya_package import SuryaPackageBackend
from .glm_ocr import GLMOCRBackend
from ..runtime_policy import RuntimeMode

def get_vlm_backend(
    model_type: str,
    device: str = "cpu",
    quantize_int8: bool | None = None,
    use_optimized_dots: bool = False,
    quantize_mode: str = "selective",
    dots_fuse_mlp_swiglu: bool = True,
    dots_int8_lm_head: bool = True,
    auto_runtime: RuntimeMode = "off",
    paddle_table_prompt: str = "fast",
) -> BaseVLMBackend:
    """Instantiate a CPU-optimized VLM OCR backend by model name.

    Args:
        model_type: One of 'falcon-ocr', 'lighton-ocr', 'dots-mocr',
            'paddleocr-vl', 'surya-ocr', or 'glm-ocr' (also accepts the full HF repo id).
        device: Torch device. CPU-only is supported.
        quantize_int8: Dynamic INT8 quantization toggle. When None we use a
            per-model default: False for Falcon-OCR and LightOn-OCR (fidelity
            and speed are better in FP32), True for Dots-MOCR (large model that
            blows up RAM otherwise).
        use_optimized_dots: Use optimized Dots-MOCR backend with custom
            decode loop (significantly faster on CPU).
        quantize_mode: Dots-MOCR only. ``selective`` (default) keeps
            attention FP32 + quantizes MLP to INT8 (~6 GB). ``full``
            quantizes everything (~4 GB but hallucinates). ``fp16`` runs
            the whole model in float16 (~5 GB, best instruction-following
            on CPU). ``none`` is full FP32 (~16 GB).
    """
    model_type_clean = model_type.strip().lower()
    if model_type_clean in ("falcon-ocr", "tiiuae/falcon-ocr"):
        return FalconOCRBackend(
            device=device,
            quantize_int8=(False if quantize_int8 is None else quantize_int8),
            auto_runtime=auto_runtime,
        )
    elif model_type_clean in ("lighton-ocr", "lightonai/lightonocr-2-1b"):
        return LightOnOCRBackend(
            device=device,
            quantize_int8=(False if quantize_int8 is None else quantize_int8),
            auto_runtime=auto_runtime,
        )
    elif model_type_clean in ("dots-mocr", "rednote-hilab/dots.mocr"):
        dots_quantize_mode = "selective" if quantize_mode == "auto" else quantize_mode
        if use_optimized_dots:
            return OptimizedDotsMOCRBackend(
                device=device,
                quantize_int8=(True if quantize_int8 is None else quantize_int8),
                quantize_mode=dots_quantize_mode,
                fuse_mlp_swiglu=dots_fuse_mlp_swiglu,
                int8_lm_head=dots_int8_lm_head,
                auto_runtime=auto_runtime,
            )
        return DotsMOCRBackend(
            device=device,
            quantize_int8=(True if quantize_int8 is None else quantize_int8),
        )
    elif model_type_clean in ("paddleocr-vl", "paddlepaddle/paddleocr-vl-1.6"):
        return PaddleOCRVLBackend(
            device=device,
            quantize_mode=quantize_mode,
            auto_runtime=auto_runtime,
            table_prompt=paddle_table_prompt,
        )
    elif model_type_clean == "paddlepaddle/paddleocr-vl-1.5":
        return PaddleOCRVLBackend(
            model_id="PaddlePaddle/PaddleOCR-VL-1.5",
            device=device,
            quantize_mode=quantize_mode,
            auto_runtime=auto_runtime,
        )
    elif model_type_clean in ("surya-ocr", "surya-ocr-2", "datalab-to/surya-ocr-2"):
        return SuryaOCRBackend(device=device, quantize_mode=quantize_mode, auto_runtime=auto_runtime)
    elif model_type_clean in ("surya-package", "surya-ocr-package"):
        return SuryaPackageBackend(device=device)
    elif model_type_clean in ("glm-ocr", "zai-org/glm-ocr"):
        return GLMOCRBackend(device=device, quantize_mode=quantize_mode, auto_runtime=auto_runtime)
    else:
        raise ValueError(
            f"Unsupported VLM backend: {model_type}. "
            "Supported backends: 'falcon-ocr', 'lighton-ocr', 'dots-mocr', "
            "'paddleocr-vl', 'surya-ocr', 'surya-package', 'glm-ocr'."
        )
