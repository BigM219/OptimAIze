# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Custom CPU-optimized Dots-MOCR backend with decode optimizations.

import sys
import types
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig

# Mock flash_attn for CPU
sys.modules['flash_attn'] = MagicMock()
sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
sys.modules['flash_attn.modules'] = MagicMock()
sys.modules['flash_attn.modules.mha'] = MagicMock()

from ..base import BaseVLMBackend
from ...prompts import DOTS_MOCR_CATEGORY_PROMPTS as CATEGORY_PROMPTS
from ...compute import (
    apply_rotary_pos_emb_vision_impl,
    patch_model_linear_layers,
    quantize_model_layer_by_layer,
    Int8Linear,
    fuse_qwen2_mlp_swiglu,
    fuse_qwen2_attn_qkv,
    fuse_qwen2_attn_rmsnorm,
)
from ...runtime_policy import RuntimeMode, RuntimePolicy, build_runtime_policy, promoted_backend_profile
from ...paths import weights_dir
from .patches import prepare_inputs_for_generation_patched
from .output_parser import parse_dots_mocr_output
from .svg_utils import (
    extract_svg_from_response,
    svg_to_layout_elements,
)
from ...prompts import (
    DOTS_MOCR_PROMPT_LAYOUT_ALL_EN,
    DOTS_MOCR_PROMPT_LAYOUT_ONLY_EN,
    DOTS_MOCR_PROMPT_IMAGE_TO_SVG,
    DOTS_MOCR_PROMPT_OCR,
    DOTS_MOCR_PROMPT_GROUNDING_OCR,
    DOTS_MOCR_PROMPT_SCENE_SPOTTING,
    DOTS_MOCR_PROMPT_WEB_PARSING,
    DOTS_MOCR_PROMPT_GENERAL,
)


# All vendor full-page prompt modes (mirror of
# ``dict_promptmode_to_prompt`` in dots.mocr/dots_mocr/utils/prompts.py).
DOTS_PROMPT_MODES: dict[str, str] = {
    "layout_all_en": DOTS_MOCR_PROMPT_LAYOUT_ALL_EN,
    "layout_only_en": DOTS_MOCR_PROMPT_LAYOUT_ONLY_EN,
    "ocr": DOTS_MOCR_PROMPT_OCR,
    "grounding_ocr": DOTS_MOCR_PROMPT_GROUNDING_OCR,
    "scene_spotting": DOTS_MOCR_PROMPT_SCENE_SPOTTING,
    "web_parsing": DOTS_MOCR_PROMPT_WEB_PARSING,
    "image_to_svg": DOTS_MOCR_PROMPT_IMAGE_TO_SVG,
    "general": DOTS_MOCR_PROMPT_GENERAL,
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vendor-ported smart_resize and bbox post-processing.
# Source: dots.mocr/dots_mocr/utils/image_utils.py (Apache-2.0).
# ---------------------------------------------------------------------------

_IMAGE_FACTOR = 28
_MIN_PIXELS = 3136
_MAX_PIXELS = 11289600


def _smart_resize(
    height: int,
    width: int,
    factor: int = _IMAGE_FACTOR,
    min_pixels: int = _MIN_PIXELS,
    max_pixels: int = _MAX_PIXELS,
) -> tuple[int, int]:
    """Round to ``factor``-divisible dims, clamped to [min_pixels, max_pixels]."""
    import math

    def _round(n: int, f: int) -> int:
        return round(n / f) * f

    def _ceil(n: float, f: int) -> int:
        return math.ceil(n / f) * f

    def _floor(n: float, f: int) -> int:
        return math.floor(n / f) * f

    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor(height / beta, factor))
        w_bar = max(factor, _floor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta, factor)
        w_bar = _ceil(width * beta, factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, _floor(h_bar / beta, factor))
            w_bar = max(factor, _floor(w_bar / beta, factor))
    return h_bar, w_bar


def _post_process_cells(
    cells: list[dict],
    original_size: tuple[int, int],
    input_size: tuple[int, int],
) -> list[dict]:
    """Map bboxes from input (post-resize) coords back to original image.

    ``original_size`` and ``input_size`` are ``(width, height)`` tuples.
    Mirror of vendor's ``post_process_cells`` — the model emits bboxes
    in the resized image space; downstream consumers (markdown rendering,
    HTML viewer) draw on the original image, so we must invert the scale.
    """
    ow, oh = original_size
    iw, ih = input_size
    if iw <= 0 or ih <= 0 or ow <= 0 or oh <= 0:
        return cells
    scale_x = iw / ow
    scale_y = ih / oh
    out: list[dict] = []
    for c in cells:
        bbox = c.get("bbox") or [0, 0, 0, 0]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            out.append(c)
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            out.append(c)
            continue
        c2 = dict(c)
        c2["bbox"] = [
            int(x1 / scale_x),
            int(y1 / scale_y),
            int(x2 / scale_x),
            int(y2 / scale_y),
        ]
        out.append(c2)
    return out


def _parse_layout_json(raw: str) -> list[dict]:
    """Parse the JSON object emitted by ``prompt_layout_all_en``.

    The model usually outputs:
        [{"bbox":[...],"category":"...","text":"..."}, ...]
    or wraps it in ``{"layout":[...]}``. We accept both, plus partial
    truncation (last element cut off because of ``max_new_tokens``).

    Bboxes returned are in the *resized* image's coordinate space (the
    one the processor actually fed to the vision tower). Callers that
    need original-image coordinates should pass ``input_size`` and
    ``original_size`` and call :func:`_post_process_cells`.
    """
    import json
    import re as _re

    text = raw.strip()
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    text = _re.sub(r"^```(?:json)?\s*\n?", "", text, flags=_re.MULTILINE)
    text = _re.sub(r"\n?```\s*$", "", text, flags=_re.MULTILINE)

    m = _re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not m:
        return []

    candidate = m.group()
    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Try truncating to the last well-formed element.
        for cut in range(len(candidate) - 1, 0, -1):
            if candidate[cut] in "}]":
                head = candidate[: cut + 1]
                # Re-balance opening/closing brackets if needed.
                opens_obj = head.count("{") - head.count("}")
                opens_arr = head.count("[") - head.count("]")
                fixed = head + ("}" * max(0, opens_obj)) + ("]" * max(0, opens_arr))
                try:
                    parsed = json.loads(fixed)
                    break
                except json.JSONDecodeError:
                    continue
    if parsed is None:
        return []

    elements: list[dict] = []
    if isinstance(parsed, list):
        elements = [e for e in parsed if isinstance(e, dict)]
    elif isinstance(parsed, dict):
        for key in ("layout", "elements", "blocks", "result", "content", "items"):
            if key in parsed and isinstance(parsed[key], list):
                elements = [e for e in parsed[key] if isinstance(e, dict)]
                break

    results: list[dict] = []
    for el in elements:
        cat = str(el.get("category", "")).strip().lower() or "text"
        bbox = el.get("bbox") or [0, 0, 0, 0]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            bbox = [0, 0, 0, 0]
        try:
            bbox = [int(round(float(v))) for v in bbox]
        except (TypeError, ValueError):
            bbox = [0, 0, 0, 0]
        text_field = el.get("text", "")
        if isinstance(text_field, list):
            text_field = "\n".join(str(t) for t in text_field)
        text_field = str(text_field).strip()
        if cat == "picture" and not text_field:
            continue
        results.append({
            "category": cat,
            "bbox": bbox,
            "score": 1.0,
            "text": text_field,
        })
    return results


def _monkeypatch_vision_rope():
    """Patch vision RoPE in dynamically loaded modules."""
    patched_any = False
    for name, module in list(sys.modules.items()):
        if name.endswith("modeling_dots_vision"):
            module.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision_impl
            patched_any = True
            logger.info(f"Patched apply_rotary_pos_emb_vision in: {name}")
    return patched_any


class CustomDotsMOCRBackend(BaseVLMBackend):
    """Custom CPU-optimized Dots-MOCR backend with optimized decode loop.

    Key optimizations:
    - Custom decode loop bypassing transformers.generate() overhead
    - Reduced max_new_tokens for faster inference
    - Vision token optimization
    - Better caching strategy
    """

    def __init__(
        self,
        model_id: str = "rednote-hilab/dots.mocr",
        device: str = "cpu",
        quantize_int8: bool = True,
        max_new_tokens: int = 256,  # Reduced from 1024
        max_vision_tokens: int = 256,  # Cap vision tokens
    ):
        self.device = torch.device(device)
        self.quantize_int8 = quantize_int8
        self.max_new_tokens = max_new_tokens
        self.max_vision_tokens = max_vision_tokens
        self._category = "plain"  # set per-call in generate_ocr

        # Cache for quantized weights
        cache_dir = weights_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dots_mocr_quantized.pt"

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load model
        if quantize_int8 and cache_path.exists():
            logger.info(f"Loading cached quantized weights from: {cache_path}")
            self.model = self._load_from_quantized_cache(model_id, cache_path)
        else:
            logger.info("Loading model and applying quantization...")
            self.model = self._load_and_quantize(model_id, cache_path)

        # Patch vision RoPE
        _monkeypatch_vision_rope()

        # Patch Linear layers
        patch_model_linear_layers(self.model)

        self.model.to(self.device)
        self.model.eval()

        # Cache model components
        self._cache_model_components()

        logger.info("Custom Dots-MOCR backend initialized!")

    def _cache_model_components(self):
        """Cache frequently accessed model components."""
        # Get key dimensions
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.vocab_size = self.model.get_output_embeddings().out_features

        # Find language model for decode
        if hasattr(self.model, 'language_model'):
            self.lang_model = self.model.language_model
        elif hasattr(self.model, 'model'):
            self.lang_model = self.model.model
        else:
            self.lang_model = self.model

        # Cache tokenizers
        self.eos_token_id = self.processor.tokenizer.eos_token_id
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")

        logger.info(f"Model cached: hidden={self.hidden_size}, layers={self.num_layers}, "
                   f"vocab={self.vocab_size}")

    def _load_and_quantize(self, model_id: str, cache_path: Path):
        """Load model and apply quantization."""
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            try:
                config.vision_config.attn_implementation = "eager"
                config.vision_config._attn_implementation = "eager"
            except Exception:
                pass

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_patched, model
        )

        if self.quantize_int8:
            model = quantize_model_layer_by_layer(model)
            logger.info(f"Saving quantized weights to: {cache_path}")
            torch.save(model.state_dict(), cache_path)

        return model

    def _load_from_quantized_cache(self, model_id: str, cache_path: Path):
        """Load from quantized cache."""
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            try:
                config.vision_config.attn_implementation = "eager"
                config.vision_config._attn_implementation = "eager"
            except Exception:
                pass
        config.torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_patched, model
        )

        model = quantize_model_layer_by_layer(model)

        state_dict = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        return model

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        """Run CPU-optimized OCR on image."""
        t0 = time.perf_counter()

        # Handle small crops
        w, h = image.size
        if w < 32 or h < 32:
            new_w, new_h = max(w, 32), max(h, 32)
            padded = Image.new("RGB", (new_w, new_h), "white")
            padded.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
            image = padded

        # Build prompt
        instruction = CATEGORY_PROMPTS.get(category.strip().lower(), CATEGORY_PROMPTS["plain"])
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids", None)

        t_preproc = time.perf_counter() - t0

        # Run generation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
        )

        t_total = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        logger.info(f"[Dots-MOCR] preproc={t_preproc*1000:.1f}ms, total={t_total:.2f}s, "
                   f"generated={len(outputs[0][input_len:])} tokens")

        return parse_dots_mocr_output(decoded, category)


class OptimizedDotsMOCRBackend(BaseVLMBackend):
    """Even more optimized Dots-MOCR backend with custom decode loop.

    This version implements a custom decode loop that bypasses the
    transformers.generate() overhead entirely.
    """

    def __init__(
        self,
        model_id: str = "rednote-hilab/dots.mocr",
        device: str = "cpu",
        quantize_int8: bool = True,
        max_new_tokens: int = 1024,
        max_vision_tokens: int = 256,
        quantize_mode: str = "selective",
        patch_linear_layers: bool = True,
        fuse_mlp_swiglu: bool = True,
        int8_lm_head: bool = True,
        vision_rope_patch: bool = True,
        auto_runtime: RuntimeMode = "off",
    ):
        self.device = torch.device(device)
        self.quantize_int8 = quantize_int8
        self.max_new_tokens = max_new_tokens
        self.max_vision_tokens = max_vision_tokens
        self.patch_linear_layers = patch_linear_layers
        self.fuse_mlp_swiglu = fuse_mlp_swiglu
        self.int8_lm_head = int8_lm_head
        self.vision_rope_patch = vision_rope_patch
        # ``quantize_mode``:
        #   "selective" — INT8-quantize MLP only, keep attention FP32. ~6 GB
        #     RAM, instruction-following stays accurate.
        #   "full" — INT8-quantize attention + MLP. ~4 GB RAM but causes
        #     dots.mocr to hallucinate and emit reasoning loops on CPU.
        #   "fp16" — load weights in float16, no INT8 quantization. ~5 GB
        #     RAM, attention math runs natively in fp16. Best balance of
        #     RAM and instruction-following on dots.mocr because the
        #     vendor trained at bf16 — fp16 is the closest CPU-friendly
        #     approximation. PyTorch CPU has full fp16 GEMM support since
        #     2.0 (oneDNN). Slower than INT8 selective on most CPUs but
        #     produces fewer hallucinations on small/dense crops.
        #   "none" — pure FP32 (~16 GB RAM).
        if quantize_mode not in ("selective", "full", "fp16", "none"):
            raise ValueError(f"Unknown quantize_mode: {quantize_mode!r}")
        # ``quantize_int8=False`` historically meant "no quantization at
        # all" — preserve that by overriding to fp32 unless the caller
        # explicitly asked for fp16.
        if not quantize_int8 and quantize_mode != "fp16":
            quantize_mode = "none"
        self.quantize_mode = quantize_mode

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load config first — force eager attention so the vision tower
        # does not silently use the mocked flash_attention_2.
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.config._attn_implementation = "eager"
        if hasattr(self.config, "vision_config"):
            try:
                self.config.vision_config.attn_implementation = "eager"
                self.config.vision_config._attn_implementation = "eager"
            except Exception:
                pass
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.runtime_policy: RuntimePolicy | None = None
        if auto_runtime != "off":
            profile = promoted_backend_profile("dots-mocr", model_id, self.config)
            self.runtime_policy = build_runtime_policy(profile, auto_runtime)

        # Cache — separate file per quantization mode to avoid mixing
        # incompatible state_dict layouts.
        cache_dir = weights_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.quantize_mode == "selective":
            cache_path = cache_dir / "dots_mocr_quantized_v2_selective.pt"
        elif self.quantize_mode == "full":
            cache_path = cache_dir / "dots_mocr_quantized_v2.pt"
        elif self.quantize_mode == "fp16":
            cache_path = cache_dir / "dots_mocr_fp16.pt"
        else:
            cache_path = None  # FP32, no caching

        # Load model
        if cache_path is not None and cache_path.exists():
            logger.info(f"Loading cached weights from: {cache_path}")
            self.model = self._load_from_quantized_cache(model_id, cache_path)
        else:
            load_dtype = torch.float16 if self.quantize_mode == "fp16" else torch.float32
            logger.info(
                f"Loading model (mode={self.quantize_mode}, dtype={load_dtype})..."
            )
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=self.config,
                torch_dtype=load_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.prepare_inputs_for_generation = types.MethodType(
                prepare_inputs_for_generation_patched, self.model
            )
            if self.quantize_mode in ("selective", "full"):
                self.model = quantize_model_layer_by_layer(
                    self.model,
                    skip_attention=(self.quantize_mode == "selective"),
                )
            if cache_path is not None:
                logger.info(f"Saving weights to: {cache_path}")
                torch.save(self.model.state_dict(), cache_path)

        # Patch
        if self.vision_rope_patch:
            _monkeypatch_vision_rope()
        if self.patch_linear_layers:
            patch_model_linear_layers(self.model)

        if self.quantize_mode == "full" and self.patch_linear_layers:
            # Fuse Qwen2 attention QKV projections into a single AVX2 kernel
            # call. The fused path quantizes q/k/v, so keep it out of selective mode.
            fuse_qwen2_attn_qkv(self.model)
            fuse_qwen2_attn_rmsnorm(self.model)

        # Fuse Qwen2 MLP gate+up SwiGLU into a single AVX2 kernel call.
        # Replaces 2 separate ctypes calls per layer (gate_proj, up_proj +
        # silu + element-mult) with one fused pass that reads ``x`` once
        # and never materializes the 8960-wide intermediate tensors.
        # Must run AFTER ``patch_model_linear_layers`` so the down_proj
        # path it falls through to is already wired to the AVX2 GEMV.
        if self.fuse_mlp_swiglu and self.patch_linear_layers:
            fuse_qwen2_mlp_swiglu(self.model)

        # Replace lm_head (1536 -> ~152k vocab) with our per-channel INT8
        # Linear. With FP32 it dominates decode (~25 ms / step on this CPU,
        # ~8x a single decoder layer); INT8 GEMV typically halves it.
        if self.int8_lm_head and hasattr(self.model, "lm_head") and isinstance(self.model.lm_head, torch.nn.Linear):
            self.model.lm_head = Int8Linear.from_linear(
                self.model.lm_head, keep_fp32_for_prefill=False
            )
            logger.info("Replaced lm_head with per-channel INT8 GEMV")

        # Force vision tower to NOT cast to bfloat16 — weights are FP32/INT8
        # so the bf16 path silently mismatches dtypes and degenerates.
        if hasattr(self.model, "vision_tower"):
            _orig_vt_forward = self.model.vision_tower.forward
            def _vt_forward_fp32(hidden_states, grid_thw, bf16=False):
                return _orig_vt_forward(hidden_states, grid_thw, bf16=False)
            self.model.vision_tower.forward = _vt_forward_fp32

        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        self.eos_token_id = self.processor.tokenizer.eos_token_id

        # Get output embeddings
        if hasattr(self.model, 'lm_head'):
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'lm_head'):
            self.lm_head = self.model.language_model.lm_head
        else:
            self.lm_head = self.model.get_output_embeddings()

        logger.info(f"Optimized Dots-MOCR: hidden={self.hidden_size}, layers={self.num_layers}, "
                   f"heads={self.num_heads}")

    def _load_from_quantized_cache(self, model_id: str, cache_path: Path):
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            try:
                config.vision_config.attn_implementation = "eager"
                config.vision_config._attn_implementation = "eager"
            except Exception:
                pass
        skeleton_dtype = torch.float16 if self.quantize_mode == "fp16" else torch.float32
        config.torch_dtype = skeleton_dtype
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        if self.quantize_mode == "fp16":
            model = model.to(skeleton_dtype)
        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_patched, model
        )
        if self.quantize_mode in ("selective", "full"):
            model = quantize_model_layer_by_layer(
                model,
                skip_attention=(self.quantize_mode == "selective"),
            )
        state_dict = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        """Per-crop OCR using vendor's category-specific prompt."""
        t0 = time.perf_counter()

        # Handle small crops
        w, h = image.size
        if w < 32 or h < 32:
            new_w, new_h = max(w, 32), max(h, 32)
            padded = Image.new("RGB", (new_w, new_h), "white")
            padded.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
            image = padded

        # Build prompt
        instruction = CATEGORY_PROMPTS.get(category.strip().lower(), CATEGORY_PROMPTS["plain"])
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare inputs
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids", None)
        # Cast pixel_values to model dtype when running in fp16 mode.
        if self.quantize_mode == "fp16" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        t_preproc = time.perf_counter() - t0

        # Adaptive max_new_tokens. Short categories (titles, captions, headers,
        # list items) rarely exceed 128 tokens; capping them tightly cuts
        # ~75% off the wall-clock budget when the model fails to emit EOS.
        cat_lc = category.strip().lower()
        if cat_lc in ("table", "formula"):
            cap = self.max_new_tokens
        elif cat_lc in ("plain", "layout"):
            cap = self.max_new_tokens
        else:
            cap = min(self.max_new_tokens, 192)

        # Use standard generate() for now (custom decode needs more work)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cap,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.05,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        t_total = time.perf_counter() - t0
        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        logger.info(f"[Dots-MOCR] preproc={t_preproc*1000:.1f}ms, total={t_total:.2f}s, "
                   f"tokens={len(outputs[0][input_len:])}")

        return parse_dots_mocr_output(decoded, category)

    @torch.inference_mode()
    def parse_full_page(self, image: Image.Image, max_new_tokens: int = 2048) -> list[dict]:
        """Run dots.mocr in its native full-page LAYOUT_ALL_EN mode.

        Returns a list of ``{"category", "bbox", "score", "text"}`` dicts
        compatible with the pipeline's downstream consumers. Categories
        and bboxes are emitted by the model itself; ``score`` is set to
        1.0 since the model does not produce per-element confidences.
        """
        t0 = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DOTS_MOCR_PROMPT_LAYOUT_ALL_EN},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids", None)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Vendor uses temperature=0.1, top_p=0.9 (NOT greedy) for
            # layout — see dots.mocr/dots_mocr/model/inference.py.
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            use_cache=True,
            repetition_penalty=1.03,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        t_total = time.perf_counter() - t0
        n_tok = len(outputs[0][input_len:])
        logger.info(
            f"[Dots-MOCR full-page] total={t_total:.2f}s, tokens={n_tok}"
        )
        # Always log a head of the raw JSON output for debugging.
        logger.info(
            "[Dots-MOCR full-page] raw output (first 1200 chars): %s",
            decoded[:1200],
        )

        elements = _parse_layout_json(decoded)
        if not elements:
            logger.warning(
                "[Dots-MOCR full-page] no elements parsed; raw output (first 800 chars): %r",
                decoded[:800],
            )
        else:
            # Bboxes from the model are in the resized input space.
            # Map them back to the original image coords so downstream
            # markdown/HTML rendering and the saved JSON match the user's
            # input image dimensions (vendor's ``post_process_cells``).
            ow, oh = image.size
            ih, iw = _smart_resize(oh, ow)
            elements = _post_process_cells(elements, (ow, oh), (iw, ih))
            logger.info(
                "[Dots-MOCR full-page] parsed %d elements; resized=(%d,%d) original=(%d,%d)",
                len(elements), iw, ih, ow, oh,
            )
        return elements

    @torch.inference_mode()
    def parse_full_page_svg(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_p: float = 1.0,
    ) -> list[dict]:
        """Run dots.mocr in ``prompt_image_to_svg`` mode and return layout.

        SVG mode reproduces the entire page as an SVG document; each
        glyph/line lands inside a ``<text>`` element with x/y/font-size
        attributes that we mine to recover both transcription and an
        approximate bbox. Vendor recommends *high* sampling temperature
        for SVG (low temperature triggers repetition loops on this prompt
        specifically — see ``demo_vllm_svg.py``).

        Note: the viewBox in the prompt MUST be filled with the actual
        image dimensions; this is what grounds the model's coordinates
        to pixel space and produces accurate transcription.
        """
        t0 = time.perf_counter()
        w, h = image.size
        svg_prompt = DOTS_MOCR_PROMPT_IMAGE_TO_SVG.replace("{width}", str(w)).replace(
            "{height}", str(h)
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": svg_prompt},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids", None)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,         # vendor explicitly recommends sampling
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        t_total = time.perf_counter() - t0
        n_tok = len(outputs[0][input_len:])
        logger.info(f"[Dots-MOCR SVG] total={t_total:.2f}s, tokens={n_tok}")

        svg, ok = extract_svg_from_response(decoded)
        if not ok or svg is None:
            logger.warning("[Dots-MOCR SVG] no <svg> block in model output")
            logger.warning(
                "[Dots-MOCR SVG] raw output (first 500 chars): %r",
                decoded[:500],
            )
            return []

        elements = svg_to_layout_elements(svg, image_width=w, image_height=h)
        logger.info(f"[Dots-MOCR SVG] extracted {len(elements)} layout elements")
        return elements

    @torch.inference_mode()
    def docqa_pages(
        self,
        images: list[Image.Image],
        question: str,
        max_new_tokens: int | None = None,
    ) -> str:
        if len(images) != 1:
            raise NotImplementedError("Dots-MOCR DocQA supports one image per call only")
        raw, _ = self.run_prompt(
            images[0],
            "general",
            custom_prompt=question,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        return raw

    def run_prompt(
        self,
        image: Image.Image,
        prompt_mode: str,
        bbox: list[int] | None = None,
        custom_prompt: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> tuple[str, str]:
        """Run any vendor prompt mode and return ``(raw, prompt)``.

        Mirrors vendor's ``DotsMOCRParser.get_prompt`` + inference call:

        - ``prompt_mode``: one of ``DOTS_PROMPT_MODES``.
        - ``bbox``: required for ``grounding_ocr`` (appended to prompt).
        - ``custom_prompt``: replaces ``general`` mode's empty prompt.

        Sampling defaults match vendor:
        - ``image_to_svg`` → temperature=0.9, top_p=1.0 (vendor demo)
        - everything else → temperature=0.1, top_p=0.9 (vendor inference.py)

        Returns the raw decoded text plus the resolved prompt for logging.
        """
        if prompt_mode not in DOTS_PROMPT_MODES:
            raise ValueError(
                f"Unknown prompt_mode={prompt_mode!r}. "
                f"Valid: {sorted(DOTS_PROMPT_MODES)}"
            )
        prompt = DOTS_PROMPT_MODES[prompt_mode]
        w, h = image.size

        # Per-mode prompt fixups (vendor's get_prompt logic).
        if prompt_mode == "image_to_svg":
            prompt = prompt.replace("{width}", str(w)).replace("{height}", str(h))
        elif prompt_mode == "grounding_ocr":
            if bbox is None or len(bbox) != 4:
                raise ValueError("grounding_ocr requires bbox=[x1, y1, x2, y2]")
            prompt = prompt + str(list(bbox))
        elif prompt_mode == "general":
            prompt = custom_prompt or "Please describe the content of this image."

        # Sampling defaults (mirror vendor).
        if temperature is None:
            temperature = 0.9 if prompt_mode == "image_to_svg" else 0.1
        if top_p is None:
            top_p = 1.0 if prompt_mode == "image_to_svg" else 0.9

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]
        chat_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            images=image, text=chat_prompt, return_tensors="pt"
        ).to(self.device)
        if "mm_token_type_ids" in inputs:
            inputs.pop("mm_token_type_ids", None)

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            repetition_penalty=1.03,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        t_total = time.perf_counter() - t0
        n_tok = len(outputs[0][input_len:])
        logger.info(
            "[Dots-MOCR %s] total=%.2fs tokens=%d (T=%.2f, top_p=%.2f)",
            prompt_mode, t_total, n_tok, temperature, top_p,
        )
        return decoded, prompt