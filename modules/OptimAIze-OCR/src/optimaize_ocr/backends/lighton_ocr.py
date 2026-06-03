# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# CPU-optimized + Low-RAM integration backend for lightonai/LightOnOCR-2-1B.
#
# Optimization stack (same recipe as Dots-MOCR):
#   1. Dynamic INT8 layer-by-layer quantization to cut weights ~4x and avoid
#      the FP32 peak-RAM spike of a one-shot `quantize_dynamic`.
#   2. State_dict cache on disk -> empty skeleton + fast load on subsequent runs.
#   3. AVX2 / Numba GEMV kernels applied to every Linear at decode time.
#   4. RoPE monkey-patch covers Mistral / Llama / Qwen2 family automatically.

import gc
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import LightOnOcrProcessor, LightOnOcrForConditionalGeneration

from .base import BaseVLMBackend
from ..prompts import DEFAULT_CATEGORY_PROMPTS as CATEGORY_PROMPTS
from ..compute import (
    patch_model_linear_layers,
    quantize_model_layer_by_layer,
)
from ..runtime_policy import RuntimeMode, RuntimePolicy, build_runtime_policy, promoted_backend_profile
from ..paths import weights_dir

logger = logging.getLogger(__name__)


class LightOnOCRBackend(BaseVLMBackend):
    """CPU-optimized + Low-RAM backend for lightonai/LightOnOCR-2-1B.

    Set `quantize_int8=False` for maximum fidelity at the cost of ~4x more RAM.
    """

    def __init__(
        self,
        model_id: str = "lightonai/LightOnOCR-2-1B",
        device: str = "cpu",
        quantize_int8: bool = True,
        max_new_tokens: int = 1024,
        auto_runtime: RuntimeMode = "off",
    ):
        self.device = torch.device(device)
        self.quantize_int8 = quantize_int8
        self.max_new_tokens = max_new_tokens
        self.runtime_policy: RuntimePolicy | None = None

        logger.info(f"Loading LightOnOCR from HF Hub ({model_id}) on CPU...")

        # 1. Processor
        self.processor = LightOnOcrProcessor.from_pretrained(model_id)

        if auto_runtime != "off":
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            profile = promoted_backend_profile("lighton-ocr", model_id, config)
            self.runtime_policy = build_runtime_policy(profile, auto_runtime)

        # 2. Cached quantized weights live under the project's `weights/` dir
        cache_dir = weights_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "lighton_ocr_quantized.pt"

        if quantize_int8 and cache_path.exists():
            logger.info(f"Found pre-quantized LightOnOCR cache: {cache_path}")
            self.model = self._load_from_quantized_cache(model_id, cache_path)
        else:
            logger.info("No cached quantized weights — performing first-time load + quantization.")
            self.model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            if quantize_int8:
                self.model = quantize_model_layer_by_layer(self.model)
                logger.info(f"Persisting quantized state_dict to {cache_path}...")
                torch.save(self.model.state_dict(), cache_path)
                logger.info("Quantized state_dict saved.")

        # 3. Patch every Linear (quantized or float32) with AVX2/Numba GEMV.
        patch_model_linear_layers(self.model)

        self.model.to(self.device)
        self.model.eval()
        gc.collect()
        logger.info("LightOnOCR backend initialized successfully on CPU!")

    # ---------------------------------------------------------------------
    # Internal loaders
    # ---------------------------------------------------------------------

    def _load_from_quantized_cache(self, model_id: str, cache_path: Path):
        """Reconstruct an empty quantized skeleton and load the cached state_dict.

        This path avoids materializing the full FP32 model in memory.
        """
        from transformers import AutoConfig

        logger.info("Building empty model skeleton from config...")
        config = AutoConfig.from_pretrained(model_id)
        # Force FP32 to avoid CPU BFloat16 surprises
        config.torch_dtype = torch.float32

        model = LightOnOcrForConditionalGeneration._from_config(config)
        # Apply the same quantization topology so the state_dict slots align.
        model = quantize_model_layer_by_layer(model)

        logger.info("Loading cached quantized state_dict from disk...")
        state_dict = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("Cached quantized weights loaded.")
        return model

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        """Run CPU-optimized VLM OCR on a crop image."""
        # Prevent fold/unfold runtime error for tiny crops
        w, h = image.size
        if w < 32 or h < 32:
            new_w = max(w, 32)
            new_h = max(h, 32)
            padded = Image.new("RGB", (new_w, new_h), "white")
            padded.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
            image = padded

        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=self.device, dtype=torch.float32) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        inputs.pop("mm_token_type_ids", None)
        inputs.pop("token_type_ids", None)

        cat = category.strip().lower()
        non_table_cap = self.runtime_policy.non_table_max_new_tokens if self.runtime_policy else 16
        table_cap = self.runtime_policy.table_max_new_tokens if self.runtime_policy else None
        if cat == "table":
            cap = min(self.max_new_tokens, table_cap) if table_cap is not None else self.max_new_tokens
        else:
            cap = self.max_new_tokens if non_table_cap is None else min(self.max_new_tokens, non_table_cap)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=cap,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.model.generation_config.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return decoded.strip()
