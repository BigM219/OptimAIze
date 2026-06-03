# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# CPU-optimized + Low-RAM integration backend for rednote-hilab/dots.mocr.

import sys
from unittest.mock import MagicMock

# Mock flash_attn to bypass Hugging Face environment checks on CPU.
# Must be set BEFORE we import transformers.AutoModelForCausalLM with
# trust_remote_code=True for dots.mocr.
sys.modules['flash_attn'] = MagicMock()
sys.modules['flash_attn.flash_attn_interface'] = MagicMock()
sys.modules['flash_attn.modules'] = MagicMock()
sys.modules['flash_attn.modules.mha'] = MagicMock()

import gc
import logging
import types
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from ..base import BaseVLMBackend
from ...prompts import DOTS_MOCR_CATEGORY_PROMPTS as CATEGORY_PROMPTS
from ...compute import (
    apply_rotary_pos_emb_vision_impl,
    patch_model_linear_layers,
    quantize_model_layer_by_layer,
)
from ...paths import weights_dir
from .patches import prepare_inputs_for_generation_patched
from .output_parser import parse_dots_mocr_output

logger = logging.getLogger(__name__)


def _monkeypatch_vision_rope() -> bool:
    """Patch every loaded modeling_dots_vision module's vision RoPE.

    dots.mocr loads its modeling code dynamically (trust_remote_code=True),
    so we have to wait until the model is imported and then walk sys.modules
    to find the relevant module(s).
    """
    patched_any = False
    for name, module in list(sys.modules.items()):
        if name.endswith("modeling_dots_vision"):
            module.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision_impl
            patched_any = True
            logger.info(f"Patched apply_rotary_pos_emb_vision in dynamically loaded module: {name}")
    if not patched_any:
        logger.warning("Could not find a loaded modeling_dots_vision module to patch.")
    return patched_any


class DotsMOCRBackend(BaseVLMBackend):
    """CPU + Low-RAM optimized backend for rednote-hilab/dots.mocr.

    Set `quantize_int8=False` for max fidelity (uses ~4x more RAM).
    """

    def __init__(
        self,
        model_id: str = "rednote-hilab/dots.mocr",
        device: str = "cpu",
        quantize_int8: bool = True,
        max_new_tokens: int = 1024,
    ):
        self.device = torch.device(device)
        self.quantize_int8 = quantize_int8
        self.max_new_tokens = max_new_tokens

        # Project-relative cache for quantized weights
        cache_dir = weights_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "dots_mocr_quantized.pt"

        # 1. Processor (always loaded fresh — small + carries tokenizer state)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # 1b. Build a config that forces eager/sdpa attention everywhere.
        # The dots.mocr vision tower defaults to flash_attention_2 which we
        # mock out for CPU — letting it run would silently produce garbage.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"
        if hasattr(config, "vision_config"):
            try:
                config.vision_config.attn_implementation = "eager"
                config.vision_config._attn_implementation = "eager"
            except Exception:
                pass

        # 2. Model: prefer the cached quantized skeleton when available.
        if quantize_int8 and cache_path.exists():
            logger.info(f"Found pre-quantized DotsMOCR cache: {cache_path}")
            self.model = self._load_from_quantized_cache(model_id, cache_path, config)
        else:
            logger.info("No cached quantized weights — performing first-time load + quantization.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.prepare_inputs_for_generation = types.MethodType(
                prepare_inputs_for_generation_patched, self.model
            )

            if quantize_int8:
                self.model = quantize_model_layer_by_layer(self.model)
                logger.info(f"Persisting quantized state_dict to {cache_path}...")
                torch.save(self.model.state_dict(), cache_path)
                logger.info("Quantized state_dict saved.")

        # 3. Patch vision RoPE on the dynamically-loaded modeling module
        _monkeypatch_vision_rope()

        # 4. Patch every Linear (quantized or float32) with AVX2/Numba GEMV
        patch_model_linear_layers(self.model)

        # Force vision tower to NOT cast to bfloat16 — weights are FP32/INT8
        # so the bf16 path silently mismatches dtypes and degenerates.
        if hasattr(self.model, "vision_tower"):
            _orig_vt_forward = self.model.vision_tower.forward
            def _vt_forward_fp32(hidden_states, grid_thw, bf16=False):
                return _orig_vt_forward(hidden_states, grid_thw, bf16=False)
            self.model.vision_tower.forward = _vt_forward_fp32

        self.model.to(self.device)
        self.model.eval()
        gc.collect()
        logger.info("DotsMOCR backend initialized successfully on CPU!")

    # ---------------------------------------------------------------------
    # Internal loaders
    # ---------------------------------------------------------------------

    def _load_from_quantized_cache(self, model_id: str, cache_path: Path, config=None):
        """Build an empty quantized skeleton and load the cached state_dict
        without ever materializing the full FP32 model in memory.
        """
        from transformers import AutoConfig

        logger.info("Building empty model skeleton from config...")
        if config is None:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config._attn_implementation = "eager"
            if hasattr(config, "vision_config"):
                try:
                    config.vision_config.attn_implementation = "eager"
                    config.vision_config._attn_implementation = "eager"
                except Exception:
                    pass
        config.torch_dtype = torch.float32

        # `from_config` keeps weights uninitialized/random — super-fast (<0.5s)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation_patched, model
        )

        # Apply identical quantization topology so state_dict keys line up
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
        """Run CPU-optimized VLM OCR on a crop image using dots.mocr."""
        # Prevent fold/unfold runtime error for tiny crops
        w, h = image.size
        if w < 32 or h < 32:
            new_w = max(w, 32)
            new_h = max(h, 32)
            padded = Image.new("RGB", (new_w, new_h), "white")
            padded.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
            image = padded

        instruction = CATEGORY_PROMPTS.get(category.strip().lower(), CATEGORY_PROMPTS["plain"])

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)
        inputs.pop("mm_token_type_ids", None)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,    # deterministic greedy
            use_cache=True,
            repetition_penalty=1.05,
        )

        input_len = inputs["input_ids"].shape[-1]
        decoded = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return parse_dots_mocr_output(decoded, category)
