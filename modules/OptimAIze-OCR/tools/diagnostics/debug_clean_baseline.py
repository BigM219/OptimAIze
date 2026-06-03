from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Debug script: load dots.mocr WITHOUT our patches and run vendor inference.

This bypasses:
- patch_model_linear_layers (Numba GEMV)
- _monkeypatch_vision_rope (our optimized RoPE)
- Int8Linear lm_head replacement
- INT8 quantization
- vision_tower forward override

If this produces correct output, our patches are the bug.
If this also hallucinates, the model itself struggles on CPU.
"""
import os
import sys
import logging
from unittest.mock import MagicMock

# Mock flash_attn for CPU
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
sys.modules["flash_attn.modules"] = MagicMock()
sys.modules["flash_attn.modules.mha"] = MagicMock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

MODEL_ID = "rednote-hilab/dots.mocr"
IMAGE_PATH = "assets/IC-Basic-Document-Control-Template-Example.png"
PROMPT = "Extract the text content from this image."

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Loading config (eager attention)...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config._attn_implementation = "eager"
if hasattr(config, "vision_config"):
    config.vision_config.attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"

print("Loading model in FP32, no quantization, no patches...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Patch ONLY prepare_inputs_for_generation (transformers 5.x compatibility)
import types
from optimaize_ocr.backends.dots_mocr.patches import (
    prepare_inputs_for_generation_patched,
)
model.prepare_inputs_for_generation = types.MethodType(
    prepare_inputs_for_generation_patched, model
)

# Force vision_tower to NOT cast to bf16 — vendor default is bf16=True but
# we have FP32 weights so it would silently mismatch.
if hasattr(model, "vision_tower"):
    _orig = model.vision_tower.forward
    def _vt_fp32(hidden_states, grid_thw, bf16=False):
        return _orig(hidden_states, grid_thw, bf16=False)
    model.vision_tower.forward = _vt_fp32

model.eval()
print("Model ready.")

print(f"Loading image {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image size: {image.size}")

messages = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print(f"Chat-templated prompt (last 200 chars): {prompt[-200:]!r}")

inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs.pop("mm_token_type_ids", None)
print(f"input_ids.shape={inputs['input_ids'].shape}")
print(f"pixel_values.shape={inputs['pixel_values'].shape}")
print(f"image_grid_thw={inputs['image_grid_thw']}")

print("Running generate (greedy, max_new_tokens=64)...")
import time
t0 = time.perf_counter()
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
dt = time.perf_counter() - t0
print(f"Done in {dt:.1f}s")

input_len = inputs["input_ids"].shape[-1]
decoded = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"=" * 60)
print(f"OUTPUT ({len(outputs[0][input_len:])} tokens):")
print(f"=" * 60)
print(repr(decoded))
print(f"=" * 60)
