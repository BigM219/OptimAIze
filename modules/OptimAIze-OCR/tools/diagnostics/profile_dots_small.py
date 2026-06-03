from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile Dots-MOCR small-crop latency: vision/prefill vs decode."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
import torch
from optimaize_ocr.backends import get_vlm_backend

backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)
backend.max_new_tokens = 8

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')
print(f'Image: {img.size}')

# Build inputs once
from optimaize_ocr.prompts import DOTS_MOCR_CATEGORY_PROMPTS
instruction = DOTS_MOCR_CATEGORY_PROMPTS['text']
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = backend.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = backend.processor(images=img, text=prompt, return_tensors="pt")
inputs.pop("mm_token_type_ids", None)

print(f'input_ids len = {inputs["input_ids"].shape[-1]}, '
      f'pixel patches = {inputs["pixel_values"].shape[0]}, '
      f'grid_thw = {inputs["image_grid_thw"].tolist()}')

# Warmup (Numba JIT)
print('\n=== Warmup ===')
with torch.inference_mode():
    _ = backend.model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)

# Now time the components separately
print('\n=== Timing breakdown ===')

# 1. Vision tower forward
with torch.inference_mode():
    t0 = time.perf_counter()
    ve = backend.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])
    t_vision = time.perf_counter() - t0
print(f'Vision tower forward: {t_vision*1000:.1f}ms (output {tuple(ve.shape)})')

# 2. Prefill (LM forward over all input_ids with vision injected)
with torch.inference_mode():
    t0 = time.perf_counter()
    out = backend.model(**inputs)
    t_prefill = time.perf_counter() - t0
print(f'Full prefill (vision + LM N tokens): {t_prefill*1000:.1f}ms')

# 3. Per-token decode (custom: just LM with KV cache for 8 new tokens)
with torch.inference_mode():
    t0 = time.perf_counter()
    out = backend.model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
    t_total_8 = time.perf_counter() - t0
print(f'Generate 8 tokens (incl prefill): {t_total_8*1000:.1f}ms')
print(f'Per-token decode (after prefill): {(t_total_8 - t_prefill) / 8 * 1000:.1f}ms')

# 4. Generate 32 tokens
with torch.inference_mode():
    t0 = time.perf_counter()
    out = backend.model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    t_total_32 = time.perf_counter() - t0
print(f'Generate 32 tokens (incl prefill): {t_total_32*1000:.1f}ms')
print(f'Avg per-token (32): {(t_total_32 - t_prefill) / 32 * 1000:.1f}ms')
