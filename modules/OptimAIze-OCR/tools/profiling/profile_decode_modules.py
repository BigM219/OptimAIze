from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile single decode step at module-block level (not just Linear)."""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
import torch, torch.nn as nn
from optimaize_ocr.backends import get_vlm_backend

torch.set_num_threads(8)
backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')
_ = backend.generate_ocr(img, category='text')
_ = backend.generate_ocr(img, category='text')

from optimaize_ocr.prompts import DOTS_MOCR_CATEGORY_PROMPTS
instruction = DOTS_MOCR_CATEGORY_PROMPTS['text']
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = backend.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = backend.processor(images=img, text=prompt, return_tensors="pt")

with torch.inference_mode():
    out = backend.model(**inputs, use_cache=True)
    past = out.past_key_values

last_id = inputs['input_ids'][:, -1:]

# Hook everything in layer0 — Linear, attention, MLP, layer-norms.
lm = backend.model.model
layer0 = lm.layers[0]
times = {}
hooks = []
import functools
def pre(name, m, args, kwargs):
    m._t0 = time.perf_counter()
def post(name, m, args, kwargs, out):
    times.setdefault(name, []).append(time.perf_counter() - m._t0)

for sub_name, sub in layer0.named_modules():
    full = f'l0.{sub_name}' if sub_name else 'l0'
    hooks.append(sub.register_forward_pre_hook(functools.partial(pre, full), with_kwargs=True))
    hooks.append(sub.register_forward_hook(functools.partial(post, full), with_kwargs=True))

with torch.inference_mode():
    for _ in range(15):
        out2 = backend.model(input_ids=last_id, past_key_values=past, use_cache=True)
        past = out2.past_key_values
for h in hooks: h.remove()

print('Per-module decode time (layer 0, avg over 10 steps after warmup):')
for name, ts in sorted(times.items(), key=lambda kv: -sum(kv[1][5:])):
    avg_us = sum(ts[5:]) / max(1, len(ts[5:])) * 1e6
    if avg_us > 50:
        print(f'  {name:<55s}: {avg_us:>8.0f} us')
