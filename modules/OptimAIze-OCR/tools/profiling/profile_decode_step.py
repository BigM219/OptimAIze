from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile a single decode step: which Linear is dominant?"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
import torch
from optimaize_ocr.backends import get_vlm_backend

torch.set_num_threads(8)
backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)
backend.max_new_tokens = 1

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')

# Warmup decode pipeline
_ = backend.generate_ocr(img, category='text')
_ = backend.generate_ocr(img, category='text')

# Prepare inputs/cached state by running 1-token decode and stopping at start of decode loop
from optimaize_ocr.prompts import DOTS_MOCR_CATEGORY_PROMPTS
instruction = DOTS_MOCR_CATEGORY_PROMPTS['text']
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = backend.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = backend.processor(images=img, text=prompt, return_tensors="pt")

# Run prefill manually to populate KV cache
with torch.inference_mode():
    out = backend.model(**inputs, use_cache=True)
    past = out.past_key_values

# Now time the decode step (1 token)
last_id = inputs['input_ids'][:, -1:]
N = 8
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(N):
        out2 = backend.model(input_ids=last_id, past_key_values=past, use_cache=True)
        past = out2.past_key_values
dt = time.perf_counter() - t0
print(f'Avg decode step (after KV warm): {dt/N*1000:.1f}ms')

# Hook all Linears in LM
lm = backend.model.model
linear_times = {}
hooks = []
import functools
def pre(name, m, args, kwargs):
    m._t0 = time.perf_counter()
def post(name, m, args, kwargs, out):
    linear_times.setdefault(name, []).append(time.perf_counter() - m._t0)

# Patch one decoder layer + lm_head
layer0 = lm.layers[0]
for sub_name, sub in layer0.named_modules():
    if isinstance(sub, (torch.nn.Linear, torch.nn.quantized.dynamic.Linear)):
        full = f'layer0.{sub_name}'
        hooks.append(sub.register_forward_pre_hook(functools.partial(pre, full), with_kwargs=True))
        hooks.append(sub.register_forward_hook(functools.partial(post, full), with_kwargs=True))

lmh = backend.model.lm_head
hooks.append(lmh.register_forward_pre_hook(functools.partial(pre, 'lm_head'), with_kwargs=True))
hooks.append(lmh.register_forward_hook(functools.partial(post, 'lm_head'), with_kwargs=True))

with torch.inference_mode():
    for _ in range(20):
        out2 = backend.model(input_ids=last_id, past_key_values=past, use_cache=True)
        past = out2.past_key_values
for h in hooks: h.remove()

print('\nPer-Linear decode time (first decoder layer + lm_head):')
for name, ts in sorted(linear_times.items()):
    avg_us = sum(ts[5:]) / max(1, len(ts[5:])) * 1e6
    print(f'  {name:<50s}: {avg_us:>7.0f} us')
