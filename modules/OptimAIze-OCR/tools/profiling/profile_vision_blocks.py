from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Micro-benchmark: where in vision_tower does time go?"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
import torch
from optimaize_ocr.backends import get_vlm_backend

backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')
from optimaize_ocr.prompts import DOTS_MOCR_CATEGORY_PROMPTS
instruction = DOTS_MOCR_CATEGORY_PROMPTS['text']
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = backend.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = backend.processor(images=img, text=prompt, return_tensors="pt")

# Warmup
with torch.inference_mode():
    _ = backend.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])
    _ = backend.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])

# Time vision_tower 5 times
ts = []
with torch.inference_mode():
    for _ in range(5):
        t0 = time.perf_counter()
        _ = backend.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])
        ts.append(time.perf_counter() - t0)
print(f'Vision tower fwd: avg {sum(ts)/5*1000:.1f}ms, min {min(ts)*1000:.1f}ms, max {max(ts)*1000:.1f}ms')

# Now hook each block to time it
vt = backend.model.vision_tower
block_times = [0.0] * len(vt.blocks)
hooks = []
import functools
def pre(i, m, args, kwargs):
    m._t0 = time.perf_counter()
def post(i, m, args, kwargs, out):
    block_times[i] += time.perf_counter() - m._t0
for i, blk in enumerate(vt.blocks):
    hooks.append(blk.register_forward_pre_hook(functools.partial(pre, i), with_kwargs=True))
    hooks.append(blk.register_forward_hook(functools.partial(post, i), with_kwargs=True))

with torch.inference_mode():
    _ = backend.model.vision_tower(inputs['pixel_values'], inputs['image_grid_thw'])

for h in hooks: h.remove()

print(f'\nPer-block forward time:')
for i, t in enumerate(block_times[:5]):
    print(f'  block[{i}]: {t*1000:.2f}ms')
print(f'  ...')
print(f'  total {len(block_times)} blocks: {sum(block_times)*1000:.1f}ms')
print(f'  avg per block: {sum(block_times)/len(block_times)*1000:.2f}ms')

# Time individual Linear layers within one block
blk = vt.blocks[0]
qkv = blk.attn.qkv
proj = blk.attn.proj
mlp = blk.mlp

x = torch.randn(88, 1536)
ts_qkv, ts_proj, ts_fc1, ts_fc2, ts_fc3 = [], [], [], [], []
with torch.inference_mode():
    for _ in range(20):
        t0 = time.perf_counter(); qkv(x); ts_qkv.append(time.perf_counter() - t0)
        y = torch.randn(88, 1536)
        t0 = time.perf_counter(); proj(y); ts_proj.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mlp.fc1(y); ts_fc1.append(time.perf_counter() - t0)
        z = torch.randn(88, 4224)
        t0 = time.perf_counter(); mlp.fc2(z); ts_fc2.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); mlp.fc3(y); ts_fc3.append(time.perf_counter() - t0)

def mn(xs): return sum(xs[5:]) / max(1, len(xs[5:])) * 1000
print(f'\nPer-Linear (88x... INT8 quantized):')
print(f'  qkv (1536->4608): {mn(ts_qkv):.3f}ms')
print(f'  proj (1536->1536): {mn(ts_proj):.3f}ms')
print(f'  fc1 (1536->4224): {mn(ts_fc1):.3f}ms')
print(f'  fc3 (1536->4224): {mn(ts_fc3):.3f}ms')
print(f'  fc2 (4224->1536): {mn(ts_fc2):.3f}ms')
print(f'  sum: {(mn(ts_qkv)+mn(ts_proj)+mn(ts_fc1)+mn(ts_fc2)+mn(ts_fc3)):.2f}ms')
print(f'  x42 blocks: {(mn(ts_qkv)+mn(ts_proj)+mn(ts_fc1)+mn(ts_fc2)+mn(ts_fc3))*42:.1f}ms')
