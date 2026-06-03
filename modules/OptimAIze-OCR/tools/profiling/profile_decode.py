from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile decode step time to find what's slow in Int8Linear path."""
import time
import numpy as np
import torch
from torch import nn

from optimaize_ocr.compute.int8_linear import Int8Linear
from optimaize_ocr.compute.linear_dispatch import patch_model_linear_layers

# Simulate: 22 layers x 4 linear (wqkv, wo, w13, w2) + 1 lm_head per decode token
torch.manual_seed(0)

sizes = [
    (1536, 768),  # wqkv
    (768, 1024),  # wo
    (4608, 768),  # w13
    (768, 2304),  # w2
] * 22 + [(65536, 768)]  # lm_head

# Two test models: AVX2-patched FP32 vs Int8Linear
class TestNet(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f, bias=False) for out_f, in_f in sizes])
    def forward(self, xs):
        out = None
        for lin, x in zip(self.layers, xs):
            out = lin(x)
        return out

net_fp32 = TestNet(sizes).eval()
patch_model_linear_layers(net_fp32)

net_int8 = TestNet(sizes).eval()
new_layers = nn.ModuleList()
for lin in net_int8.layers:
    new_layers.append(Int8Linear.from_linear(lin, keep_fp32_for_prefill=False))
net_int8.layers = new_layers

xs = [torch.randn(1, 1, in_f) for _, in_f in sizes]

# Warm both
for _ in range(3):
    net_fp32(xs)
    net_int8(xs)

n_iter = 30
torch.set_num_threads(12)

t = time.perf_counter()
for _ in range(n_iter):
    net_fp32(xs)
dt_fp = (time.perf_counter() - t) / n_iter
print(f"AVX2 FP32-patched nn.Linear:      {dt_fp*1000:7.2f} ms / token")

t = time.perf_counter()
for _ in range(n_iter):
    net_int8(xs)
dt_int8 = (time.perf_counter() - t) / n_iter
print(f"Int8Linear module (AVX2 INT8):    {dt_int8*1000:7.2f} ms / token")

if dt_int8 > dt_fp:
    print(f"\n!!! INT8 is SLOWER by {(dt_int8-dt_fp)*1000:.2f} ms ({(dt_int8/dt_fp):.2f}x). Investigating overhead...")
else:
    print(f"Speedup: {dt_fp/dt_int8:.2f}x")

# Direct C++ kernel calls (no nn.Module overhead)
from optimaize_ocr.compute.avx2 import cpp_gemv_int8_per_channel, cpp_gemv_float32
packed_int8 = []
packed_fp32 = []
xs_np = []
for sz, lin8, linf in zip(sizes, net_int8.layers, net_fp32.layers):
    packed_int8.append((lin8._w_int8, lin8._scales, lin8._bias, np.empty(lin8.out_features, dtype=np.float32)))
    packed_fp32.append((linf._cached_w_float32, linf._cached_bias, np.empty(linf.out_features, dtype=np.float32)))
xs_np = [np.ascontiguousarray(x.numpy().reshape(-1)) for x in xs]

t = time.perf_counter()
for _ in range(n_iter):
    for (w, b, out), x in zip(packed_fp32, xs_np):
        cpp_gemv_float32(w, x, b, out)
dt_fp_raw = (time.perf_counter() - t) / n_iter
print(f"Pure C++ FP32 (no module):        {dt_fp_raw*1000:7.2f} ms / token")

t = time.perf_counter()
for _ in range(n_iter):
    for (w, s, b, out), x in zip(packed_int8, xs_np):
        cpp_gemv_int8_per_channel(w, s, x, b, out)
dt_int8_raw = (time.perf_counter() - t) / n_iter
print(f"Pure C++ INT8 (no module):        {dt_int8_raw*1000:7.2f} ms / token")
print(f"Pure speedup INT8 vs FP32: {dt_fp_raw/dt_int8_raw:.2f}x")
print(f"Module overhead INT8: {(dt_int8 - dt_int8_raw)*1000:.2f} ms / token")
print(f"Module overhead FP32: {(dt_fp - dt_fp_raw)*1000:.2f} ms / token")
