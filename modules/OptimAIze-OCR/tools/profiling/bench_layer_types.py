from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile each Linear layer type in isolation."""
import time
import numpy as np
import torch

# Import AVX2 kernels
from optimaize_ocr.compute.avx2.avx2_backend import cpp_gemv_int8_per_channel, cpp_gemv_float32

# Layer sizes in Falcon-OCR (out_features, in_features)
layers = {
    "wqkv": (1536, 768),     # QKV projection
    "wo": (768, 1024),       # Output projection
    "w13": (4608, 768),      # FFN gate + up
    "w2": (768, 2304),       # FFN down
    "lm_head": (65536, 768), # Vocab projection
}

# Quantize weight per-channel
def quantize(w):
    w_np = w.detach().cpu().float().numpy()
    abs_max = np.maximum(np.abs(w_np).max(axis=1), 1e-8)
    scales = (abs_max / 127.0).astype(np.float32)
    q = np.round(w_np / scales[:, None]).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scales)

# Prepare data
data = {}
for name, (out_f, in_f) in layers.items():
    w = torch.randn(out_f, in_f)
    q, s = quantize(w)
    b = np.zeros(out_f, dtype=np.float32)
    x = np.random.randn(in_f).astype(np.float32)
    out_fp32 = np.empty(out_f, dtype=np.float32)
    out_int8 = np.empty(out_f, dtype=np.float32)
    data[name] = (w.numpy(), q, s, b, x, out_fp32, out_int8, in_f, out_f)

# Warmup
print("Warmup...")
for name, (w, q, s, b, x, out_fp32, out_int8, in_f, out_f) in data.items():
    cpp_gemv_float32(w, x, b, out_fp32)
    cpp_gemv_int8_per_channel(q, s, x, b, out_int8)
print("Done.\n")

# Benchmark (use fewer iterations)
n_iter = 100
print(f"Benchmarking {n_iter} iterations per layer...")
print(f"{'Layer':<12} {'FP32 (us)':<12} {'INT8 (us)':<12} {'Speedup':<10} {'FLOPs':<12}")
print("-" * 70)

for name, (w, q, s, b, x, out_fp32, out_int8, in_f, out_f) in data.items():
    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_float32(w, x, b, out_fp32)
    t_fp32 = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_int8_per_channel(q, s, x, b, out_int8)
    t_int8 = time.perf_counter() - t

    flops = 2 * out_f * in_f  # multiply-add
    print(f"{name:<12} {t_fp32/n_iter*1e6:8.2f}     {t_int8/n_iter*1e6:8.2f}     {t_fp32/t_int8:8.2f}x   {flops:12,}")

# Calculate per-token cost with 22 layers
print("\nPer-token cost breakdown (22 layers):")
per_token_fp32 = 0
per_token_int8 = 0
for name, (_, q, s, b, x, out_fp32, out_int8, in_f, out_f) in data.items():
    if name == "lm_head":
        count = 1
    elif name == "wqkv":
        count = 22
    else:
        count = 22  # wo, w13, w2 each appear 22 times

    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_float32(w, x, b, out_fp32)
    t_fp32 = (time.perf_counter() - t) / n_iter

    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_int8_per_channel(q, s, x, b, out_int8)
    t_int8 = (time.perf_counter() - t) / n_iter

    per_token_fp32 += t_fp32 * count
    per_token_int8 += t_int8 * count

print(f"  FP32: {per_token_fp32*1000:7.2f} ms/token")
print(f"  INT8: {per_token_int8*1000:7.2f} ms/token")
print(f"  Speedup: {per_token_fp32/per_token_int8:.2f}x")