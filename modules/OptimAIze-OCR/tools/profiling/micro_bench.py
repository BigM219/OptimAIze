from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Microbenchmark the GEMV kernels at sizes relevant to Falcon-OCR."""
import time
import numpy as np

# Set OMP threads to use all cores
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")

from optimaize_ocr.compute.avx2 import (
    AVX2_CPP_AVAILABLE,
    cpp_gemv_int8,
    cpp_gemv_float32,
    cpp_gemv_int8_per_channel,
    has_cpp_per_channel,
)
print("AVX2_CPP_AVAILABLE:", AVX2_CPP_AVAILABLE)
print("has_cpp_per_channel:", has_cpp_per_channel())


def bench_one(out_features, in_features, n_iter=200):
    rng = np.random.default_rng(0)
    # FP32
    w_fp = np.ascontiguousarray(rng.standard_normal((out_features, in_features)).astype(np.float32))
    # INT8 per-channel
    abs_max = np.maximum(np.abs(w_fp).max(axis=1), 1e-8)
    scales = (abs_max / 127.0).astype(np.float32)
    w_q = np.round(w_fp / scales[:, None]).clip(-127, 127).astype(np.int8)
    # INT8 single scale
    one_scale = float(np.abs(w_fp).max() / 127.0)
    w_q_pt = np.round(w_fp / one_scale).clip(-127, 127).astype(np.int8)

    x = np.ascontiguousarray(rng.standard_normal(in_features).astype(np.float32))
    bias = np.ascontiguousarray(rng.standard_normal(out_features).astype(np.float32))
    out = np.empty(out_features, dtype=np.float32)

    # Warm
    cpp_gemv_float32(w_fp, x, bias, out)
    cpp_gemv_int8_per_channel(w_q, scales, x, bias, out)
    cpp_gemv_int8(w_q_pt, one_scale, 0, x, bias, out)

    # Time FP32
    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_float32(w_fp, x, bias, out)
    dt_fp = (time.perf_counter() - t) / n_iter

    # Time INT8 per-channel
    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_int8_per_channel(w_q, scales, x, bias, out)
    dt_pc = (time.perf_counter() - t) / n_iter

    # Time INT8 single
    t = time.perf_counter()
    for _ in range(n_iter):
        cpp_gemv_int8(w_q_pt, one_scale, 0, x, bias, out)
    dt_pt = (time.perf_counter() - t) / n_iter

    print(f"  size [{out_features:>6} x {in_features:>5}]   "
          f"FP32={dt_fp*1e6:7.1f}us  INT8-pc={dt_pc*1e6:7.1f}us  INT8-pt={dt_pt*1e6:7.1f}us  "
          f"speedup pc={dt_fp/dt_pc:4.2f}x  pt={dt_fp/dt_pt:4.2f}x")


print("\n=== Falcon-OCR Linear sizes ===")
# Per-layer projections
bench_one(1536, 768)   # wqkv (n_heads=16 + 2*n_kv_heads=8) * head_dim=64
bench_one(768, 1024)   # wo (n_heads=16 * head_dim=64 -> dim=768)
bench_one(4608, 768)   # w13 (2 * ffn_dim=2304)
bench_one(768, 2304)   # w2 (ffn_dim=2304 -> dim=768)

print("\n=== lm_head (768 -> 65536) ===")
bench_one(65536, 768)
