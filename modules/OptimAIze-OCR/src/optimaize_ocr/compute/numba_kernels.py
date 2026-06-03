# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Numba JIT-compiled GEMV kernels with multi-threaded parallelism and fastmath.
#
# Each row is processed by an independent OpenMP-style thread via prange.
# Inside a row we use 8 independent accumulators (8-way unroll) to break the
# FMA dependency chain — Numba/LLVM auto-vectorizes the inner loop into AVX2
# FMA instructions when -mavx2 is available, matching the C++ AVX2 path.

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def numba_gemv_int8(w_int8, scale, zero_point, x, bias, out):
    """Numba JIT INT8 quantized GEMV with 8-way loop unrolling.

    Args:
        w_int8: [out_features, in_features] int8 weight matrix.
        scale: float quantization scale.
        zero_point: int quantization zero point.
        x: [in_features] float32 input vector.
        bias: [out_features] float32 bias vector.
        out: [out_features] float32 output vector (written in-place).
    """
    out_features = w_int8.shape[0]
    in_features = w_int8.shape[1]
    zp = np.float32(zero_point)

    unroll = 8
    limit = (in_features // unroll) * unroll

    for i in prange(out_features):
        # 8 independent accumulators to fully saturate FMA pipeline
        sum0 = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)
        sum3 = np.float32(0.0)
        sum4 = np.float32(0.0)
        sum5 = np.float32(0.0)
        sum6 = np.float32(0.0)
        sum7 = np.float32(0.0)

        j = 0
        while j < limit:
            sum0 += (np.float32(w_int8[i, j])     - zp) * x[j]
            sum1 += (np.float32(w_int8[i, j + 1]) - zp) * x[j + 1]
            sum2 += (np.float32(w_int8[i, j + 2]) - zp) * x[j + 2]
            sum3 += (np.float32(w_int8[i, j + 3]) - zp) * x[j + 3]
            sum4 += (np.float32(w_int8[i, j + 4]) - zp) * x[j + 4]
            sum5 += (np.float32(w_int8[i, j + 5]) - zp) * x[j + 5]
            sum6 += (np.float32(w_int8[i, j + 6]) - zp) * x[j + 6]
            sum7 += (np.float32(w_int8[i, j + 7]) - zp) * x[j + 7]
            j += unroll

        sum_val = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7)
        while j < in_features:
            sum_val += (np.float32(w_int8[i, j]) - zp) * x[j]
            j += 1

        out[i] = scale * sum_val + bias[i]


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def numba_gemv_float32(w_float32, x, bias, out):
    """Numba JIT float32 GEMV with 8-way loop unrolling.

    Args:
        w_float32: [out_features, in_features] float32 weight matrix.
        x: [in_features] float32 input vector.
        bias: [out_features] float32 bias vector.
        out: [out_features] float32 output vector (written in-place).
    """
    out_features = w_float32.shape[0]
    in_features = w_float32.shape[1]

    unroll = 8
    limit = (in_features // unroll) * unroll

    for i in prange(out_features):
        sum0 = np.float32(0.0)
        sum1 = np.float32(0.0)
        sum2 = np.float32(0.0)
        sum3 = np.float32(0.0)
        sum4 = np.float32(0.0)
        sum5 = np.float32(0.0)
        sum6 = np.float32(0.0)
        sum7 = np.float32(0.0)

        j = 0
        while j < limit:
            sum0 += w_float32[i, j]     * x[j]
            sum1 += w_float32[i, j + 1] * x[j + 1]
            sum2 += w_float32[i, j + 2] * x[j + 2]
            sum3 += w_float32[i, j + 3] * x[j + 3]
            sum4 += w_float32[i, j + 4] * x[j + 4]
            sum5 += w_float32[i, j + 5] * x[j + 5]
            sum6 += w_float32[i, j + 6] * x[j + 6]
            sum7 += w_float32[i, j + 7] * x[j + 7]
            j += unroll

        sum_val = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7)
        while j < in_features:
            sum_val += w_float32[i, j] * x[j]
            j += 1

        out[i] = sum_val + bias[i]
