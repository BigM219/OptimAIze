# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Custom INT8 Linear module that bypasses torch.quantization (which is
# deprecated and partially broken in torch >= 2.8) and instead drives our
# own AVX2 / Numba INT8 GEMV kernel directly.
#
# Each Linear is replaced with an Int8Linear that holds:
#   - w_int8     : np.ndarray int8 [out, in]    (per-channel symmetric)
#   - scales     : np.ndarray fp32 [out]        (per-row scale; zero_point=0)
#   - bias       : np.ndarray fp32 [out]
# At decode time (batch=1, seq=1) we dispatch into a vectorized per-channel
# INT8 GEMV. For prefill (longer sequences) we fall back to a dequantized
# FP32 matmul via PyTorch's BLAS — keeps prefill fast and avoids accuracy
# regressions on long inputs.

import logging
import numpy as np
import torch
from torch import nn

from .avx2 import (
    cpp_gemv_int8_per_channel,
    has_cpp_per_channel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-channel symmetric INT8 quantization (weights only).
# ---------------------------------------------------------------------------

def quantize_weight_per_channel_symmetric(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric INT8 quantization.

    Returns (q_w_int8 [out, in], scales [out]). Zero-point is implicitly 0.
    """
    w_np = w.detach().cpu().float().numpy()
    # max-abs per row (output channel)
    abs_max = np.maximum(np.abs(w_np).max(axis=1), 1e-8)
    scales = (abs_max / 127.0).astype(np.float32)
    # Quantize: divide row by its scale, round to int8
    q = np.round(w_np / scales[:, None]).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scales)


# ---------------------------------------------------------------------------
# Per-channel INT8 GEMV kernels (numba fallback). The C++ AVX2 path is
# binding-compatible and added at runtime when the DLL exports it.
# ---------------------------------------------------------------------------

try:
    from numba import jit, prange

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def numba_gemv_int8_per_channel(w_int8, scales, x, bias, out):
        """Per-row symmetric INT8 GEMV: out[i] = scales[i] * sum(w_int8[i,:] * x) + bias[i]."""
        out_features = w_int8.shape[0]
        in_features = w_int8.shape[1]

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
                sum0 += np.float32(w_int8[i, j])     * x[j]
                sum1 += np.float32(w_int8[i, j + 1]) * x[j + 1]
                sum2 += np.float32(w_int8[i, j + 2]) * x[j + 2]
                sum3 += np.float32(w_int8[i, j + 3]) * x[j + 3]
                sum4 += np.float32(w_int8[i, j + 4]) * x[j + 4]
                sum5 += np.float32(w_int8[i, j + 5]) * x[j + 5]
                sum6 += np.float32(w_int8[i, j + 6]) * x[j + 6]
                sum7 += np.float32(w_int8[i, j + 7]) * x[j + 7]
                j += unroll
            sum_val = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7)
            while j < in_features:
                sum_val += np.float32(w_int8[i, j]) * x[j]
                j += 1
            out[i] = scales[i] * sum_val + bias[i]
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def _cpp_gemv_int8_per_channel(w_int8, scales, x, bias, out):
    """Per-channel C++ AVX2 GEMV (added at runtime when DLL exports it)."""
    cpp_gemv_int8_per_channel(w_int8, scales, x, bias, out)


def _has_cpp_per_channel() -> bool:
    return has_cpp_per_channel()


# ---------------------------------------------------------------------------
# Custom Int8 Linear (replaces nn.Linear in place)
# ---------------------------------------------------------------------------

# Resolve which backend to use ONCE at import time. Each forward call would
# otherwise pay a hasattr lookup; 4000 lookups per crop add up.
_USE_CPP = has_cpp_per_channel()
_USE_NUMBA = _HAS_NUMBA and not _USE_CPP


class Int8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with per-channel INT8 weight quant.

    Prefill (long seq) takes the FP32 dequantized fast path via BLAS. Decode
    (batch=1, seq=1) takes the AVX2 / Numba per-channel INT8 GEMV.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Buffers (filled in `pack_from_fp32`)
        self._w_int8: np.ndarray | None = None
        self._scales: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        # Double-buffered output avoids copying while keeping returned tensors stable.
        self._out_bufs: tuple[np.ndarray, np.ndarray] = (
            np.empty(out_features, dtype=np.float32),
            np.empty(out_features, dtype=np.float32),
        )
        self._out_tensors: tuple[torch.Tensor, torch.Tensor] = (
            torch.from_numpy(self._out_bufs[0]),
            torch.from_numpy(self._out_bufs[1]),
        )
        self._out_buf_idx = 0
        # Dequantized FP32 weight for prefill (held only as long as needed)
        self.register_buffer("_w_fp32", torch.empty(0), persistent=False)
        self.has_bias = has_bias

    @classmethod
    def from_linear(cls, lin: nn.Linear, keep_fp32_for_prefill: bool = True) -> "Int8Linear":
        m = cls(lin.in_features, lin.out_features, has_bias=lin.bias is not None)
        q, s = quantize_weight_per_channel_symmetric(lin.weight)
        m._w_int8 = q
        m._scales = s
        if lin.bias is not None:
            m._bias = np.ascontiguousarray(lin.bias.detach().cpu().float().numpy(), dtype=np.float32)
        else:
            m._bias = np.zeros(lin.out_features, dtype=np.float32)
        if keep_fp32_for_prefill:
            # Reconstruct dequantized FP32 weights (used during prefill GEMM).
            # This costs ~same RAM as a fresh nn.Linear; we accept it because
            # prefill is much faster with BLAS GEMM than with our GEMV.
            w_deq = (q.astype(np.float32) * s[:, None])
            m._w_fp32 = torch.from_numpy(w_deq)
        return m

    def _decode_path(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.detach().reshape(-1).contiguous()
        if x_flat.dtype != torch.float32:
            x_flat = x_flat.float()
        x_np = x_flat.numpy()
        self._out_buf_idx ^= 1
        out_np = self._out_bufs[self._out_buf_idx]

        if _USE_CPP:
            _cpp_gemv_int8_per_channel(self._w_int8, self._scales, x_np, self._bias, out_np)
        elif _USE_NUMBA:
            numba_gemv_int8_per_channel(self._w_int8, self._scales, x_np, self._bias, out_np)
        else:
            # Pure numpy fallback
            tmp = (self._w_int8.astype(np.float32) @ x_np) * self._scales + self._bias
            out_np[:] = tmp

        out_t = self._out_tensors[self._out_buf_idx]
        if out_t.dtype != x.dtype:
            out_t = out_t.to(x.dtype)
        if x.ndim == 3:
            return out_t.view(1, 1, self.out_features)
        return out_t.view(1, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast shape check (avoids attribute lookups in hot path)
        sh = x.shape
        nd = x.ndim
        if (nd == 3 and sh[0] == 1 and sh[1] == 1) or (nd == 2 and sh[0] == 1):
            return self._decode_path(x)

        # Prefill: use dequantized FP32 GEMM via PyTorch BLAS
        if self._w_fp32.numel() > 0:
            w = self._w_fp32
        else:
            # Build dequant on the fly (no FP32 cache)
            w = torch.from_numpy(self._w_int8.astype(np.float32) * self._scales[:, None])
        b = torch.from_numpy(self._bias) if self._bias is not None else None
        return torch.nn.functional.linear(x, w, b)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8=True"


# ---------------------------------------------------------------------------
# Module-tree replacement
# ---------------------------------------------------------------------------

def replace_linears_with_int8(
    model: nn.Module,
    skip_names: tuple[str, ...] = (),
    only_names: tuple[str, ...] | None = None,
    keep_fp32_for_prefill: bool = True,
) -> int:
    """Replace every `nn.Linear` in `model` with an `Int8Linear` in place.

    Args:
        model: The root module.
        skip_names: Substring filters — any submodule whose dotted name
            contains one of these is left as FP32.
        only_names: If provided, only modules whose dotted name contains one
            of these strings are quantized; everything else is left FP32.
        keep_fp32_for_prefill: If True, each Int8Linear also caches a
            dequantized FP32 weight to keep prefill BLAS-fast. Costs ~same
            RAM as the original but typically halves prefill latency.

    Returns the number of layers replaced.
    """
    replaced = 0
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip_names):
                continue
            if only_names is not None and not any(s in full for s in only_names):
                continue
            new = Int8Linear.from_linear(child, keep_fp32_for_prefill=keep_fp32_for_prefill)
            setattr(parent, child_name, new)
            replaced += 1
            # Drop the FP32 weight reference for the original child to allow GC
            del child
    logger.info(f"Replaced {replaced} nn.Linear -> Int8Linear (skip={skip_names}, only={only_names})")
    return replaced
