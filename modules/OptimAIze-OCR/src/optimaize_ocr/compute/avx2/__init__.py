# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Low-level C++ AVX2 and OpenMP acceleration backend with JIT fallbacks.

from .avx2_backend import (
    AVX2_CPP_AVAILABLE,
    cpp_gemv_int8,
    cpp_gemv_float32,
    cpp_gemv_int8_per_channel,
    cpp_swiglu_gate_up_int8_per_channel,
    cpp_qkv_int8_per_channel,
    cpp_rmsnorm_qkv_int8_per_channel,
    has_cpp_per_channel,
    has_swiglu_int8_per_channel,
    has_qkv_int8_per_channel,
    has_rmsnorm_qkv_int8_per_channel,
)

__all__ = [
    "AVX2_CPP_AVAILABLE",
    "cpp_gemv_int8",
    "cpp_gemv_float32",
    "cpp_gemv_int8_per_channel",
    "cpp_swiglu_gate_up_int8_per_channel",
    "cpp_qkv_int8_per_channel",
    "cpp_rmsnorm_qkv_int8_per_channel",
    "has_cpp_per_channel",
    "has_swiglu_int8_per_channel",
    "has_qkv_int8_per_channel",
    "has_rmsnorm_qkv_int8_per_channel",
]
