# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Hardware-accelerated math kernels for CPU-optimized inference.

from .avx2 import AVX2_CPP_AVAILABLE, cpp_gemv_int8, cpp_gemv_float32
from .numba_kernels import numba_gemv_int8, numba_gemv_float32
from .rope import (
    apply_rotary_pos_emb_optimized,
    apply_rotary_pos_emb_vision_impl,
)
from .linear_dispatch import patch_model_linear_layers
from .quantization import (
    quantize_model_layer_by_layer,
    quantize_decoder_layers_inplace,
    quantize_submodule_inplace,
)
from .int8_linear import (
    Int8Linear,
    replace_linears_with_int8,
    quantize_weight_per_channel_symmetric,
)
from .fused_mlp import (
    fuse_qwen2_mlp_swiglu,
    fuse_qwen2_attn_qkv,
    fuse_qwen2_attn_rmsnorm,
)

__all__ = [
    "AVX2_CPP_AVAILABLE",
    "cpp_gemv_int8",
    "cpp_gemv_float32",
    "numba_gemv_int8",
    "numba_gemv_float32",
    "apply_rotary_pos_emb_optimized",
    "apply_rotary_pos_emb_vision_impl",
    "patch_model_linear_layers",
    "quantize_model_layer_by_layer",
    "quantize_decoder_layers_inplace",
    "quantize_submodule_inplace",
    "Int8Linear",
    "replace_linears_with_int8",
    "quantize_weight_per_channel_symmetric",
    "fuse_qwen2_mlp_swiglu",
    "fuse_qwen2_attn_qkv",
    "fuse_qwen2_attn_rmsnorm",
]
