# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Linear layer dispatch: patches nn.Linear and quantized Linear with
# hardware-accelerated GEMV for autoregressive decoding (batch=1, seq=1).
#
# At decode time every Linear in the model is shape (1, 1, D_in) @ W.T -> (1, 1, D_out).
# PyTorch's matmul calls into BLAS which is heavily tuned for GEMM but not for
# tall-skinny GEMV at batch=1. Replacing those calls with a dedicated AVX2/Numba
# GEMV kernel cuts decode latency dramatically — for Falcon-OCR (22 layers, 4
# Linear projections per layer + the 768->65536 lm_head), this is the single
# biggest CPU optimization in the pipeline.

import types
import logging
import numpy as np
import torch

from .avx2 import AVX2_CPP_AVAILABLE, cpp_gemv_int8, cpp_gemv_float32
from .numba_kernels import numba_gemv_int8, numba_gemv_float32

logger = logging.getLogger(__name__)


def _is_gemv_shape(x: torch.Tensor) -> bool:
    """True iff x is (1, 1, D) or (1, D) — the only shapes the GEMV path handles."""
    if x.ndim == 3:
        return x.shape[0] == 1 and x.shape[1] == 1
    if x.ndim == 2:
        return x.shape[0] == 1
    return False


def _x_to_flat_numpy(x: torch.Tensor) -> np.ndarray:
    """Return a contiguous float32 1-D numpy view of x with minimal copying."""
    # .contiguous() is a no-op if x is already contiguous; .reshape is safer than .view.
    flat = x.reshape(-1)
    if flat.dtype != torch.float32:
        flat = flat.float()
    if not flat.is_contiguous():
        flat = flat.contiguous()
    # torch->numpy is zero-copy for CPU float32 contiguous tensors.
    return flat.numpy()


def _wrap_output(out_np: np.ndarray, x: torch.Tensor) -> torch.Tensor:
    """Wrap a 1-D numpy result back into a torch tensor matching x's leading dims.

    The numpy buffer is shared with a per-module pre-allocated array, so we
    must copy here — the caller's next forward() pass would otherwise
    overwrite this output. The copy is fast (a few KB at most for typical
    Linear out_features) compared to the surrounding GEMV.
    """
    out_t = torch.from_numpy(out_np).clone().to(dtype=x.dtype)
    if x.ndim == 3:
        return out_t.unsqueeze(0).unsqueeze(0)
    return out_t.unsqueeze(0)


def _patched_quantized_linear_forward(self, x):
    """Fast CPU GEMV path for quantized INT8 linear layers during autoregressive decoding."""
    if _is_gemv_shape(x):
        x_flat = _x_to_flat_numpy(x)
        out_np = self._out_buf  # pre-allocated, reused across calls

        if AVX2_CPP_AVAILABLE:
            cpp_gemv_int8(
                self._cached_w_int8,
                self._cached_scale,
                self._cached_zero_point,
                x_flat,
                self._cached_bias,
                out_np
            )
        else:
            numba_gemv_int8(
                self._cached_w_int8,
                self._cached_scale,
                self._cached_zero_point,
                x_flat,
                self._cached_bias,
                out_np
            )

        return _wrap_output(out_np, x)
    return self._old_forward(x)


def _patched_float32_linear_forward(self, x):
    """Fast CPU GEMV path for float32 linear layers during autoregressive decoding."""
    if _is_gemv_shape(x):
        x_flat = _x_to_flat_numpy(x)
        out_np = self._out_buf  # pre-allocated, reused across calls

        if AVX2_CPP_AVAILABLE:
            cpp_gemv_float32(
                self._cached_w_float32,
                x_flat,
                self._cached_bias,
                out_np
            )
        else:
            numba_gemv_float32(
                self._cached_w_float32,
                x_flat,
                self._cached_bias,
                out_np
            )

        return _wrap_output(out_np, x)
    return self._old_forward(x)


def patch_model_linear_layers(model):
    """Patch all model Linear layers with AVX2/OpenMP or Numba JIT accelerated GEMV kernels.

    For each quantized dynamic Linear layer, extracts the INT8 weight matrix, scale, and
    zero point into cached contiguous numpy arrays and replaces forward() with the
    accelerated path. For each float32 Linear layer, caches the weight matrix similarly.

    The cached arrays are explicitly forced to C-contiguous float32 (or int8) to guarantee
    the GEMV kernels see well-aligned row-major memory.
    """
    backend_str = "AVX2 C++ (FMA + OpenMP)" if AVX2_CPP_AVAILABLE else "Numba JIT (LLVM + prange)"
    logger.info(f"Patching all model Linear layers with {backend_str} GEMV kernels...")

    num_quantized_patched = 0
    num_float32_patched = 0
    num_skipped = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.dynamic.Linear):
            weight, bias = module._packed_params._weight_bias()
            w_int8 = np.ascontiguousarray(weight.int_repr().cpu().numpy(), dtype=np.int8)
            module._cached_w_int8 = w_int8
            module._cached_scale = float(weight.q_scale())
            module._cached_zero_point = int(weight.q_zero_point())
            if bias is not None:
                module._cached_bias = np.ascontiguousarray(
                    bias.detach().cpu().numpy(), dtype=np.float32
                )
            else:
                module._cached_bias = np.zeros(module.out_features, dtype=np.float32)
            # Pre-allocate output buffer reused across forward calls. Saves
            # ~1 us / call * ~200 calls / decode step = ~0.2 ms / token; small
            # but cumulative across the decode loop.
            module._out_buf = np.empty(module.out_features, dtype=np.float32)

            module._old_forward = module.forward
            module.forward = types.MethodType(_patched_quantized_linear_forward, module)
            num_quantized_patched += 1

        elif isinstance(module, torch.nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                w_fp32 = np.ascontiguousarray(
                    module.weight.detach().cpu().float().numpy(), dtype=np.float32
                )
                module._cached_w_float32 = w_fp32
                if module.bias is not None:
                    module._cached_bias = np.ascontiguousarray(
                        module.bias.detach().cpu().float().numpy(), dtype=np.float32
                    )
                else:
                    module._cached_bias = np.zeros(module.out_features, dtype=np.float32)
                # Pre-allocate output buffer (see comment above).
                module._out_buf = np.empty(module.out_features, dtype=np.float32)

                module._old_forward = module.forward
                module.forward = types.MethodType(_patched_float32_linear_forward, module)
                num_float32_patched += 1
            else:
                num_skipped += 1

    logger.info(
        f"Patched {num_quantized_patched} quantized + {num_float32_patched} float32 linear layers "
        f"(skipped: {num_skipped})."
    )
    return num_quantized_patched + num_float32_patched
