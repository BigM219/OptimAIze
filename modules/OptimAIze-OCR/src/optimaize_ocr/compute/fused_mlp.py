# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Fused SwiGLU MLP forward for Qwen2-style decoder layers.
#
# Replaces the standard ``down_proj(silu(gate_proj(x)) * up_proj(x))`` with
# a single ctypes call into our AVX2 kernel that:
#   - reads ``x`` once (instead of twice) across the 1536 -> 8960 GEMV,
#   - applies ``silu(g) * u`` row-by-row before any 8960-element write,
#   - issues a single OpenMP fork/join (vs. two for separate gate/up calls).
#
# Per Qwen2-1.5B layer at decode time this saves ~0.7 ms of the 2.5 ms
# Linear cost and removes a 35 KB intermediate write/read round trip.

import logging
import types
import numpy as np
import torch

from .avx2 import (
    cpp_swiglu_gate_up_int8_per_channel,
    cpp_qkv_int8_per_channel,
    cpp_rmsnorm_qkv_int8_per_channel,
    has_swiglu_int8_per_channel,
    has_qkv_int8_per_channel,
    has_rmsnorm_qkv_int8_per_channel,
)
from .int8_linear import quantize_weight_per_channel_symmetric

logger = logging.getLogger(__name__)


def _patched_qwen2_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward for autoregressive decode (batch=1, seq=1).

    Falls back to the original ``forward`` for prefill or any other shape.
    """
    sh = x.shape
    is_decode = (
        (x.ndim == 3 and sh[0] == 1 and sh[1] == 1)
        or (x.ndim == 2 and sh[0] == 1)
    )
    if not is_decode:
        return self._old_mlp_forward(x)

    x_flat = x.detach().reshape(-1).contiguous().to(torch.float32).numpy()
    out_np = self._fused_out_buf

    cpp_swiglu_gate_up_int8_per_channel(
        self._fused_w_gate_int8, self._fused_scales_gate, self._fused_bias_gate,
        self._fused_w_up_int8,   self._fused_scales_up,   self._fused_bias_up,
        x_flat, out_np,
    )

    intermediate = torch.from_numpy(out_np.copy()).to(dtype=x.dtype)
    if x.ndim == 3:
        intermediate = intermediate.view(1, 1, -1)
    else:
        intermediate = intermediate.view(1, -1)

    # down_proj keeps its existing patched-Linear path
    return self.down_proj(intermediate)


def fuse_qwen2_mlp_swiglu(model) -> int:
    """Patch every Qwen2-style decoder MLP with the fused SwiGLU kernel.

    Looks for ``model.model.layers[i].mlp`` modules that own ``gate_proj``
    and ``up_proj`` Linear-like submodules. Only operates when the AVX2
    SwiGLU kernel is available.

    Returns the number of MLP modules fused.
    """
    if not has_swiglu_int8_per_channel():
        logger.info("Fused SwiGLU kernel unavailable — leaving Qwen2 MLPs untouched.")
        return 0

    layers_owner = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_owner = model.model
    elif (hasattr(model, "language_model")
          and hasattr(model.language_model, "model")
          and hasattr(model.language_model.model, "layers")):
        layers_owner = model.language_model.model

    if layers_owner is None:
        logger.warning("fuse_qwen2_mlp_swiglu: no decoder layer list found")
        return 0

    fused = 0
    for i, layer in enumerate(layers_owner.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate_proj", None)
        up   = getattr(mlp, "up_proj",   None)
        down = getattr(mlp, "down_proj", None)
        if gate is None or up is None or down is None:
            continue

        # Pull the FP32 weights out of whatever Linear flavor we have:
        # nn.Linear (FP32), our Int8Linear, or torch.nn.quantized.dynamic.Linear.
        w_gate_fp32 = _extract_fp32_weight(gate)
        w_up_fp32   = _extract_fp32_weight(up)
        if w_gate_fp32 is None or w_up_fp32 is None:
            continue

        w_gate_int8, scales_gate = quantize_weight_per_channel_symmetric(w_gate_fp32)
        w_up_int8,   scales_up   = quantize_weight_per_channel_symmetric(w_up_fp32)

        bias_gate = _extract_bias(gate)
        bias_up   = _extract_bias(up)

        mlp._fused_w_gate_int8 = w_gate_int8
        mlp._fused_w_up_int8   = w_up_int8
        mlp._fused_scales_gate = scales_gate
        mlp._fused_scales_up   = scales_up
        mlp._fused_bias_gate   = bias_gate
        mlp._fused_bias_up     = bias_up
        mlp._fused_out_buf     = np.empty(w_gate_int8.shape[0], dtype=np.float32)

        mlp._old_mlp_forward = mlp.forward
        mlp.forward = types.MethodType(_patched_qwen2_mlp_forward, mlp)
        fused += 1

    logger.info(f"Fused SwiGLU patched {fused} Qwen2 MLP blocks")
    return fused


def _extract_fp32_weight(linear) -> "torch.Tensor | None":
    """Return the FP32 weight as a torch.Tensor for any of the supported Linear types."""
    # Plain nn.Linear (or our Int8Linear which keeps a dequantized cache)
    if hasattr(linear, "weight") and isinstance(linear.weight, torch.Tensor) and linear.weight.numel() > 0:
        return linear.weight.detach().cpu().float()
    # Our Int8Linear (no weight attr; reconstruct from quantized buffers)
    if hasattr(linear, "_w_int8") and linear._w_int8 is not None and linear._scales is not None:
        deq = linear._w_int8.astype(np.float32) * linear._scales[:, None]
        return torch.from_numpy(deq)
    # torch.nn.quantized.dynamic.Linear: weights stored in _packed_params
    if hasattr(linear, "_packed_params"):
        try:
            w_q, _ = linear._packed_params._weight_bias()
            return torch.dequantize(w_q).detach().cpu().float()
        except Exception:
            return None
    return None


def _extract_bias(linear) -> "np.ndarray | None":
    """Return the bias as a contiguous float32 numpy array, or None."""
    if hasattr(linear, "bias") and isinstance(linear.bias, torch.Tensor):
        return np.ascontiguousarray(linear.bias.detach().cpu().float().numpy(), dtype=np.float32)
    if hasattr(linear, "_bias") and linear._bias is not None:
        return np.ascontiguousarray(linear._bias, dtype=np.float32)
    if hasattr(linear, "_packed_params"):
        try:
            _, b_q = linear._packed_params._weight_bias()
            if b_q is None:
                return None
            return np.ascontiguousarray(b_q.detach().cpu().float().numpy(), dtype=np.float32)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Fused QKV: chain q_proj -> compute Q,K,V at once, cache K and V; subsequent
# k_proj(x) and v_proj(x) calls return the cached results.
# ---------------------------------------------------------------------------

def _patched_q_proj_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute Q,K,V together; cache K,V on the sibling modules for reuse.

    When ``self._fused_norm_gamma`` is populated we additionally fold the
    upstream RMSNorm into this kernel — the LayerNorm module that fed us
    is patched to identity, so ``x`` here is the *pre-norm* hidden state.
    """
    sh = x.shape
    is_decode = (
        (x.ndim == 3 and sh[0] == 1 and sh[1] == 1)
        or (x.ndim == 2 and sh[0] == 1)
    )
    # Vendor's prefill / unexpected shape -> fall through to original.
    # We also clear any stale K/V cache so a subsequent k_proj/v_proj call
    # on this prefill tensor does NOT mistakenly read the decode buffer
    # via Python id() reuse.
    if not is_decode or self._fused_x_id_match is None:
        self._fused_k_proj_ref._fused_cached_xid = None
        self._fused_v_proj_ref._fused_cached_xid = None
        return self._old_q_forward(x)

    x_flat = x.detach().reshape(-1).contiguous().to(torch.float32).numpy()
    out_q = self._fused_q_buf
    out_k = self._fused_k_buf
    out_v = self._fused_v_buf

    if self._fused_norm_gamma is not None:
        cpp_rmsnorm_qkv_int8_per_channel(
            x_flat, self._fused_norm_gamma, self._fused_norm_eps, self._fused_norm_buf,
            self._fused_w_q_int8, self._fused_scales_q, self._fused_bias_q,
            self._fused_w_k_int8, self._fused_scales_k, self._fused_bias_k,
            self._fused_w_v_int8, self._fused_scales_v, self._fused_bias_v,
            out_q, out_k, out_v,
        )
    else:
        cpp_qkv_int8_per_channel(
            self._fused_w_q_int8, self._fused_scales_q, self._fused_bias_q,
            self._fused_w_k_int8, self._fused_scales_k, self._fused_bias_k,
            self._fused_w_v_int8, self._fused_scales_v, self._fused_bias_v,
            x_flat, out_q, out_k, out_v,
        )

    # Stash K and V on the sibling modules so their forward calls can grab
    # them. Sentinel = (id(x), shape) so prefill calls (which reuse Python
    # ids after gc) cannot accidentally hit the decode cache.
    sentinel = (id(x), tuple(sh))
    self._fused_k_proj_ref._fused_cached_xid = sentinel
    self._fused_v_proj_ref._fused_cached_xid = sentinel

    out_q_t = torch.from_numpy(out_q.copy()).to(dtype=x.dtype)
    if x.ndim == 3:
        return out_q_t.view(1, 1, -1)
    return out_q_t.view(1, -1)


def _patched_k_proj_forward(self, x: torch.Tensor) -> torch.Tensor:
    sentinel = (id(x), tuple(x.shape))
    if getattr(self, "_fused_cached_xid", None) == sentinel:
        out_k = self._fused_k_buf_ref
        out_t = torch.from_numpy(out_k.copy()).to(dtype=x.dtype)
        if x.ndim == 3:
            return out_t.view(1, 1, -1)
        return out_t.view(1, -1)
    return self._old_k_forward(x)


def _patched_v_proj_forward(self, x: torch.Tensor) -> torch.Tensor:
    sentinel = (id(x), tuple(x.shape))
    if getattr(self, "_fused_cached_xid", None) == sentinel:
        out_v = self._fused_v_buf_ref
        out_t = torch.from_numpy(out_v.copy()).to(dtype=x.dtype)
        if x.ndim == 3:
            return out_t.view(1, 1, -1)
        return out_t.view(1, -1)
    return self._old_v_forward(x)


def fuse_qwen2_attn_qkv(model) -> int:
    """Patch every Qwen2-style decoder attention with fused QKV projection.

    Replaces three independent ctypes calls per layer (q_proj/k_proj/v_proj)
    with a single fused call that reads ``x`` once. Falls back to the
    original ``forward`` when shapes don't match the decode hot path.

    Returns the number of attention modules fused.
    """
    if not has_qkv_int8_per_channel():
        logger.info("Fused QKV kernel unavailable — leaving Qwen2 attention untouched.")
        return 0

    layers_owner = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_owner = model.model
    elif (hasattr(model, "language_model")
          and hasattr(model.language_model, "model")
          and hasattr(model.language_model.model, "layers")):
        layers_owner = model.language_model.model

    if layers_owner is None:
        return 0

    fused = 0
    for layer in layers_owner.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        q = getattr(attn, "q_proj", None)
        k = getattr(attn, "k_proj", None)
        v = getattr(attn, "v_proj", None)
        if q is None or k is None or v is None:
            continue

        w_q_fp32 = _extract_fp32_weight(q)
        w_k_fp32 = _extract_fp32_weight(k)
        w_v_fp32 = _extract_fp32_weight(v)
        if w_q_fp32 is None or w_k_fp32 is None or w_v_fp32 is None:
            continue

        w_q_int8, scales_q = quantize_weight_per_channel_symmetric(w_q_fp32)
        w_k_int8, scales_k = quantize_weight_per_channel_symmetric(w_k_fp32)
        w_v_int8, scales_v = quantize_weight_per_channel_symmetric(w_v_fp32)

        bias_q = _extract_bias(q)
        bias_k = _extract_bias(k)
        bias_v = _extract_bias(v)

        q_out = w_q_int8.shape[0]
        kv_out = w_k_int8.shape[0]

        # Buffers shared between the q / k / v patched forwards. Allocated
        # once per layer and reused on every decode step.
        q_buf = np.empty(q_out, dtype=np.float32)
        k_buf = np.empty(kv_out, dtype=np.float32)
        v_buf = np.empty(kv_out, dtype=np.float32)

        # Cache everything on q_proj so its forward has all the inputs and
        # buffers it needs without an extra dict lookup.
        q._fused_w_q_int8 = w_q_int8
        q._fused_scales_q = scales_q
        q._fused_bias_q   = bias_q
        q._fused_w_k_int8 = w_k_int8
        q._fused_scales_k = scales_k
        q._fused_bias_k   = bias_k
        q._fused_w_v_int8 = w_v_int8
        q._fused_scales_v = scales_v
        q._fused_bias_v   = bias_v
        q._fused_q_buf    = q_buf
        q._fused_k_buf    = k_buf
        q._fused_v_buf    = v_buf
        q._fused_k_proj_ref = k
        q._fused_v_proj_ref = v
        q._fused_x_id_match = True
        # RMSNorm fold disabled by default — call ``fuse_qwen2_attn_rmsnorm``
        # afterwards to wire it in.
        q._fused_norm_gamma = None
        q._fused_norm_eps   = 1e-6
        q._fused_norm_buf   = None

        k._fused_k_buf_ref = k_buf
        v._fused_v_buf_ref = v_buf

        q._old_q_forward = q.forward
        k._old_k_forward = k.forward
        v._old_v_forward = v.forward
        q.forward = types.MethodType(_patched_q_proj_forward, q)
        k.forward = types.MethodType(_patched_k_proj_forward, k)
        v.forward = types.MethodType(_patched_v_proj_forward, v)
        fused += 1

    logger.info(f"Fused QKV patched {fused} Qwen2 attention blocks")
    return fused


# ---------------------------------------------------------------------------
# Fold input_layernorm into the QKV kernel: identity-out the norm module so
# q_proj receives the *pre-norm* hidden state, then run the fused
# RMSNorm+QKV kernel inside ``_patched_q_proj_forward``. Saves a full read
# pass over the 1536-wide tensor + a tensor allocation + the Python-side
# ``F.rms_norm`` dispatch on every decode step.
# ---------------------------------------------------------------------------

def _identity_layernorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Pass-through ``forward`` used after the layernorm has been folded."""
    sh = x.shape
    is_decode = (
        (x.ndim == 3 and sh[0] == 1 and sh[1] == 1)
        or (x.ndim == 2 and sh[0] == 1)
    )
    if is_decode:
        return x
    return self._old_norm_forward(x)


def fuse_qwen2_attn_rmsnorm(model) -> int:
    """Fold each ``input_layernorm`` into the fused QKV kernel.

    Must be called *after* ``fuse_qwen2_attn_qkv``. Looks for layers whose
    ``self_attn.q_proj`` already has the fused-QKV state (set up by
    ``fuse_qwen2_attn_qkv``) AND whose ``input_layernorm`` is a Qwen2-style
    RMSNorm with a single ``weight`` parameter and ``variance_epsilon``.

    Returns the number of layers folded.
    """
    if not has_rmsnorm_qkv_int8_per_channel():
        logger.info("Fused RMSNorm+QKV kernel unavailable — skipping fold.")
        return 0

    layers_owner = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_owner = model.model
    elif (hasattr(model, "language_model")
          and hasattr(model.language_model, "model")
          and hasattr(model.language_model.model, "layers")):
        layers_owner = model.language_model.model

    if layers_owner is None:
        return 0

    folded = 0
    for layer in layers_owner.layers:
        ln = getattr(layer, "input_layernorm", None)
        attn = getattr(layer, "self_attn", None)
        if ln is None or attn is None:
            continue
        q = getattr(attn, "q_proj", None)
        if q is None or not hasattr(q, "_fused_w_q_int8"):
            continue
        if not hasattr(ln, "weight") or not hasattr(ln, "variance_epsilon"):
            continue

        gamma = np.ascontiguousarray(ln.weight.detach().cpu().float().numpy(), dtype=np.float32)
        in_features = gamma.shape[0]

        q._fused_norm_gamma = gamma
        q._fused_norm_eps = float(ln.variance_epsilon)
        q._fused_norm_buf = np.empty(in_features, dtype=np.float32)

        ln._old_norm_forward = ln.forward
        ln.forward = types.MethodType(_identity_layernorm_forward, ln)
        folded += 1

    logger.info(f"Folded input_layernorm -> RMSNorm+QKV in {folded} attention blocks")
    return folded
