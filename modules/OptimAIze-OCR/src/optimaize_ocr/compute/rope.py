# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# CPU-optimized Rotary Position Embedding (RoPE) implementations.
# Includes standard text RoPE, vision RoPE, and Numba JIT accelerated variants.

import logging
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# High-Performance CPU Text RoPE (Vector Fused Operation)
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb_optimized(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Highly optimized CPU-specific Rotary Position Embedding (RoPE) implementation.
    
    Bypasses standard multiple tensor allocations (slicing, negation, concatenation)
    by performing fused element-wise mathematical operations with zero-copy views,
    massively saving memory bandwidth and CPU cache misses.
    """
    if position_ids is not None:
        # cos, sin shape: [seq_len, head_dim] -> [B, 1, S, D]
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        # Pre-sliced cos and sin (newer transformers standard)
        # Ensure we match the dimensions by unsqueezing if needed
        if cos.ndim < q.ndim:
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            
    half_d = q.shape[-1] // 2
    
    # zero-copy tensor views (no memory allocation!)
    q_left = q[..., :half_d]
    q_right = q[..., half_d:]
    
    cos_left = cos[..., :half_d]
    sin_left = sin[..., :half_d]
    
    # Fused element-wise multiply-add operations
    q_embed_left = q_left * cos_left - q_right * sin_left
    q_embed_right = q_right * cos_left + q_left * sin_left
    q_embed = torch.cat([q_embed_left, q_embed_right], dim=-1)
    
    if k is not None:
        k_left = k[..., :half_d]
        k_right = k[..., half_d:]
        k_embed_left = k_left * cos_left - k_right * sin_left
        k_embed_right = k_right * cos_left + k_left * sin_left
        k_embed = torch.cat([k_embed_left, k_embed_right], dim=-1)
        return q_embed, k_embed
        
    return q_embed, None


# ---------------------------------------------------------------------------
# High-Performance CPU Vision RoPE (Vector Fused / Numba JIT)
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb_vision_optimized(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Highly optimized CPU-specific Vision Rotary Position Embedding (RoPE) implementation.
    
    Eliminates high-cost un-squeezes, repeats, and multiple rotate_half tensor concatenations
    by utilizing memory-efficient slicing views and zero-copy broadcasting.
    """
    orig_dtype = tensor.dtype
    
    # freqs shape: [S, D_half]
    # tensor shape: [1, S, H, D]
    D = tensor.shape[-1]
    half_d = D // 2
    
    cos_freq = freqs.cos().float()
    sin_freq = freqs.sin().float()
    
    # Broadcast-friendly shapes: [1, S, 1, half_d]
    cos = cos_freq.unsqueeze(0).unsqueeze(2)
    sin = sin_freq.unsqueeze(0).unsqueeze(2)
    
    tensor_float = tensor.float()
    
    # zero-copy tensor views (no memory allocation!)
    q_left = tensor_float[..., :half_d]
    q_right = tensor_float[..., half_d:]
    
    out_left = q_left * cos - q_right * sin
    out_right = q_right * cos + q_left * sin
    
    output = torch.cat([out_left, out_right], dim=-1).to(orig_dtype)
    return output


# ---------------------------------------------------------------------------
# Numba JIT Vision RoPE (fastest path if numba available)
# ---------------------------------------------------------------------------

try:
    import numpy as np
    from numba import jit, prange
    
    @jit(nopython=True, fastmath=True, parallel=True)
    def numba_rope_vision(x, cos, sin, out):
        S = x.shape[0]
        H = x.shape[1]
        D = x.shape[2]
        half_d = D // 2
        
        for s in prange(S):
            for h in prange(H):
                for d in range(half_d):
                    c = cos[s, d]
                    s_val = sin[s, d]
                    
                    val_left = x[s, h, d]
                    val_right = x[s, h, d + half_d]
                    
                    out[s, h, d] = val_left * c - val_right * s_val
                    out[s, h, d + half_d] = val_right * c + val_left * s_val

    def apply_rotary_pos_emb_vision_numba(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        # tensor: [1, S, H, D]
        # freqs: [S, half_d]
        tensor_squeezed = tensor.squeeze(0).float()
        
        cos = freqs.cos().float()
        sin = freqs.sin().float()
        
        x_np = tensor_squeezed.cpu().numpy()
        cos_np = cos.cpu().numpy()
        sin_np = sin.cpu().numpy()
        
        out_np = np.empty_like(x_np)
        numba_rope_vision(x_np, cos_np, sin_np, out_np)
        
        output = torch.from_numpy(out_np).to(tensor.device).unsqueeze(0).to(orig_dtype)
        return output
        
    apply_rotary_pos_emb_vision_impl = apply_rotary_pos_emb_vision_numba
    logger.info("Numba JIT accelerated Vision RoPE loaded successfully!")
except Exception as e:
    apply_rotary_pos_emb_vision_impl = apply_rotary_pos_emb_vision_optimized
    logger.info(f"Numba JIT accelerated Vision RoPE not available. Using optimized PyTorch CPU vector implementation.")


# ---------------------------------------------------------------------------
# Monkeypatch transformers RoPE across multiple causal-LM architectures.
# Each call patches `apply_rotary_pos_emb` in whichever modeling modules are
# importable. We use try/except per-arch so a missing optional model doesn't
# break the others.
# ---------------------------------------------------------------------------

def _patch_arch_rope(module_path: str, attr: str = "apply_rotary_pos_emb") -> bool:
    """Try to monkeypatch `attr` in the given `module_path`. Returns True on success."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if hasattr(mod, attr):
            setattr(mod, attr, apply_rotary_pos_emb_optimized)
            logger.info(f"Patched {module_path}.{attr} with optimized CPU RoPE")
            return True
    except Exception as e:
        logger.debug(f"Skipping RoPE patch for {module_path}: {e}")
    return False


logger.info("Monkeypatching transformers RoPE implementations across causal-LM architectures...")
# Qwen2 family (Dots-MOCR)
_patch_arch_rope("transformers.models.qwen2.modeling_qwen2")
# Mistral family (LightOn-OCR likely)
_patch_arch_rope("transformers.models.mistral.modeling_mistral")
# Llama family (fallback for many derivatives)
_patch_arch_rope("transformers.models.llama.modeling_llama")
