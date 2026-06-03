# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Highly optimized CPU runner for Falcon-OCR, mathematically equivalent, zero FlexAttention/CUDA dependencies.
#
# Performance optimizations:
#   1. Pre-allocated KV cache (eliminates per-step torch.cat reallocation)
#   2. All Linear layers patched with AVX2 / Numba GEMV kernels for decode (batch=1, seq=1)
#   3. Optional dynamic INT8 quantization (4x weight bandwidth reduction)
#   4. Cached output buffers in the linear dispatch (no per-call numpy allocation)
#   5. torch.inference_mode() throughout

import os
import json
import math
import logging
from pathlib import Path
from PIL import Image
import einops as E
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from huggingface_hub import snapshot_download

from .base import BaseVLMBackend
from ..prompts import FALCON_CATEGORY_PROMPTS as CATEGORY_PROMPTS
from ..compute import patch_model_linear_layers, replace_linears_with_int8
from ..runtime_policy import RuntimeMode, build_runtime_policy, extract_model_compute_profile

logger = logging.getLogger(__name__)

# Image parameters
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class _FalconTokenizer:
    """Lightweight Rust-backed tokenizer for Falcon Models."""
    _TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self._source_dir = path
        self._tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

        config = {}
        config_file = os.path.join(path, "tokenizer_config.json")
        if os.path.isfile(config_file):
            config = json.loads(Path(config_file).read_text(encoding="utf-8"))

        stm = {}
        stm_file = os.path.join(path, "special_tokens_map.json")
        if os.path.isfile(stm_file):
            stm = json.loads(Path(stm_file).read_text(encoding="utf-8"))

        self.special_tokens_map = {}
        for k, v in stm.items():
            if isinstance(v, str):
                self.special_tokens_map[k] = v
        for k, v in config.get("model_specific_special_tokens", {}).items():
            if isinstance(v, str):
                self.special_tokens_map[k] = v

        for token_name, token_str in self.special_tokens_map.items():
            setattr(self, token_name, token_str)
            tid = self._tok.token_to_id(token_str)
            setattr(self, token_name + "_id", tid)

        self.eos_token_id = self._tok.token_to_id(
            config.get("eos_token", stm.get("eos_token", "<|end_of_text|>"))
        )
        self.bos_token_id = None
        bos_str = config.get("bos_token") or stm.get("bos_token")
        if bos_str:
            self.bos_token_id = self._tok.token_to_id(bos_str)
        self.bos_id = self.bos_token_id

        self.pad_token_id = self._tok.token_to_id(
            config.get("pad_token", stm.get("pad_token", "<|pad|>"))
        )
        self.padding_side = "left"

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if not isinstance(ids, list):
            ids = list(ids)
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

# ---------------------------------------------------------------------------
# RoPE Position Computations
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.ndim == 3, "Freqs_cis must have shape (B,S,D)"
    freqs_cis = E.rearrange(freqs_cis, "b s d -> b s 1 d")
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_golden_freqs_cis_to_visual_pos(freqs_hFP, pos_BSP) -> torch.Tensor:
    theta_BShF = torch.einsum("bsp,hfp->bshf", pos_BSP.float(), freqs_hFP.float())
    freqs_cis_BShF = torch.polar(torch.ones_like(theta_BShF), theta_BShF)
    return freqs_cis_BShF

def apply_golden_rotary_emb(input_BShd, freqs_cis_BShF) -> torch.Tensor:
    x = input_BShd.float()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = freqs_cis_BShF.real
    sin = freqs_cis_BShF.imag
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.type_as(input_BShd)

def apply_3d_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, freqs_cis_2d: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d)

    xq_out = torch.concat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.concat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out

# ---------------------------------------------------------------------------
# Vision & Token Processing
# ---------------------------------------------------------------------------

class CPUImageProcessor:
    def __init__(self, patch_size=16, merge_size=1, min_pixels=56*56, max_pixels=16*16*4096):
        """Image preprocessor.

        Default `max_pixels` keeps OCR crops close to their natural dimensions.
        Layout detection may use a resized canvas, but OCR should avoid strong
        downscaling because small text quality is more important than token caps.
        """
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def preprocess(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_arr = np.array(image)

        # smart resize
        height, width = img_arr.shape[0], img_arr.shape[1]
        factor = self.patch_size * self.merge_size
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > self.max_pixels:
            beta = np.sqrt((height * width) / self.max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < self.min_pixels:
            beta = np.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height / beta / factor) * factor
            w_bar = math.ceil(width / beta / factor) * factor

        pil_img = Image.fromarray(img_arr)
        pil_img = pil_img.resize((w_bar, h_bar), Image.Resampling.BICUBIC)
        res = np.array(pil_img).astype(np.float32)

        # Normalize
        res = res / 255.0
        mean = np.array(IMAGE_MEAN, dtype=np.float32)[None, None, :]
        std = np.array(IMAGE_STD, dtype=np.float32)[None, None, :]
        res = (res - mean) / std
        return res[None, None, ...] # (1, 1, H, W, C)

def tokenize_inputs_cpu(prompt: str, image_tokens: int, tokenizer) -> np.ndarray:
    img_reg_ids = [
        tokenizer.image_reg_1_token_id,
        tokenizer.image_reg_2_token_id,
        tokenizer.image_reg_3_token_id,
        tokenizer.image_reg_4_token_id,
    ]
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(tokenizer.image_token)]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and tokenizer.bos_id is not None and prompt_chunks[0][0] == tokenizer.bos_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    tokens = [tokenizer.image_token_id] * image_tokens
    image_block = [
        tokenizer.image_cls_token_id,
        *img_reg_ids,
        *tokens,
        tokenizer.end_of_image_token_id,
    ]

    if len(prompt_chunks) >= 2:
        input_ids.extend(prompt_chunks[0][offset:])
        input_ids.extend(image_block)
        input_ids.extend(prompt_chunks[1])
    else:
        input_ids.extend(prompt_chunks[0][offset:])

    return np.array(input_ids, dtype=np.int64)

def get_pos_thw_single_cpu(tokens: np.ndarray, h_v: int, w_v: int, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    S = tokens.shape[0]
    spatial_img_mask = tokens == tokenizer.image_token_id
    no_increase_mask = (
        spatial_img_mask
        | (tokens == tokenizer.image_reg_1_token_id)
        | (tokens == tokenizer.image_reg_2_token_id)
        | (tokens == tokenizer.image_reg_3_token_id)
        | (tokens == tokenizer.image_reg_4_token_id)
        | (tokens == tokenizer.end_of_image_token_id)
    )

    hpos = np.zeros(S, dtype=np.float32)
    wpos = np.zeros(S, dtype=np.float32)

    if spatial_img_mask.any():
        xlim = np.sqrt(w_v / h_v)
        ylim = np.sqrt(h_v / w_v)
        xpos = np.linspace(-xlim, xlim, w_v)
        ypos = np.linspace(-ylim, ylim, h_v)
        wgrid, hgrid = np.meshgrid(xpos, ypos, indexing="xy")

        hpos[spatial_img_mask] = hgrid.flatten()
        wpos[spatial_img_mask] = wgrid.flatten()

    tpos = np.ones(S, dtype=np.float32)
    tpos[no_increase_mask] = 0
    tpos = np.cumsum(tpos) - 1

    return tpos.astype(np.int64), np.stack([hpos, wpos], axis=-1)

# ---------------------------------------------------------------------------
# CPU Model Components
# ---------------------------------------------------------------------------

class ModelArgs:
    max_seq_len: int = 8192
    rope_theta: int = 10000
    dim: int = 768
    n_layers: int = 22
    n_heads: int = 16
    head_dim: int = 64
    n_kv_heads: int = 8
    vocab_size: int = 65536
    ffn_dim: int = 2304
    norm_eps: float = 1e-5
    channel_size: int = 3
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    perception_heads: bool = False

    def update(self, tokenizer):
        self.eos_id = tokenizer.eos_token_id
        self.img_id = tokenizer.image_token_id
        self.img_start_id = tokenizer.start_of_image_token_id
        self.img_end_id = tokenizer.end_of_image_token_id
        self.img_row_sep_id = tokenizer.image_row_sep_token_id
        self.image_cls_token_id = tokenizer.image_cls_token_id
        self.image_reg_1_token_id = tokenizer.image_reg_1_token_id
        self.image_reg_2_token_id = tokenizer.image_reg_2_token_id
        self.image_reg_3_token_id = tokenizer.image_reg_3_token_id
        self.image_reg_4_token_id = tokenizer.image_reg_4_token_id

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, S, H, D = x.shape
    if n_rep == 1:
        return x
    return x.unsqueeze(dim=3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class CPUAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)

        self.wqkv = nn.Linear(args.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.sinks = nn.Parameter(torch.empty((args.n_heads,)))

    def _decode_forward(self, x, freqs_cis, kv_cache):
        """Streamlined attention for autoregressive decode (S=1, no mask).

        Differences from the general `forward`:
          - Skips attention_mask (length-1 query is always all-pass).
          - Skips the `apply_3d_rotary_emb` wrapper: no `freqs_cis_2d` at
            decode time, so we apply RoPE to the first half of head_dim and
            leave the second half untouched (saves chunk/concat overhead).
          - Inlines `kv_cache.insert` to drop a Python call.
        """
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        # qkv shape: (1, 1, q_dim + 2*kv_dim)
        # Split into Q, K, V (views, no copy)
        xq = qkv[..., :self.q_dim].view(1, 1, self.n_heads, self.head_dim)
        xk = qkv[..., self.q_dim:self.q_dim + self.kv_dim].view(1, 1, self.n_kv_heads, self.head_dim)
        xv = qkv[..., self.q_dim + self.kv_dim:].view(1, 1, self.n_kv_heads, self.head_dim)

        xq = F.rms_norm(xq, (self.head_dim,))
        xk = F.rms_norm(xk, (self.head_dim,))

        # Repeat K, V to match n_heads (Group-Query Attention expansion).
        # View-based expansion: no actual data copy.
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).expand(1, 1, self.n_kv_heads, self.n_rep, self.head_dim).reshape(1, 1, self.n_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(1, 1, self.n_kv_heads, self.n_rep, self.head_dim).reshape(1, 1, self.n_heads, self.head_dim)

        # Inline RoPE for the first half of head_dim only (freqs_cis_2d=None at decode).
        # xq, xk shape: (1, 1, n_heads, head_dim).
        half_d = self.head_dim // 2
        # apply_rotary_emb operates on first half; we do it in-place style.
        xq_t = xq[..., :half_d]   # (1, 1, n_heads, half_d) — view
        xk_t = xk[..., :half_d]   # view
        # complex-multiply rotation
        xq_t_c = torch.view_as_complex(xq_t.float().reshape(1, 1, self.n_heads, half_d // 2, 2))
        xk_t_c = torch.view_as_complex(xk_t.float().reshape(1, 1, self.n_heads, half_d // 2, 2))
        fcis = freqs_cis.view(1, 1, 1, half_d // 2)  # broadcast over heads
        xq_rot = torch.view_as_real(xq_t_c * fcis).flatten(3)
        xk_rot = torch.view_as_real(xk_t_c * fcis).flatten(3)
        # Build full Q, K with rotated front + unchanged back
        xq = torch.cat([xq_rot, xq[..., half_d:].float()], dim=-1).to(qkv.dtype)
        xk = torch.cat([xk_rot, xk[..., half_d:].float()], dim=-1).to(qkv.dtype)

        # Inline KV-cache insert
        start = kv_cache.cur_len[self.layer_id]
        end = start + 1
        kv_cache._k[self.layer_id][:, start:end] = xk
        kv_cache._v[self.layer_id][:, start:end] = xv
        kv_cache.cur_len[self.layer_id] = end
        k_past = kv_cache._k[self.layer_id][:, :end]
        v_past = kv_cache._v[self.layer_id][:, :end]

        # Attention math at S_q=1
        # Q: (1, 1, H, D) -> (1, H, 1, D)
        # K, V: (1, S, H, D) -> (1, H, S, D)
        xq2 = xq.transpose(1, 2)
        k2 = k_past.transpose(1, 2)
        v2 = v_past.transpose(1, 2)

        # scores: (1, H, 1, S)
        scores = (xq2 @ k2.transpose(-2, -1)) * self.inv_sqrt_head_dim
        # logsumexp over S for sink scaling
        lse = torch.logsumexp(scores, dim=-1)        # (1, H, 1)
        attn_probs = F.softmax(scores, dim=-1)
        output = attn_probs @ v2                      # (1, H, 1, D)

        # Sink scaling: sigmoid(lse - sinks) acts as gate
        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1))   # (1, H, 1)
        output = output * sink_scale.unsqueeze(-1)

        # (1, H, 1, D) -> (1, 1, H*D)
        output = output.transpose(1, 2).reshape(1, 1, self.n_heads * self.head_dim)
        return self.wo(output)

    def forward(self, x, freqs_cis, freqs_cis_2d, kv_cache, attention_mask=None):
        # Fast path: pure decode (batch=1, seq=1, no 2D RoPE)
        if x.size(1) == 1 and freqs_cis_2d is None:
            return self._decode_forward(x, freqs_cis, kv_cache)

        # General path (prefill + 2D RoPE for image tokens)
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        B, S, _ = x.shape
        xq = xq.view(B, S, -1, self.head_dim)
        xk = xk.view(B, S, -1, self.head_dim)
        xv = xv.view(B, S, -1, self.head_dim)

        xq = F.rms_norm(xq, (xq.size(-1),))
        xk = F.rms_norm(xk, (xk.size(-1),))

        xk = repeat_kv(xk, n_rep=self.n_rep)
        xv = repeat_kv(xv, n_rep=self.n_rep)

        # Apply RoPE
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d)

        # Insert KV into pre-allocated cache (no torch.cat allocation!)
        k_past, v_past = kv_cache.insert(self.layer_id, xk, xv)

        # PyTorch CPU-optimized attention
        xq = xq.transpose(1, 2)         # (B, H, S_q, D)
        k_past = k_past.transpose(1, 2)  # (B, H, S_k, D)
        v_past = v_past.transpose(1, 2)  # (B, H, S_k, D)

        # Standard manual attention with custom Sinks support on CPU
        scores = (xq @ k_past.transpose(-2, -1)) * self.inv_sqrt_head_dim
        if attention_mask is not None:
            # attention_mask shape (B, 1, S_q, S_k)
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        lse = torch.logsumexp(scores, dim=-1) # (B, H, S_q)
        attn_probs = F.softmax(scores, dim=-1)
        output = attn_probs @ v_past          # (B, H, S_q, D)

        # Apply Sink scaling
        sinks_BHS = self.sinks.view(1, -1, 1)         # (1, H, 1)
        sink_scale = torch.sigmoid(lse - sinks_BHS)   # (B, H, S)
        output = output * sink_scale.unsqueeze(-1)

        output = output.transpose(1, 2).flatten(2)    # (B, S, H*D)
        return self.wo(output)


class CPUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        w13_out = self.w13(x)  # (B, S, 2*hidden_dim)

        # CPU Squared-ReLU gate
        gate = w13_out[..., 0::2]
        up = w13_out[..., 1::2]
        fused = F.relu(gate).square() * up
        return self.w2(fused)


class CPUTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = CPUAttention(args, layer_id)
        self.feed_forward = CPUFeedForward(args.dim, args.ffn_dim)

    def forward(self, x, freqs_cis, freqs_cis_2d, kv_cache, attention_mask=None):
        x = x + self.attention(x, freqs_cis, freqs_cis_2d, kv_cache, attention_mask)
        x = x + self.feed_forward(x)
        return x


class FalconPerceptionCPU(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        img_in_dim = args.temporal_patch_size * args.spatial_patch_size**2 * args.channel_size
        self.img_projector = nn.Linear(img_in_dim, args.dim, bias=False)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = CPUTransformerBlock(layer_id, args)

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        rope_dim = args.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, args.max_seq_len, args.rope_theta)
        freqs_cis_golden = torch.empty((args.n_heads, rope_dim // 2, 2), dtype=torch.float)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden, persistent=True)

    def forward(self, tokens, rope_pos_t, rope_pos_hw, pixel_values, img_start, n_tokens, h_v, w_v, kv_cache, attention_mask=None):
        h_BSD = self.tok_embeddings(tokens)

        # Embed image features
        if pixel_values is not None:
            pt, ps = self.args.temporal_patch_size, self.args.spatial_patch_size
            img_tensor = torch.from_numpy(pixel_values).to(h_BSD.device, dtype=h_BSD.dtype)

            # Rearrange patches: (1, T, H, W, C) -> (T*h_v*w_v, pt*ps*ps*C)
            patches = E.rearrange(
                img_tensor[0],
                "(t pt) (h ph) (w pw) c -> (t h w) (pt ph pw c)",
                pt=pt, ph=ps, pw=ps
            )
            grid = patches.reshape(img_tensor.shape[1] // pt, h_v, w_v, patches.shape[-1])
            valid_patches = grid.reshape(-1, patches.shape[-1])
            valid_feats = self.img_projector(valid_patches)

            # Scatter image tokens into token embeddings
            h_BSD[0, img_start : img_start + n_tokens, :] = valid_feats

        freqs_cis = self.freqs_cis[rope_pos_t]
        freqs_cis_golden = None
        if rope_pos_hw is not None:
            freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden, rope_pos_hw)

        for layer in self.layers.values():
            h_BSD = layer(h_BSD, freqs_cis, freqs_cis_golden, kv_cache, attention_mask)

        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return logits_BSV

# ---------------------------------------------------------------------------
# Pre-allocated CPU KV Cache
# ---------------------------------------------------------------------------

class CPUKVCache:
    """Pre-allocated KV cache that grows in-place rather than via torch.cat.

    A naive implementation reallocates the full (S+1)-token cache every decode
    step. For a 22-layer model with ~700-token sequences and 256 output tokens,
    that allocates and copies ~13 GB of float32 data over the full decode loop.
    Pre-allocating once removes that cost entirely.
    """

    def __init__(self, num_layers: int, max_seq_len: int = 8192,
                 n_heads: int = 16, head_dim: int = 64,
                 dtype: torch.dtype = torch.float32, device: str = "cpu"):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Layer-keyed pre-allocated tensors. Shape: (1, max_seq_len, H, D)
        self._k = [
            torch.empty((1, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self._v = [
            torch.empty((1, max_seq_len, n_heads, head_dim), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.cur_len = [0] * num_layers

    def reset(self):
        # Just zero out the indices — no need to clear the buffers.
        for i in range(self.num_layers):
            self.cur_len[i] = 0

    def insert(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        # k, v shape: (1, S_new, H, D)
        s_new = k.shape[1]
        start = self.cur_len[layer_id]
        end = start + s_new
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow on layer {layer_id}: {end} > max_seq_len={self.max_seq_len}"
            )

        # Lazily adapt head count: when the first insert tells us the model
        # repeats KV heads to n_heads, our buffer is already that size.
        if k.shape[2] != self.n_heads or k.shape[3] != self.head_dim:
            # Hot-resize the buffer to match the actual K/V layout (one-time)
            new_h, new_d = k.shape[2], k.shape[3]
            self._k[layer_id] = torch.empty(
                (1, self.max_seq_len, new_h, new_d), dtype=self.dtype, device=self.device
            )
            self._v[layer_id] = torch.empty(
                (1, self.max_seq_len, new_h, new_d), dtype=self.dtype, device=self.device
            )
            self.n_heads = new_h
            self.head_dim = new_d

        self._k[layer_id][:, start:end] = k
        self._v[layer_id][:, start:end] = v
        self.cur_len[layer_id] = end

        return self._k[layer_id][:, :end], self._v[layer_id][:, :end]

# ---------------------------------------------------------------------------
# Falcon OCR CPU Backend Implementation
# ---------------------------------------------------------------------------

class FalconOCRBackend(BaseVLMBackend):
    """CPU-optimized pure-PyTorch Falcon-OCR Backend."""

    def __init__(
        self,
        model_id: str = "tiiuae/Falcon-OCR",
        device: str = "cpu",
        quantize_int8: bool = False,
        max_seq_len: int = 4096,
        max_new_tokens: int = 1024,
        auto_runtime: RuntimeMode = "off",
    ):
        """
        Args:
            model_id: Hugging Face repo id.
            device: Torch device (CPU only is supported).
            quantize_int8: If True, dynamically quantize all Linear layers to
                INT8. Cuts weight memory ~4x and (with the AVX2 INT8 kernel)
                speeds up decode meaningfully on memory-bound CPUs. Slight
                accuracy drop is possible — leave False for max fidelity.
            max_seq_len: Upper bound for the pre-allocated KV cache. A typical
                Falcon-OCR crop needs prompt (~30) + visual tokens (~600) +
                generated (~512) << 4096, so 4k is safe and uses ~80 MB total.
        """
        self.device = torch.device(device)
        self.quantize_int8 = quantize_int8
        self.max_new_tokens = max_new_tokens
        logger.info(f"Downloading/loading Falcon-OCR from HF Hub ({model_id}) on CPU...")

        # Download snap from Hugging Face
        export_dir = Path(snapshot_download(repo_id=model_id, repo_type="model"))
        self.tokenizer = _FalconTokenizer(str(export_dir))

        # Initialize model architecture and args
        self.args = ModelArgs()
        self.args.update(self.tokenizer)
        self.runtime_policy = None
        if auto_runtime != "off":
            config_like = type("FalconPolicyConfig", (), {
                "model_type": "falcon_ocr",
                "num_hidden_layers": self.args.n_layers,
                "hidden_size": self.args.dim,
                "intermediate_size": self.args.ffn_dim,
                "num_attention_heads": self.args.n_heads,
                "num_key_value_heads": self.args.n_kv_heads,
                "vocab_size": self.args.vocab_size,
                "patch_size": self.args.spatial_patch_size,
            })()
            profile = extract_model_compute_profile(
                config_like,
                backend="falcon-ocr",
                model_id=model_id,
                output_style="plain_ocr",
                correctness_status="passed",
            )
            self.runtime_policy = build_runtime_policy(profile, auto_runtime)
        self.model = FalconPerceptionCPU(self.args).eval()

        # Load weights
        state = safetensors_load_file(str(export_dir / "model.safetensors"))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

        # Optional custom per-channel symmetric INT8 quantization. This uses our
        # own kernel-friendly format (NOT PyTorch's quantize_dynamic, which is
        # broken in torch >= 2.8). The lm_head is the biggest layer (~192 MB
        # FP32), so quantizing it gives the largest single-layer win; we
        # quantize the rest of the decoder Linears as well to cut decode-time
        # bandwidth ~4x.
        if quantize_int8:
            logger.info("Applying custom per-channel INT8 quantization to Falcon-OCR Linear layers...")
            replace_linears_with_int8(
                self.model,
                skip_names=("img_projector",),  # keep the tiny image patch projector FP32
                keep_fp32_for_prefill=True,     # prefill stays BLAS-fast
            )

        # Apply AVX2 / Numba GEMV kernels to all remaining FP32 Linear layers.
        # (Int8Linear modules have their own kernel-backed forward.)
        patch_model_linear_layers(self.model)

        self.image_processor = CPUImageProcessor()

        # Pre-allocated KV cache. n_heads dimension matches AFTER repeat_kv.
        self.kv_cache = CPUKVCache(
            num_layers=self.args.n_layers,
            max_seq_len=max_seq_len,
            n_heads=self.args.n_heads,      # after repeat_kv
            head_dim=self.args.head_dim,
            dtype=torch.float32,
            device=device,
        )

    def _build_attention_mask(self, tokens: np.ndarray, img_start: int, img_len: int, device) -> torch.Tensor:
        """Create a CPU-friendly 2D attention mask (causal + bidirectional image)."""
        S = tokens.shape[0]
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))

        # Allow bidirectional attention within visual tokens
        if img_len > 0:
            img_end = img_start + img_len
            mask[img_start:img_end, img_start:img_end] = True

        # Reshape to (1, 1, S, S) for PyTorch scaled_dot_product_attention compatible format
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.inference_mode()
    def generate_ocr(self, image: Image.Image, category: str = "plain") -> str:
        """Run CPU-optimized VLM OCR on a crop image."""
        import time
        self.kv_cache.reset()

        # Build prompt category instruction
        instruction = CATEGORY_PROMPTS.get(category.strip().lower(), CATEGORY_PROMPTS["plain"])
        prompt = f"<|image|>{instruction}\n<|OCR_PLAIN|>"

        # Preprocess crop image
        t_pre = time.perf_counter()
        pixel_values = self.image_processor.preprocess(image)
        h_v = pixel_values.shape[2] // self.args.spatial_patch_size
        w_v = pixel_values.shape[3] // self.args.spatial_patch_size
        image_tokens = h_v * w_v
        t_pre_dt = time.perf_counter() - t_pre

        # Tokenize and format positions
        tokens = tokenize_inputs_cpu(prompt, image_tokens, self.tokenizer)
        tpos, hwpos = get_pos_thw_single_cpu(tokens, h_v, w_v, self.tokenizer)

        # Identify image token indexes
        img_start = int(np.where(tokens == self.tokenizer.image_cls_token_id)[0][0]) + 5 # skip cls + 4 reg tokens
        n_tokens = image_tokens

        # Prepare inputs
        tokens_T = torch.from_numpy(tokens).unsqueeze(0).to(self.device)
        pos_t_T = torch.from_numpy(tpos).unsqueeze(0).to(self.device)
        pos_hw_T = torch.from_numpy(hwpos).unsqueeze(0).to(self.device)

        # Precompute the 2D attention mask for prefill
        attn_mask = self._build_attention_mask(tokens, img_start, n_tokens, self.device)

        # Prefill forward pass (GEMM-bound — PyTorch's BLAS path handles this well)
        t_prefill = time.perf_counter()
        logits = self.model(
            tokens=tokens_T,
            rope_pos_t=pos_t_T,
            rope_pos_hw=pos_hw_T,
            pixel_values=pixel_values,
            img_start=img_start,
            n_tokens=n_tokens,
            h_v=h_v,
            w_v=w_v,
            kv_cache=self.kv_cache,
            attention_mask=attn_mask
        )
        t_prefill_dt = time.perf_counter() - t_prefill

        generated_ids = []
        max_new_tokens = self.max_new_tokens

        stop_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "end_of_query_token_id"):
            stop_ids.append(self.tokenizer.end_of_query_token_id)

        # Decode loop
        curr_token = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
        curr_tpos = int(tpos[-1]) + 1

        t_decode = time.perf_counter()
        for _ in range(max_new_tokens):
            token_id = int(curr_token.item())
            if token_id in stop_ids:
                break
            generated_ids.append(token_id)

            curr_pos_t = torch.tensor([[curr_tpos]], dtype=torch.long, device=self.device)

            logits = self.model(
                tokens=curr_token,
                rope_pos_t=curr_pos_t,
                rope_pos_hw=None,
                pixel_values=None,
                img_start=0,
                n_tokens=0,
                h_v=0,
                w_v=0,
                kv_cache=self.kv_cache,
                attention_mask=None,
            )

            curr_token = torch.argmax(logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            curr_tpos += 1
        t_decode_dt = time.perf_counter() - t_decode
        n_decoded = len(generated_ids)
        ms_per_token = (t_decode_dt / max(1, n_decoded)) * 1000
        logger.info(
            f"[Falcon timing] preproc={t_pre_dt*1000:.1f}ms  "
            f"prefill[{len(tokens)} tok]={t_prefill_dt*1000:.1f}ms  "
            f"decode[{n_decoded} tok]={t_decode_dt*1000:.1f}ms ({ms_per_token:.2f}ms/tok)"
        )

        # Decode output token IDs to clean string
        text = self.tokenizer.decode(generated_ids)
        return text.replace("<|end_of_query|>", "").replace("<|endoftext|>", "").strip()
