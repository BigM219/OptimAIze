# Copyright (c) 2026 Technology Innovation Institute (TII), UAE.
# Shared dynamic INT8 quantization utilities for HF-style causal LMs.
#
# The standard `torch.quantization.quantize_dynamic(model, {nn.Linear}, ...)`
# call allocates an *entire copy* of the model in memory before swapping
# layers in. For a 2B-parameter model that means a ~8 GB peak RAM spike
# during quantization — fatal on a 16 GB machine.
#
# This utility quantizes the model **in place, one decoder layer at a time**,
# garbage-collecting after each layer. Peak memory stays close to the steady
# state.

import gc
import logging
import torch

logger = logging.getLogger(__name__)


def quantize_decoder_layers_inplace(model, dtype=torch.qint8, log_every: int = 5,
                                    skip_attention: bool = False):
    """Dynamic-quantize every decoder layer one at a time, freeing the FP32
    weights immediately after each one.

    Supports the common HF layout `model.model.layers[i]`. Returns the same
    model (mutated). Skips silently if no such attribute path exists.

    If ``skip_attention`` is True, only the MLP submodule of each decoder
    layer is INT8-quantized; the self-attention block (q/k/v/o projections)
    is kept in its original dtype. This trades RAM for instruction-following
    fidelity — INT8 dynamic quantization on attention weights is the main
    cause of dots.mocr emitting hallucinated/looped outputs on CPU.
    """
    # Common HF causal-LM path: model.model.layers
    layers_owner = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers_owner = model.model
    # Llava/multi-modal style: model.language_model.model.layers
    elif (hasattr(model, "language_model")
          and hasattr(model.language_model, "model")
          and hasattr(model.language_model.model, "layers")):
        layers_owner = model.language_model.model

    if layers_owner is None:
        logger.warning("Could not locate `.model.layers` or `.language_model.model.layers` — skipping decoder quantization.")
        return model

    num_layers = len(layers_owner.layers)
    mode = "MLP-only" if skip_attention else "full"
    logger.info(f"Dynamic-quantizing {num_layers} decoder layers ({mode}, in place, layer-by-layer)...")

    for i in range(num_layers):
        layer = layers_owner.layers[i]
        if skip_attention and hasattr(layer, "mlp"):
            # Quantize only the MLP block; leave self_attn in FP32
            layer.mlp = torch.quantization.quantize_dynamic(
                layer.mlp, {torch.nn.Linear}, dtype=dtype
            )
        else:
            layers_owner.layers[i] = torch.quantization.quantize_dynamic(
                layer, {torch.nn.Linear}, dtype=dtype
            )
        gc.collect()
        if (i + 1) % log_every == 0 or (i + 1) == num_layers:
            logger.info(f"  quantized decoder layer {i + 1}/{num_layers}")

    return model


def quantize_submodule_inplace(model, attr_path: str, dtype=torch.qint8):
    """Quantize a single named submodule (e.g. 'lm_head', 'vision_tower') in place.

    `attr_path` may be dotted (e.g. 'language_model.lm_head'). Silently
    ignores missing paths.
    """
    parts = attr_path.split(".")
    parent = model
    try:
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        leaf = getattr(parent, leaf_name)
    except AttributeError:
        return False

    if leaf is None:
        return False

    # For a bare nn.Linear, quantize directly; for a container (e.g. vision_tower),
    # quantize all its nn.Linear children.
    if isinstance(leaf, torch.nn.Linear):
        setattr(parent, leaf_name, torch.quantization.quantize_dynamic(
            leaf, {torch.nn.Linear}, dtype=dtype
        ))
    else:
        setattr(parent, leaf_name, torch.quantization.quantize_dynamic(
            leaf, {torch.nn.Linear}, dtype=dtype
        ))
    gc.collect()
    logger.info(f"  quantized submodule '{attr_path}'")
    return True


def quantize_model_layer_by_layer(
    model,
    dtype=torch.qint8,
    extra_submodules: tuple[str, ...] = ("lm_head", "vision_tower", "multi_modal_projector"),
    skip_attention: bool = False,
):
    """Apply dynamic INT8 quantization layer-by-layer to a causal LM in place.

    Order of operations (chosen to flatten RAM peaks):
        1. Each decoder layer is quantized + gc'd individually.
        2. The lm_head, vision_tower, and projector are quantized as separate
           submodules afterwards (each one is small relative to a layer).

    The vision_tower IS quantized — for batched-GEMM workloads (vision
    prefill at batch = num_image_patches) PyTorch's dynamic-INT8 FBGEMM
    kernel runs ~5x faster than the FP32 BLAS path on CPU. We measured
    7.5 s → 1.4 s on a 320x60 small crop.

    Both LightOn-OCR (Mistral-like) and Dots-MOCR (Qwen2-like) fit this layout.

    When ``skip_attention=True``, only decoder-layer MLP blocks are
    INT8-quantized (attention stays FP32). Use this for instruction-tuned
    models like dots.mocr where INT8 attention destroys the model's
    ability to follow OCR prompts.
    """
    logger.info("Applying Dynamic INT8 Quantization (layer-by-layer, low-RAM)...")
    quantize_decoder_layers_inplace(model, dtype=dtype, skip_attention=skip_attention)
    for path in extra_submodules:
        # Try the bare path AND the language_model.<path> variant
        if not quantize_submodule_inplace(model, path, dtype=dtype):
            quantize_submodule_inplace(model, f"language_model.{path}", dtype=dtype)
    return model
