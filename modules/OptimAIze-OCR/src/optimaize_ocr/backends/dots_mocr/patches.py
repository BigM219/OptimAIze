# Hugging Face Generation patches for dots.mocr backend.
#
# Transformers 5.x changed ``Qwen2ForCausalLM.prepare_inputs_for_generation``:
# the old ``cache_position`` kwarg was replaced with ``is_first_iteration``,
# and the vendor's dots.mocr ``prepare_inputs_for_generation`` (still on the
# 4.x API) ends up dropping ``pixel_values`` / ``image_grid_thw`` on every
# step, so the LM never sees the image and decodes from a hallucinated SVG
# prior. We replace the method with one that:
#   1. matches the new 5.x base signature, and
#   2. re-injects ``pixel_values`` and ``image_grid_thw`` on the *first*
#      decoding step only — the KV cache covers later steps.
import inspect
import logging

logger = logging.getLogger(__name__)

_patch_call_count = 0  # debug counter


def prepare_inputs_for_generation_patched(
    self,
    input_ids,
    next_sequence_length=None,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    is_first_iteration=None,
    pixel_values=None,
    image_grid_thw=None,
    cache_position=None,
    num_logits_to_keep=None,
    **kwargs,
):
    """transformers-5.x-compatible ``prepare_inputs_for_generation`` that
    keeps ``pixel_values`` + ``image_grid_thw`` flowing on the first step.
    """
    global _patch_call_count
    _patch_call_count += 1
    call_idx = _patch_call_count

    from transformers import Qwen2ForCausalLM

    base_sig = inspect.signature(Qwen2ForCausalLM.prepare_inputs_for_generation)
    base_params = base_sig.parameters

    base_kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
    }
    if "next_sequence_length" in base_params:
        base_kwargs["next_sequence_length"] = next_sequence_length
    if "is_first_iteration" in base_params:
        base_kwargs["is_first_iteration"] = is_first_iteration
    if "cache_position" in base_params:
        base_kwargs["cache_position"] = cache_position
    if "num_logits_to_keep" in base_params:
        base_kwargs["num_logits_to_keep"] = num_logits_to_keep
    base_kwargs.update(kwargs)

    model_inputs = Qwen2ForCausalLM.prepare_inputs_for_generation(
        self,
        input_ids,
        **base_kwargs,
    )

    first_step = False
    if is_first_iteration is not None:
        first_step = bool(is_first_iteration)
    else:
        pos = cache_position if cache_position is not None else model_inputs.get("cache_position")
        if pos is not None:
            try:
                first_step = bool(pos[0] == 0)
            except Exception:
                first_step = past_key_values is None
        else:
            first_step = past_key_values is None

    if first_step:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw

    # Log first 3 calls for diagnostics
    if call_idx <= 3:
        pv_shape = pixel_values.shape if pixel_values is not None else None
        thw_shape = image_grid_thw.shape if image_grid_thw is not None else None
        pv_in_inputs = "pixel_values" in model_inputs
        thw_in_inputs = "image_grid_thw" in model_inputs
        logger.debug(
            f"[patch call={call_idx}] is_first_iteration={is_first_iteration} "
            f"first_step={first_step} "
            f"pixel_values_arg={pv_shape} image_grid_thw_arg={thw_shape} "
            f"pv_in_model_inputs={pv_in_inputs} thw_in_model_inputs={thw_in_inputs}"
        )

    return model_inputs
