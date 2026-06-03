from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Debug: identify which patch is breaking dots.mocr by toggling them.

We use selective INT8 quantization (works on 8GB) but progressively disable
each of our optimization patches to find which one breaks the model.

Test order:
1. quant only, no patches → if works, patches are bug
2. quant + RoPE patch → if breaks here, RoPE is bug
3. quant + RoPE + vision_tower fp32 override → ...
4. quant + RoPE + vt + linear_dispatch → ...
5. full stack (current code path) → control

Each test runs the same prompt on the same image and prints first 200 chars
of output. Cache is reused so each test is ~1-2 minutes.
"""
import os
import sys
import logging
import types
from pathlib import Path

# Trigger our package's flash_attn mocking (in backends/dots_mocr/backend.py)
# before any direct transformers import.
import optimaize_ocr.backends.dots_mocr.backend  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

from optimaize_ocr.backends.dots_mocr.patches import (
    prepare_inputs_for_generation_patched,
)
from optimaize_ocr.compute import quantize_model_layer_by_layer

MODEL_ID = "rednote-hilab/dots.mocr"
IMAGE_PATH = "assets/IC-Basic-Document-Control-Template-Example.png"
PROMPT = "Extract the text content from this image."
MAX_TOKENS = 64

print("Loading processor + config...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config._attn_implementation = "eager"
if hasattr(config, "vision_config"):
    config.vision_config.attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"

print("Loading FP32 weights from HF cache...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Apply prepare_inputs_for_generation patch (this is REQUIRED for tf 5.x)
model.prepare_inputs_for_generation = types.MethodType(
    prepare_inputs_for_generation_patched, model
)

# Apply selective INT8 quantization (MLP only)
print("Applying selective INT8 quantization (MLP only)...")
model = quantize_model_layer_by_layer(model, skip_attention=True)
model.eval()

# Load test image
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"Image: {image.size}")


def run_inference(label: str, vision_bf16: bool):
    """Run a single inference and print result."""
    print(f"\n{'='*70}")
    print(f"TEST: {label} (vision_bf16={vision_bf16})")
    print(f"{'='*70}")

    # Apply or remove the vision_tower override
    if hasattr(model, "vision_tower"):
        if vision_bf16 is False:
            # Force fp32
            if not hasattr(model.vision_tower, "_orig_forward_saved"):
                model.vision_tower._orig_forward_saved = model.vision_tower.forward
            _orig = model.vision_tower._orig_forward_saved
            def _vt_fp32(hidden_states, grid_thw, bf16=False):
                return _orig(hidden_states, grid_thw, bf16=False)
            model.vision_tower.forward = _vt_fp32
        else:
            # Restore vendor default (bf16=True)
            if hasattr(model.vision_tower, "_orig_forward_saved"):
                model.vision_tower.forward = model.vision_tower._orig_forward_saved

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=chat_prompt, return_tensors="pt")
    inputs.pop("mm_token_type_ids", None)

    import time
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    dt = time.perf_counter() - t0
    n_tok = len(outputs[0][inputs["input_ids"].shape[-1]:])
    decoded = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    )
    print(f"[{dt:.1f}s, {n_tok} tokens]")
    print(f"OUTPUT: {decoded!r}")


# Test A: vision_tower bf16=True (vendor default)
run_inference("vendor default (bf16=True)", vision_bf16=True)

# Test B: our override (bf16=False)
run_inference("our override (bf16=False)", vision_bf16=False)
