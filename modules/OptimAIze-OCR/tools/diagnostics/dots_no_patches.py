from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Test Dots-MOCR with all CPU optimizations DISABLED.
- No INT8 quantization
- No AVX2 Linear patch
- No Vision RoPE patch
Force eager attention. Goal: confirm baseline accuracy.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import types as _types, importlib.machinery as _mach
def _mk_pkg(name):
    m = _types.ModuleType(name)
    spec = _mach.ModuleSpec(name=name, loader=None, is_package=True)
    spec.submodule_search_locations = []
    m.__spec__ = spec
    m.__path__ = []
    m.__version__ = "2.0.0"
    return m
def _mk_mod(name):
    m = _types.ModuleType(name)
    m.__spec__ = _mach.ModuleSpec(name=name, loader=None)
    return m
sys.modules["flash_attn"] = _mk_pkg("flash_attn")
sys.modules["flash_attn.flash_attn_interface"] = _mk_mod("flash_attn.flash_attn_interface")
sys.modules["flash_attn.modules"] = _mk_pkg("flash_attn.modules")
sys.modules["flash_attn.modules.mha"] = _mk_mod("flash_attn.modules.mha")

import transformers.utils.import_utils as _iu
_orig = _iu._is_package_available
def _patched(pkg_name, return_version=False):
    if pkg_name == "flash_attn":
        return (False, "N/A")
    return _orig(pkg_name, return_version=return_version)
_iu._is_package_available = _patched

import logging, time, types, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from optimaize_ocr.backends.dots_mocr.patches import prepare_inputs_for_generation_patched

logging.basicConfig(level=logging.WARNING)

MODEL_ID = "rednote-hilab/dots.mocr"
print('[1/2] Loading FP32 Dots-MOCR (eager attn, no patches)...')
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config._attn_implementation = "eager"
if hasattr(config, "vision_config"):
    config.vision_config.attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, config=config, torch_dtype=torch.float32,
    trust_remote_code=True, low_cpu_mem_usage=True,
)
model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_patched, model)
model.eval()

# Force vision tower to NOT cast to bfloat16 — matches FP32 weights
_orig_vt_forward = model.vision_tower.forward
def _vt_forward_fp32(hidden_states, grid_thw, bf16=False):
    return _orig_vt_forward(hidden_states, grid_thw, bf16=False)
model.vision_tower.forward = _vt_forward_fp32

# Inspect what attention class the vision blocks actually use
vt = model.vision_tower
print(f'[debug] vision_tower.config.attn_implementation = {vt.config.attn_implementation!r}')
first_block = vt.blocks[0]
print(f'[debug] first vision block attn class = {type(first_block.attn).__name__}')
print(f'[debug] num vision blocks = {len(vt.blocks)}')

src = Image.open('assets/dots_demo1.jpg').convert('RGB')
# Downsize so this completes within reasonable CPU time
W, H = src.size
scale = min(1.0, 800 / max(W, H))
if scale < 1.0:
    src = src.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
img = src
print(f'[debug] image size used: {img.size}')

# img is no longer a single crop — it's the full doc
# (variable name `img` is reused below)
img = src  # full page, not a crop

instruction = "Extract the text content from this image."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
print(f'[debug] prompt = {prompt!r}')
print(f'[debug] prompt repr (first 300 chars) = {prompt[:300]!r}')
inputs = processor(images=img, text=prompt, return_tensors="pt")
inputs.pop("mm_token_type_ids", None)
img_pad_id = processor.tokenizer.convert_tokens_to_ids('<|imgpad|>')
n_pad = (inputs['input_ids'][0] == img_pad_id).sum().item()
print(f'[debug] input_ids.shape={inputs["input_ids"].shape}, n_imgpad={n_pad}, expected={inputs["image_grid_thw"].prod(-1).item()}')
print(f'[debug] image_grid_thw={inputs["image_grid_thw"].tolist()}')
print(f'[debug] config.image_token_id={model.config.image_token_id}, processor imgpad id={img_pad_id}')

# Probe vision output sanity
with torch.inference_mode():
    ve = model.vision_tower(inputs["pixel_values"], inputs["image_grid_thw"])
    print(f'[debug] vision_embeddings shape={tuple(ve.shape)} dtype={ve.dtype} '
          f'mean={ve.float().mean().item():.4f} std={ve.float().std().item():.4f} '
          f'min={ve.float().min().item():.4f} max={ve.float().max().item():.4f} '
          f'has_nan={torch.isnan(ve).any().item()} has_inf={torch.isinf(ve).any().item()}')

    # Same model, with WHITE image - if outputs differ a lot, vision is reading the image
    white = Image.new("RGB", img.size, "white")
    inputs_white = processor(images=white, text=prompt, return_tensors="pt")
    inputs_white.pop("mm_token_type_ids", None)
    ve_w = model.vision_tower(inputs_white["pixel_values"], inputs_white["image_grid_thw"])
    diff = (ve.float().mean(0) - ve_w.float().mean(0)).abs().mean().item()
    print(f'[debug] WHITE vs REAL mean-diff per-dim = {diff:.4f}, '
          f'(white std={ve_w.float().std().item():.4f})')

print('[2/2] Generating max_new_tokens=128 ...')
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
dt = time.time() - t0

input_len = inputs["input_ids"].shape[-1]
decoded = processor.decode(out[0][input_len:], skip_special_tokens=True)
print(f'Time: {dt:.1f}s')
print(f'FP32 (no patches) result: {decoded!r}')
