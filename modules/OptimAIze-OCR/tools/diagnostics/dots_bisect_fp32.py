from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Make a real-looking flash_attn module so transformers' find_spec returns a valid spec
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
    spec = _mach.ModuleSpec(name=name, loader=None)
    m.__spec__ = spec
    return m

sys.modules["flash_attn"] = _mk_pkg("flash_attn")
sys.modules["flash_attn"].flash_attn_func = lambda *a, **k: None
sys.modules["flash_attn"].flash_attn_varlen_func = lambda *a, **k: None
sys.modules["flash_attn.flash_attn_interface"] = _mk_mod("flash_attn.flash_attn_interface")
sys.modules["flash_attn.modules"] = _mk_pkg("flash_attn.modules")
sys.modules["flash_attn.modules.mha"] = _mk_mod("flash_attn.modules.mha")

# Pretend it's not installed at the metadata level so transformers chooses eager.
# We patch _is_package_available to report not-installed for flash_attn.
import transformers.utils.import_utils as _iu
_orig = _iu._is_package_available
def _patched(pkg_name, return_version=False):
    if pkg_name == "flash_attn":
        return (False, "N/A")
    return _orig(pkg_name, return_version=return_version)
_iu._is_package_available = _patched

import logging, time, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

logging.basicConfig(level=logging.WARNING)

MODEL_ID = "rednote-hilab/dots.mocr"

print("[1/2] Loading FP32 model (no quant, no patches)…")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
try:
    config._attn_implementation = "eager"
except Exception:
    pass
if hasattr(config, "vision_config"):
    try:
        config.vision_config._attn_implementation = "eager"
    except Exception:
        pass

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
import types
from optimaize_ocr.backends.dots_mocr.patches import prepare_inputs_for_generation_patched
model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation_patched, model)
model.eval()

src = Image.open("assets/test_document.png").convert("RGB")
W, H = src.size
crop = src.crop((W // 4, H // 4, W // 4 + 320, H // 4 + 64))
crop.save("outputs/synth_crops/_dbg_real_crop.png")
img = crop

instruction = "Extract the text content from this image."
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=img, text=prompt, return_tensors="pt")
inputs.pop("mm_token_type_ids", None)

print("[2/2] Generating with FP32, max_new_tokens=64 …")
t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
dt = time.time() - t0
input_len = inputs["input_ids"].shape[-1]
decoded = processor.decode(out[0][input_len:], skip_special_tokens=True)
print(f"Time: {dt:.1f}s")
print(f"FP32 result: {decoded!r}")
