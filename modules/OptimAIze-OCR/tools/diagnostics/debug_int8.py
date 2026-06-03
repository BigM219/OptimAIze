from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Investigate what quantize_dynamic actually does in torch 2.8."""
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn

torch.manual_seed(0)
fp = nn.Linear(64, 32)
print("Original type:", type(fp).__module__, type(fp).__name__)
print("Original weight dtype:", fp.weight.dtype)

qmod = torch.quantization.quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8)
print("After quantize_dynamic type:", type(qmod).__module__, type(qmod).__name__)
print("After quantize_dynamic weight dtype:", qmod.weight.dtype if hasattr(qmod, 'weight') else 'no weight attr')
print("Attributes:", [a for a in dir(qmod) if not a.startswith('_')][:20])

# Check if it's the new ao.nn.quantized
print("\nAll module classes torch knows:")
import torch.ao.nn.quantized.dynamic as q
print("Available:", [x for x in dir(q) if not x.startswith('_')])

# Try the newer way: torch.ao.quantization
print("\nTrying torch.ao.quantization.quantize_dynamic...")
qmod2 = torch.ao.quantization.quantize_dynamic(fp, {nn.Linear}, dtype=torch.qint8)
print("ao.quantization quantize_dynamic type:", type(qmod2).__module__, type(qmod2).__name__)

import torch.ao.nn.quantized.dynamic
print("ao.nn.quantized.dynamic.Linear:", torch.ao.nn.quantized.dynamic.Linear)
print("ao.nn.quantized.dynamic.Linear instance?:", isinstance(qmod2, torch.ao.nn.quantized.dynamic.Linear))

# Print structural details of qmod2
if hasattr(qmod2, '_packed_params'):
    w, b = qmod2._packed_params._weight_bias()
    print("Weight qscheme:", w.qscheme())
    print("Weight shape:", w.shape, "dtype:", w.dtype)
    try:
        print("Scale (per-tensor):", w.q_scale())
    except Exception as e:
        print("q_scale err:", e)
    try:
        print("Per-channel scales shape:", w.q_per_channel_scales().shape)
        print("Per-channel zps shape:", w.q_per_channel_zero_points().shape)
    except Exception as e:
        print("per-channel err:", e)
