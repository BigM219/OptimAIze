from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Check what tensor shapes/strides flow through Int8Linear in the model."""
import torch
# Simulate attention output -> wo flow
B, H, S, D = 1, 16, 1, 64
attn = torch.randn(B, H, S, D)
print(f"attn after Q@K^T@V: shape={attn.shape}, stride={attn.stride()}, contig={attn.is_contiguous()}")

# transpose(1, 2)
t = attn.transpose(1, 2)
print(f"after transpose(1,2): shape={t.shape}, stride={t.stride()}, contig={t.is_contiguous()}")

# flatten(2)
f = t.flatten(2)
print(f"after flatten(2): shape={f.shape}, stride={f.stride()}, contig={f.is_contiguous()}")

# reshape(-1)
r = f.reshape(-1)
print(f"after reshape(-1): shape={r.shape}, stride={r.stride()}, contig={r.is_contiguous()}")
print(f"  same data ptr? {r.data_ptr() == f.data_ptr()}")

# What about the input to wqkv? It comes from rms_norm(x) where x is the residual output of attention.
# rms_norm typically returns contiguous output.
print()
print("From residual + rms_norm:")
x = torch.randn(1, 1, 768)
print(f"  residual: shape={x.shape}, stride={x.stride()}, contig={x.is_contiguous()}")
import torch.nn.functional as F
rn = F.rms_norm(x, (x.size(-1),))
print(f"  after rms_norm: shape={rn.shape}, stride={rn.stride()}, contig={rn.is_contiguous()}")
