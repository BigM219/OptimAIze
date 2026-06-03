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

from optimaize_ocr.backends import get_vlm_backend
from PIL import Image
import logging
import time

logging.basicConfig(level=logging.WARNING)

print('Loading optimized Dots-MOCR (INT8 + AVX2)...')
backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)
backend.max_new_tokens = 64

crops = [
    ('small_text.png', 'outputs/synth_crops/small_text.png'),
    ('small_text.png', 'outputs/synth_crops/small_text.png'),
    ('medium_para.png', 'outputs/synth_crops/medium_para.png'),
    ('large_text.png', 'outputs/synth_crops/large_text.png'),
]

# Warmup on small crop (Numba JIT + first-call overhead)
print('Warming up...')
warm_img = Image.open(crops[0][1]).convert('RGB')
t0 = time.time()
_ = backend.generate_ocr(warm_img, category='text')
print(f'Warmup: {time.time() - t0:.2f}s')

print()
for name, path in crops:
    img = Image.open(path).convert('RGB')
    t0 = time.time()
    result = backend.generate_ocr(img, category='text')
    dt = time.time() - t0
    status = 'PASS' if dt < 5 else 'FAIL'
    print(f'[{status}] {name:<20s} size={img.size} t={dt:.2f}s')
    print(f'         output: {result[:100]!r}')