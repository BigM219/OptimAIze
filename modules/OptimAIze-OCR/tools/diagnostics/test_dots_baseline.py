from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Run the ORIGINAL DotsMOCRBackend on the same crop to verify accuracy baseline."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from optimaize_ocr.backends import get_vlm_backend
from PIL import Image
import logging
import time

logging.basicConfig(level=logging.INFO)

print('Loading ORIGINAL Dots-MOCR (max_new_tokens=1024)...')
backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=False)

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')
start = time.time()
result = backend.generate_ocr(img, category='text')
dt = time.time() - start

print(f'Time: {dt:.2f}s')
print(f'Result: {result!r}')
