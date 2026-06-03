from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Thread sweep: find optimal number of CPU threads for Dots-MOCR small crop.

Runs the full OCR (no max_new_tokens cap, EOS-driven) across a range of
thread counts to find the sweet spot. Hyperthreads usually hurt; small
GEMVs sync more than they compute past a point.
"""
import sys, io, os, time, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

parser = argparse.ArgumentParser()
parser.add_argument('--threads', type=str, default='2,4,6,8,10,12,14',
                    help='Comma-separated thread counts to test')
parser.add_argument('--max-new-tokens', type=int, default=0,
                    help='0 = no cap (let EOS stop). >0 = hard cap.')
parser.add_argument('--repeats', type=int, default=2,
                    help='Runs per thread count (after warmup)')
args = parser.parse_args()

# Parse threads BEFORE importing torch so OMP env vars are set first
thread_counts = [int(x) for x in args.threads.split(',')]

# Set first thread count for initial torch import
os.environ['OMP_NUM_THREADS'] = str(thread_counts[0])
os.environ['MKL_NUM_THREADS'] = str(thread_counts[0])

import logging
logging.basicConfig(level=logging.WARNING)

from PIL import Image
import torch
from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization

setup_cpu_optimization(thread_counts[0])
print(f'Loading Dots-MOCR (initial threads={thread_counts[0]})...')
backend = get_vlm_backend('dots-mocr', device='cpu', use_optimized_dots=True)
if args.max_new_tokens > 0:
    backend.max_new_tokens = args.max_new_tokens
else:
    backend.max_new_tokens = 4096  # let EOS decide

img = Image.open('outputs/synth_crops/small_text.png').convert('RGB')

# Warmup
print('Warming up (Numba JIT)...')
_ = backend.generate_ocr(img, category='text')

print(f'\n=== Thread sweep on small crop (max_new_tokens={"EOS" if args.max_new_tokens==0 else args.max_new_tokens}) ===')
print(f'{"threads":>8} | {"avg(s)":>7} | {"min(s)":>7} | {"output preview":<40}')
print('-' * 72)

results = []
for n in thread_counts:
    setup_cpu_optimization(n)
    times = []
    out = ''
    for _ in range(args.repeats):
        t0 = time.time()
        out = backend.generate_ocr(img, category='text')
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    mn = min(times)
    preview = out[:80].replace('\n', ' ')
    print(f'{n:>8} | {avg:>7.2f} | {mn:>7.2f} | {preview!r:<80}')
    results.append((n, avg, mn))

best = min(results, key=lambda r: r[2])
print(f'\nFastest: {best[0]} threads -> {best[2]:.2f}s')
print(f'\nFull output (last run): {out!r}')
