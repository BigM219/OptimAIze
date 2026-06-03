from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""Profile generate_ocr to find decode-step hotspots."""
import cProfile, pstats, io
from PIL import Image
from pathlib import Path
import logging
logging.basicConfig(level=logging.WARNING)

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.pipeline import setup_cpu_optimization

setup_cpu_optimization(12)

backend = get_vlm_backend("falcon-ocr", device="cpu", quantize_int8=True)
img = Image.open(Path(__file__).resolve().parent.parent / "outputs/synth_crops/medium_para.png").convert("RGB")

# Warmup
_ = backend.generate_ocr(img, category="text")

# Profile
pr = cProfile.Profile()
pr.enable()
_ = backend.generate_ocr(img, category="text")
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(40)
print(s.getvalue())
