from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from optimaize_ocr.backends import get_vlm_backend
from optimaize_ocr.core.layout import LAYOUT_TO_OCR_CATEGORY

IMAGE_PATH = Path("assets/IC-Basic-Document-Control-Template-Example.png")
FALCON_JSON = Path("outputs-falcon/ocr_results.json")
CROPS_DIR = Path("outputs/dots_eval_falcon_crops")
MAX_NEW_TOKENS = 192


def compact(text: str, limit: int = 500) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    items = json.loads(FALCON_JSON.read_text(encoding="utf-8"))
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    backend = get_vlm_backend(
        "dots-mocr",
        device="cpu",
        use_optimized_dots=True,
        quantize_mode="selective",
    )
    backend.max_new_tokens = MAX_NEW_TOKENS

    print(f"image={IMAGE_PATH} size={image.size}")
    print(f"falcon_refs={FALCON_JSON} items={len(items)}")
    print(f"dots max_new_tokens={MAX_NEW_TOKENS}\n")

    for idx, item in enumerate(items):
        category = str(item["category"]).strip().lower()
        ocr_category = LAYOUT_TO_OCR_CATEGORY.get(category, "text") or "text"
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crop = image.crop((x1, y1, x2, y2))
        crop_path = CROPS_DIR / f"crop_{idx:02d}_{category}.png"
        crop.save(crop_path)

        t0 = time.time()
        dots_text = backend.generate_ocr(crop, category=ocr_category)
        dt = time.time() - t0

        print("=" * 90)
        print(f"#{idx} category={category} ocr_category={ocr_category} bbox={[x1, y1, x2, y2]} size={crop.size} t={dt:.2f}s")
        print(f"crop={crop_path}")
        print(f"FALCON: {compact(item.get('text', ''))}")
        print(f"DOTS  : {compact(dots_text)}")


if __name__ == "__main__":
    main()
