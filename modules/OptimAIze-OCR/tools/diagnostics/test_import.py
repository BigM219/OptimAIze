from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Importing package...")
from optimaize_ocr import LayoutAwareOCRPipeline

logger.info("Instantiating pipeline...")
pipeline = LayoutAwareOCRPipeline(model_type="dots-mocr")
logger.info("Successfully instantiated pipeline!")
