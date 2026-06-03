from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Helper script to generate a synthetic test document image.
from PIL import Image, ImageDraw

# Create a blank white image
img = Image.new("RGB", (800, 1000), "white")
draw = ImageDraw.Draw(img)

# Draw structured text
draw.text((50, 50), "AN ADVANCED MODULAR CPU OCR TEST PAGE", fill="black")
draw.text((50, 100), "This is a paragraph of standard document text to verify layout detection and OCR backends.", fill="black")
draw.text((50, 130), "The system runs entirely on CPU and supports multiple VLM architectures dynamically.", fill="black")

draw.text((50, 200), "List of features:", fill="black")
draw.text((70, 230), "- High performance on multi-core CPUs", fill="black")
draw.text((70, 260), "- Precision layout analysis using PP-DocLayoutV3", fill="black")
draw.text((70, 290), "- Full Markdown reconstruction from crops", fill="black")

draw.text((50, 350), "Formula example:", fill="black")
draw.text((80, 380), "E = mc^2", fill="black")

# Draw a structured table outline to help the detector recognize it
draw.text((50, 450), "Table example:", fill="black")
# Table border
draw.rectangle([50, 480, 750, 600], outline="black", width=2)
# Row divider
draw.line([50, 520, 750, 520], fill="black", width=2)
# Column divider
draw.line([250, 480, 250, 600], fill="black", width=2)
# Table cells
draw.text((70, 495), "Item Name", fill="black")
draw.text((270, 495), "Quantity", fill="black")
draw.text((70, 550), "Falcon OCR", fill="black")
draw.text((270, 550), "1 unit", fill="black")

img.save("test_document.png")
print("Test image 'test_document.png' generated successfully!")
