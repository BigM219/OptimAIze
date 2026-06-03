# Prompt mappings for different categories of document segments

# Prompts for LightOn-OCR and Dots-MOCR
DEFAULT_CATEGORY_PROMPTS = {
    "plain": "Transcribe this document into clean Markdown.",
    "formula": "Extract the formula from this image.",
    "table": "Transcribe this table into markdown format.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption from this image.",
    "footnote": "Extract the footnote from this image.",
    "list-item": "Extract the list-item from this image.",
    "page-footer": "Extract the page-footer from this image.",
    "page-header": "Extract the page-header from this image.",
    "section-header": "Extract the section-header from this image.",
    "title": "Extract the title from this image.",
}

# Prompts specific to Falcon-OCR
FALCON_CATEGORY_PROMPTS = {
    "plain": "Extract the text content from this image.",
    "formula": "Extract the formula content from this image.",
    "table": "Extract the table content from this image.",
    "text": "Extract the text content from this image.",
    "caption": "Extract the caption content from this image.",
    "footnote": "Extract the footnote content from this image.",
    "list-item": "Extract the list-item content from this image.",
    "page-footer": "Extract the page-footer content from this image.",
    "page-header": "Extract the page-header content from this image.",
    "section-header": "Extract the section-header content from this image.",
    "title": "Extract the title content from this image.",
}

# Dots-MOCR official prompts (rednote-hilab/dots.mocr).
# These are the *exact* training-distribution strings from the vendor's
# ``dict_promptmode_to_prompt``.  Using anything else makes the model
# emit Chinese ``<think>`` traces, SVG, or repeated reasoning loops.
#
# The model is fine-tuned on FULL PAGES with ``prompt_layout_all_en``;
# when we feed it pre-cropped layout elements (one block at a time) we
# must use the per-task prompts below — otherwise it tries to re-detect
# layout inside the crop and dumps a useless bbox JSON list.
DOTS_MOCR_PROMPT_OCR = (
    "Extract the text content from this image."
)
DOTS_MOCR_PROMPT_TABLE_HTML = (
    "Convert the table in this image to HTML."
)
DOTS_MOCR_PROMPT_FORMULA_LATEX = (
    "Convert the formula in this image to LaTeX."
)
# SVG mode — vendor's ``prompt_image_to_svg``. Width/height placeholders MUST
# be replaced with the actual image dimensions before sending to the model;
# the viewBox grounds output coordinates in pixel space, which significantly
# improves character-level transcription accuracy versus plain ``prompt_ocr``.
DOTS_MOCR_PROMPT_IMAGE_TO_SVG = (
    'Please generate the SVG code based on the image.'
    'viewBox="0 0 {width} {height}"'
)

# Layout-only mode — emits bbox + category, NO text.
DOTS_MOCR_PROMPT_LAYOUT_ONLY_EN = (
    "Please output the layout information from this PDF image, including "
    "each layout's bbox and its category. The bbox should be in the format "
    "[x1, y1, x2, y2]. The layout categories for the PDF document include "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', "
    "'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. "
    "Do not output the corresponding text. The layout result should be in "
    "JSON format."
)

# Grounding OCR — extract text inside a given bbox. The bbox string must
# be appended (e.g. ``prompt + str([x1, y1, x2, y2])``) before sending.
DOTS_MOCR_PROMPT_GROUNDING_OCR = (
    "Extract text from the given bounding box on the image (format: "
    "[x1, y1, x2, y2]).\nBounding Box:\n"
)

# Scene-text spotting — for natural images, returns coord-based output:
# ``(x1,y1),(x2,y2),(x3,y3),(x4,y4) TEXT`` repeated per detection.
DOTS_MOCR_PROMPT_SCENE_SPOTTING = (
    "Detect and recognize the text in the image."
)

# Web-page parsing — JSON layout for screenshots of web pages.
DOTS_MOCR_PROMPT_WEB_PARSING = (
    "Parsing the layout info of this webpage image with format json:\n"
)

# Free-form QA — vendor sends a single space; user must supply
# ``custom_prompt`` for actual queries. Useful for debugging the model.
DOTS_MOCR_PROMPT_GENERAL = " "
DOTS_MOCR_PROMPT_LAYOUT_ALL_EN = (
    "Please output the layout information from the PDF image, including each "
    "layout element's bbox, its category, and the corresponding text content "
    "within the bbox.\n"
    "\n"
    "1. Bbox format: [x1, y1, x2, y2]\n"
    "\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', "
    "'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', "
    "'Section-header', 'Table', 'Text', 'Title'].\n"
    "\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n"
    "\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n"
    "\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)

DOTS_MOCR_CATEGORY_PROMPTS = {
    # ``plain`` is used for the full-page fallback path (no layout detector
    # results).  Only there does the LAYOUT prompt make sense.
    "plain": DOTS_MOCR_PROMPT_LAYOUT_ALL_EN,
    "layout": DOTS_MOCR_PROMPT_LAYOUT_ALL_EN,
    # Per-element prompts for cropped regions.
    "formula": DOTS_MOCR_PROMPT_FORMULA_LATEX,
    "table": DOTS_MOCR_PROMPT_TABLE_HTML,
    "text": DOTS_MOCR_PROMPT_OCR,
    "caption": DOTS_MOCR_PROMPT_OCR,
    "footnote": DOTS_MOCR_PROMPT_OCR,
    "list-item": DOTS_MOCR_PROMPT_OCR,
    "page-footer": DOTS_MOCR_PROMPT_OCR,
    "page-header": DOTS_MOCR_PROMPT_OCR,
    "section-header": DOTS_MOCR_PROMPT_OCR,
    "title": DOTS_MOCR_PROMPT_OCR,
}
