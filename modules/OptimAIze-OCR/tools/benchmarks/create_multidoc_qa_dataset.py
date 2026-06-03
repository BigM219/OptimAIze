from pathlib import Path
import sys

OCR_MODULE_ROOT = Path(__file__).resolve().parents[3]
OCR_SRC_ROOT = OCR_MODULE_ROOT / "src"
for _path in (OCR_SRC_ROOT, OCR_MODULE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


VENDORS = [
    "Aurora Grid Systems",
    "Beacon EV Services",
    "Cobalt Charge Works",
    "Delta Field Robotics",
    "Evergreen Power Labs",
    "Harbor Sensor Group",
]
CUSTOMERS = [
    "Northwind Transit",
    "Silverline Logistics",
    "Metro Health Campus",
    "Cedar County Schools",
    "Atlas Retail Group",
    "Summit Water Authority",
]
MANAGERS = ["Mira Chen", "Owen Patel", "Lina Brooks", "Noah Reed", "Sofia Marin", "Theo Walsh"]
APPROVERS = ["Diana Kennedy", "Everett Cross", "Florella Fitzgerald", "Jonas Kim", "Priya Shah"]
PROJECTS = ["EV Depot Upgrade", "Rural Charger Rollout", "Fleet OCR Audit", "Solar Meter Retrofit", "Warehouse Safety Refresh"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic synthetic multi-document QA dataset.")
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--pages-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multidoc_qa_dataset"))
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=1150)
    return parser.parse_args()


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    regular = next((p for p in candidates if Path(p).exists()), None)
    bold = next((p for p in bold_candidates if Path(p).exists()), regular)
    if regular is None:
        font = ImageFont.load_default()
        return font, font, font
    return (
        ImageFont.truetype(regular, 26),
        ImageFont.truetype(bold, 34),
        ImageFont.truetype(regular, 22),
    )


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], x: int, y: int, font: ImageFont.ImageFont, spacing: int = 42) -> None:
    for line in lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += spacing


def draw_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, title_font, body_font, lines: list[str]) -> None:
    x1, y1, x2, y2 = xy
    draw.rectangle(xy, outline=(40, 40, 40), width=2)
    draw.rectangle((x1, y1, x2, y1 + 52), fill=(235, 241, 250), outline=(40, 40, 40), width=2)
    draw.text((x1 + 18, y1 + 10), title, fill=(20, 35, 70), font=title_font)
    draw_lines(draw, lines, x1 + 22, y1 + 76, body_font, spacing=40)


def page_image(width: int, height: int, heading: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 86), fill=(26, 47, 87))
    return image, draw


def render_page(path: Path, width: int, height: int, heading: str, sections: list[tuple[str, list[str]]]) -> str:
    body_font, title_font, small_font = load_fonts()
    image, draw = page_image(width, height, heading)
    draw.text((32, 22), heading, fill="white", font=title_font)
    y = 122
    reference_lines = [heading]
    for title, lines in sections:
        box_h = 88 + 40 * len(lines)
        draw_box(draw, (42, y, width - 42, y + box_h), title, small_font, body_font, lines)
        reference_lines.append(title)
        reference_lines.extend(lines)
        y += box_h + 34
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return "\n".join(reference_lines)


def money(value: int) -> str:
    return f"${value:,}.00"


def make_group(rng: random.Random, group_index: int, pages_per_group: int, images_dir: Path, width: int, height: int) -> dict:
    group_id = f"group_{group_index:03d}"
    vendor = rng.choice(VENDORS)
    customer = rng.choice(CUSTOMERS)
    manager = rng.choice(MANAGERS)
    approver = rng.choice(APPROVERS)
    project = rng.choice(PROJECTS)
    invoice_no = f"INV-{rng.randint(1200, 9899)}"
    contract_no = f"CTR-{rng.randint(220, 899)}-{rng.choice(['A', 'B', 'C'])}"
    due_date = f"2026-{rng.randint(6, 12):02d}-{rng.randint(1, 28):02d}"
    issue_date = f"2026-{rng.randint(1, 5):02d}-{rng.randint(1, 28):02d}"
    subtotal = rng.randrange(2400, 8900, 100)
    tax = int(subtotal * 0.08)
    total = subtotal + tax
    doc_b_total = total + rng.choice([-700, -350, 450, 900])
    payment_code = f"PAY-{rng.randint(10000, 99999)}"
    missing_field = rng.choice(["routing number", "purchase order", "site contact phone"])

    page_specs = [
        (
            "Document A - Project Summary",
            [
                ("Header", [f"Vendor: {vendor}", f"Customer: {customer}", f"Project: {project}", f"Invoice No.: {invoice_no}"]),
                ("Responsibility", [f"Project Manager: {manager}", f"Issue Date: {issue_date}", f"Contract No.: {contract_no}"]),
            ],
        ),
        (
            "Document A - Charges",
            [
                ("Line Items", ["Hardware kit: " + money(subtotal - 900), "Installation labor: $900.00", f"Tax: {money(tax)}", f"Invoice Total: {money(total)}"]),
                ("Payment", [f"Payment Code: {payment_code}", f"Due Date: {due_date}", f"Missing Field: {missing_field}"]),
            ],
        ),
        (
            "Document B - Comparison Record",
            [
                ("Comparison", [f"Document A Total: {money(total)}", f"Document B Total: {money(doc_b_total)}", f"Larger Document: {'Document B' if doc_b_total > total else 'Document A'}"]),
                ("Approval", [f"Approved By: {approver}", f"Approved Project: {project}", f"Approval Date: {due_date}"]),
            ],
        ),
    ]
    if pages_per_group > 3:
        for page_no in range(4, pages_per_group + 1):
            page_specs.append((
                f"Document C - Appendix {page_no - 3}",
                [("Notes", [f"Appendix Page: {page_no}", f"Related Contract: {contract_no}", "No additional payment changes."])],
            ))

    pages = []
    for page_index, (heading, sections) in enumerate(page_specs[:pages_per_group], start=1):
        image_path = images_dir / f"{group_id}_page_{page_index}.png"
        reference = render_page(image_path, width, height, heading, sections)
        pages.append({"page": page_index, "image": str(image_path.as_posix()), "reference": reference})

    larger = "Document B" if doc_b_total > total else "Document A"
    questions = [
        {
            "type": "multi_page_join",
            "question": "What is the invoice total for the vendor named on page 1?",
            "answer": f"{vendor}: {money(total)}",
            "required_terms": [vendor, money(total)],
            "evidence_pages": [1, 2],
        },
        {
            "type": "multi_page_join",
            "question": "Which project manager is responsible for the project approved on the final page?",
            "answer": f"{manager}",
            "required_terms": [manager, project],
            "evidence_pages": [1, min(3, pages_per_group)],
        },
        {
            "type": "cross_document_compare",
            "question": "Compare Document A and Document B. Which document has the larger total?",
            "answer": larger,
            "required_terms": [larger, money(total), money(doc_b_total)],
            "evidence_pages": [2, min(3, pages_per_group)],
        },
        {
            "type": "missing_info",
            "question": "What payment instruction field is missing?",
            "answer": missing_field,
            "required_terms": [missing_field],
            "evidence_pages": [2],
        },
    ]
    return {"group_id": group_id, "pages": pages, "questions": questions}


def main() -> None:
    args = parse_args()
    if args.pages_per_group < 2:
        raise SystemExit("--pages-per-group must be >= 2")
    rng = random.Random(args.seed)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    images_dir = args.output_dir / "images"
    groups = [make_group(rng, i, args.pages_per_group, images_dir, args.width, args.height) for i in range(args.groups)]
    manifest = {
        "version": 1,
        "seed": args.seed,
        "pages_per_group": args.pages_per_group,
        "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote={manifest_path} groups={len(groups)} pages={sum(len(g['pages']) for g in groups)}")


if __name__ == "__main__":
    main()
