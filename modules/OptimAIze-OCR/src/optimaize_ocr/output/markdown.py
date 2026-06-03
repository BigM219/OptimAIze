# Markdown assembly and formatting utilities for OCR results

def assemble_markdown(results: list[dict]) -> str:
    """Format and assemble raw OCR result list into a structured Markdown document."""
    markdown_blocks = []
    for res in results:
        cat = res["category"].strip().lower()
        text = res["text"]
        if not text:
            continue
            
        if cat in ("title", "doc_title"):
            markdown_blocks.append(f"# {text}\n")
        elif cat in ("section-header", "paragraph_title"):
            markdown_blocks.append(f"## {text}\n")
        elif cat in ("page-header", "header"):
            markdown_blocks.append(f"*{text}*\n")
        elif cat == "list-item":
            # Ensure it has bullet formatting
            if not text.startswith(("- ", "* ", "1. ")):
                markdown_blocks.append(f"- {text}")
            else:
                markdown_blocks.append(text)
        elif cat == "formula":
            # Ensure LaTeX block format
            if not text.startswith("$$"):
                markdown_blocks.append(f"\n$$\n{text}\n$$\n")
            else:
                markdown_blocks.append(text)
        elif cat == "table":
            markdown_blocks.append(f"\n{text}\n")
        elif cat == "caption":
            markdown_blocks.append(f"\n*Caption: {text}*\n")
        elif cat == "footnote":
            markdown_blocks.append(f"\n[^footnote]: {text}\n")
        else:
            markdown_blocks.append(f"{text}\n")

    return "\n".join(markdown_blocks)
