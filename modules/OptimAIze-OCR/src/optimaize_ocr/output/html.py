# HTML rendering engine for OCR results

import re

def format_inline_elements(text_str: str) -> str:
    """Escape HTML tags and parse basic markdown inline styling (bold, italic, code)."""
    # Escape HTML tags safely
    escaped = text_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Bold: **bold** or __bold__
    escaped = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<strong>\1\2</strong>', escaped)
    # Italic: *italic* or _italic_
    escaped = re.sub(r'\*(.*?)\*|_(.*?)_', r'<em>\1\2</em>', escaped)
    # Inline code: `code`
    escaped = re.sub(r'`(.*?)`', r'<code>\1</code>', escaped)
    # Newlines to HTML line breaks
    escaped = escaped.replace('\n', '<br>')
    return escaped

def markdown_table_to_html(md_table: str) -> str:
    """Convert raw Markdown table string into valid HTML table representation."""
    lines = [line.strip() for line in md_table.strip().split('\n') if line.strip()]
    if not lines:
        return ""
    
    html_rows = []
    is_header = True
    
    for line in lines:
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        cells = [c.strip() for c in line.split('|')]
        
        # Check if it is a separator row
        if all(all(char in '-: \t' for char in cell) for cell in cells if cell):
            continue
            
        if is_header:
            cells_html = "".join(f"<th>{format_inline_elements(cell)}</th>" for cell in cells)
            html_rows.append(f"<thead><tr>{cells_html}</tr></thead><tbody>")
            is_header = False
        else:
            cells_html = "".join(f"<td>{format_inline_elements(cell)}</td>" for cell in cells)
            html_rows.append(f"<tr>{cells_html}</tr>")
            
    if not is_header:
        html_rows.append("</tbody>")
        
    return f'<table>{"".join(html_rows)}</table>'

def generate_html(
    results: list[dict],
    model_type: str,
    image_name: str = "Document"
) -> str:
    """Generate a clean and readable HTML file showing OCR results with visible table borders."""
    html_blocks = []
    
    for idx, res in enumerate(results):
        cat = res["category"].strip().lower()
        text = res["text"]
        if not text:
            continue
            
        if cat in ("title", "doc_title"):
            html_blocks.append(f'<h1>{format_inline_elements(text)}</h1>')
        elif cat in ("section-header", "paragraph_title"):
            html_blocks.append(f'<h2>{format_inline_elements(text)}</h2>')
        elif cat in ("page-header", "header"):
            html_blocks.append(f'<p><em>{format_inline_elements(text)}</em></p>')
        elif cat in ("page-footer", "footer"):
            html_blocks.append(f'<p><small style="color: #64748b;">{format_inline_elements(text)}</small></p>')
        elif cat == "list-item":
            clean_li = text.strip()
            if clean_li.startswith(("- ", "* ", "+ ")):
                clean_li = clean_li[2:]
            elif clean_li.startswith("1. "):
                clean_li = clean_li[3:]
            html_blocks.append(f'<li class="list-item">{format_inline_elements(clean_li)}</li>')
        elif cat == "formula":
            math_text = text.strip()
            if math_text.startswith("$$"):
                math_text = math_text[2:]
            if math_text.endswith("$$"):
                math_text = math_text[:-2]
            math_text = math_text.strip()
            html_blocks.append(f'<div class="math-block tex2jax_process">$$\n{math_text}\n$$</div>')
        elif cat == "table":
            if "<table" in text:
                # Let's keep it but make sure standard CSS applies
                html_blocks.append(text)
            else:
                table_html = markdown_table_to_html(text)
                html_blocks.append(table_html)
        elif cat == "caption":
            html_blocks.append(f'<p style="font-style: italic; color: #64748b; text-align: center;">{format_inline_elements(text)}</p>')
        elif cat == "footnote":
            html_blocks.append(f'<p style="font-size: 0.9em; color: #64748b;">{format_inline_elements(text)}</p>')
        else:
            html_blocks.append(f'<p>{format_inline_elements(text)}</p>')
            
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parsed Document: {{IMAGE_NAME}}</title>
    <!-- MathJax for high-fidelity LaTeX -->
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true
            },
            options: {
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background-color: #ffffff;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
        }
        h1 {
            font-size: 2.2em;
            margin-bottom: 0.8em;
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.3em;
        }
        h2 {
            font-size: 1.6em;
            margin-top: 1.5em;
            margin-bottom: 0.6em;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 0.2em;
        }
        p {
            margin-bottom: 1em;
            font-size: 1.05em;
        }
        .list-item {
            margin-bottom: 0.5em;
            font-size: 1.05em;
            margin-left: 20px;
        }
        .math-block {
            margin: 1.5em 0;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 1em;
        }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 10px 12px;
            text-align: left;
        }
        th {
            background-color: #f1f5f9;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
    </style>
</head>
<body>
    {{CONTENT}}
</body>
</html>
"""
    return html_template.replace("{{IMAGE_NAME}}", image_name).replace("{{CONTENT}}", "\n".join(html_blocks))
