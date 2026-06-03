from __future__ import annotations

import argparse
import json
from typing import Any

import gradio as gr
from PIL import Image

from optimaize.modules.ocr_bridge import (
    child_status,
    child_status_json,
    launch_child_ui,
    run_single_image_ocr,
    save_parent_upload,
)

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,750&family=Manrope:wght@400;500;650;750;850&display=swap');

:root {
    --oa-bg: #f7f1e8;
    --oa-bg-2: #ede2d3;
    --oa-surface: rgba(255, 252, 246, 0.88);
    --oa-surface-2: rgba(250, 243, 233, 0.78);
    --oa-text: #24201b;
    --oa-muted: #70675d;
    --oa-border: rgba(61, 48, 36, 0.14);
    --oa-accent: #c96342;
    --oa-accent-2: #df9b63;
    --oa-ink: #15130f;
    --oa-glow: rgba(211, 125, 75, 0.18);
    --oa-shadow: rgba(82, 58, 37, 0.14);
    --oa-cursor-x: 50%;
    --oa-cursor-y: 10%;
}

.dark,
body.dark,
.gradio-container.dark {
    --oa-bg: #171411;
    --oa-bg-2: #221c16;
    --oa-surface: rgba(37, 31, 26, 0.88);
    --oa-surface-2: rgba(47, 39, 32, 0.78);
    --oa-text: #f1e7da;
    --oa-muted: #c3b6a6;
    --oa-border: rgba(244, 210, 178, 0.16);
    --oa-accent: #e18a5a;
    --oa-accent-2: #efc27d;
    --oa-ink: #fff9ef;
    --oa-glow: rgba(225, 138, 90, 0.16);
    --oa-shadow: rgba(0, 0, 0, 0.34);
}

html, body, .gradio-container {
    min-height: 100%;
    font-family: 'Manrope', ui-sans-serif, system-ui, sans-serif !important;
    color: var(--oa-text) !important;
    background:
        radial-gradient(circle 520px at var(--oa-cursor-x) var(--oa-cursor-y), var(--oa-glow), transparent 64%),
        linear-gradient(135deg, var(--oa-bg), var(--oa-bg-2)) !important;
    transition: background 180ms ease-out, color 180ms ease-out;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.18;
    background-image: linear-gradient(rgba(61, 48, 36, 0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(61, 48, 36, 0.06) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(circle at 50% 18%, black, transparent 72%);
}

.gradio-container > .main,
.gradio-container .wrap,
.gradio-container .contain {
    position: relative;
    z-index: 1;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container .markdown h1,
.gradio-container .markdown h2 {
    color: var(--oa-text) !important;
    letter-spacing: -0.045em;
}

.gradio-container h1,
.oa-hero h1 {
    font-family: 'Fraunces', Georgia, serif !important;
}

.gradio-container p,
.gradio-container label,
.gradio-container span,
.gradio-container .markdown p {
    color: var(--oa-muted) !important;
}

.gradio-container .tabs {
    border: 1px solid var(--oa-border) !important;
    background: rgba(255, 255, 255, 0.14) !important;
    border-radius: 28px !important;
    padding: 8px !important;
    box-shadow: 0 22px 70px var(--oa-shadow) !important;
    backdrop-filter: blur(18px) saturate(1.08);
}

.gradio-container button[role='tab'] {
    border-radius: 999px !important;
    color: var(--oa-muted) !important;
    font-weight: 750 !important;
    letter-spacing: -0.015em;
}

.gradio-container button[role='tab'][aria-selected='true'] {
    color: var(--oa-text) !important;
    background: var(--oa-surface) !important;
    box-shadow: inset 0 0 0 1px var(--oa-border), 0 8px 22px var(--oa-shadow) !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .dataframe,
.gradio-container .tabitem {
    border-color: var(--oa-border) !important;
    background: var(--oa-surface) !important;
    border-radius: 24px !important;
    box-shadow: 0 18px 55px var(--oa-shadow), inset 0 1px 0 rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(16px) saturate(1.05);
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container .secondary-wrap {
    color: var(--oa-text) !important;
    background: var(--oa-surface-2) !important;
    border-color: var(--oa-border) !important;
    border-radius: 18px !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    outline: 2px solid rgba(201, 99, 66, 0.38) !important;
    box-shadow: 0 0 0 5px rgba(201, 99, 66, 0.10) !important;
}

.gradio-container button:not([role='tab']) {
    border-radius: 999px !important;
    color: #fff9ef !important;
    font-weight: 850 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--oa-accent), var(--oa-accent-2)) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    box-shadow: 0 12px 30px rgba(201, 99, 66, 0.22) !important;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease !important;
}

.gradio-container button:not([role='tab']):hover {
    transform: translateY(-1px);
    filter: saturate(1.04);
    box-shadow: 0 18px 40px rgba(201, 99, 66, 0.28) !important;
}

.oa-shell {
    max-width: 1180px;
    margin: 0 auto 18px;
}

.oa-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 18px 4px 24px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.76rem;
    font-weight: 850;
}

.oa-brand {
    color: var(--oa-text);
}

.oa-nav-links {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: flex-end;
}

.oa-pill,
.oa-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--oa-border);
    background: var(--oa-surface);
    color: var(--oa-muted) !important;
}

.oa-hero {
    position: relative;
    overflow: hidden;
    min-height: 430px;
    padding: clamp(32px, 7vw, 78px);
    border-radius: 34px;
    border: 1px solid var(--oa-border);
    background:
        linear-gradient(120deg, rgba(255,255,255,0.28), rgba(255,255,255,0.04)),
        radial-gradient(circle at 92% 8%, rgba(201, 99, 66, 0.16), transparent 35%);
    box-shadow: 0 28px 90px var(--oa-shadow);
}

.oa-hero::after {
    content: 'AIR / OCR / MODULES';
    position: absolute;
    right: clamp(20px, 5vw, 58px);
    bottom: clamp(18px, 4vw, 44px);
    max-width: 260px;
    color: rgba(112, 103, 93, 0.42);
    font-weight: 850;
    letter-spacing: 0.16em;
    line-height: 1.45;
    text-align: right;
}

.oa-kicker {
    color: var(--oa-accent) !important;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.78rem;
}

.oa-hero h1 {
    max-width: 820px;
    font-size: clamp(3rem, 8vw, 7.4rem) !important;
    line-height: 0.88 !important;
    margin: 0.42rem 0 1.1rem !important;
}

.oa-lede {
    max-width: 650px;
    font-size: clamp(1rem, 1.7vw, 1.25rem);
    line-height: 1.65;
}

.oa-actions,
.oa-card-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 24px;
}

.oa-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    margin: 24px 0;
}

.oa-card,
.oa-composer {
    padding: 22px;
    border-radius: 26px;
    border: 1px solid var(--oa-border);
    background: var(--oa-surface);
    box-shadow: 0 18px 55px var(--oa-shadow);
}

.oa-card h3,
.oa-composer h3 {
    margin: 0 0 8px;
    font-size: 1.15rem;
}

.oa-stat {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.3rem;
    color: var(--oa-text);
    line-height: 1;
}

.oa-section-title {
    margin: 8px 0 14px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--oa-muted) !important;
    font-weight: 850;
    font-size: 0.78rem;
}

@media (max-width: 760px) {
    .oa-nav { align-items: flex-start; flex-direction: column; }
    .oa-hero { min-height: auto; }
    .oa-hero::after { display: none; }
}
"""

APP_JS = r"""
() => {
    const root = document.documentElement;
    const update = (event) => {
        root.style.setProperty('--oa-cursor-x', `${event.clientX}px`);
        root.style.setProperty('--oa-cursor-y', `${event.clientY}px`);
    };
    window.addEventListener('pointermove', update, { passive: true });
    return 'ok';
}
"""


def status_rows() -> list[list[Any]]:
    status = child_status()
    return [
        ["Child path", status.child_path],
        ["Project directory", "available" if status.exists else "missing"],
        ["Source package", "available" if status.source_available else "missing"],
        ["OCR UI", "available" if status.ui_available else "missing"],
        ["History tooling", "available" if status.history_available else "missing"],
        ["Status", status.message],
    ]


def refresh_status() -> tuple[list[list[Any]], str]:
    return status_rows(), child_status_json()


def launch_ocr_ui(port: int, share: bool, inbrowser: bool) -> str:
    result = launch_child_ui(server_port=int(port), share=share, inbrowser=inbrowser)
    return json.dumps(result, indent=2, ensure_ascii=False)


def parent_single_image_ocr(image: Image.Image | None, model: str, threshold: float) -> tuple[str, str, str]:
    path = save_parent_upload(image)
    if path is None:
        return "", "", "Please upload an image first."
    result = run_single_image_ocr(str(path), model=model, threshold=float(threshold))
    return str(result.get("markdown", "")), str(result.get("html", "")), json.dumps(result, indent=2, ensure_ascii=False, default=str)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="OptimAIze") as app:
        gr.HTML(
            """
            <div class="oa-shell">
              <header class="oa-nav">
                <div class="oa-brand">OptimAIze</div>
                <div class="oa-nav-links">
                  <span class="oa-pill">Parent workspace</span>
                  <span class="oa-pill">OptimAIze-OCR ready</span>
                  <span class="oa-pill">Local CPU modules</span>
                </div>
              </header>
              <section class="oa-hero">
                <div class="oa-kicker">Claude calm × AirCenter editorial</div>
                <h1>Build with modular AI tools.</h1>
                <p class="oa-lede">A warm parent workspace for launching and orchestrating independent child systems. OCR runs as <strong>OptimAIze-OCR</strong> with its own UI, history database, visual cache, and CPU runtime.</p>
                <div class="oa-actions">
                  <span class="oa-chip">Lazy model loading</span>
                  <span class="oa-chip">Independent child UI</span>
                  <span class="oa-chip">History-aware OCR</span>
                </div>
              </section>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(
                    """
                    <div class="oa-card-grid">
                      <article class="oa-card">
                        <div class="oa-section-title">Module</div>
                        <div class="oa-stat">OCR</div>
                        <h3>OptimAIze-OCR</h3>
                        <p>Independent child project with its own source, UI, outputs, history DB, and visual-cache workflows.</p>
                      </article>
                      <article class="oa-card">
                        <div class="oa-section-title">Runtime</div>
                        <div class="oa-stat">CPU</div>
                        <h3>Lazy orchestration</h3>
                        <p>The parent checks status at startup, but heavy OCR models load only after an explicit OCR action.</p>
                      </article>
                      <article class="oa-card">
                        <div class="oa-section-title">Bridge</div>
                        <div class="oa-stat">API</div>
                        <h3>Parent-callable</h3>
                        <p>OptimAIze can launch the child UI or call OCR through a thin bridge while preserving child independence.</p>
                      </article>
                    </div>
                    """
                )
                status_table = gr.Dataframe(headers=["Item", "Value"], value=status_rows(), interactive=False)
                refresh_btn = gr.Button("Refresh module status")
                status_json = gr.Code(value=child_status_json(), language="json", label="Status JSON")
                with gr.Accordion("Workspace FAQ", open=False):
                    gr.Markdown(
                        """
                        **Can OCR run independently?** Yes. Run `python modules/OptimAIze-OCR/legacy/gradio/app_ui.py --server-port 7860`.

                        **Does the parent load OCR models at startup?** No. OCR imports/model loading happen only after explicit actions.

                        **Where are OCR outputs saved?** Child outputs remain under `modules/OptimAIze-OCR/outputs/`, including history and parent-call artifacts.
                        """
                    )
                refresh_btn.click(refresh_status, outputs=[status_table, status_json])

            with gr.Tab("OCR Module"):
                gr.HTML(
                    """
                    <div class="oa-composer">
                      <div class="oa-section-title">Primary action</div>
                      <h3>Run OCR from the parent workspace</h3>
                      <p>Upload one image for a direct bridge call, or launch the full child OCR UI for history, multi-page DocQA, and visual-cache workflows.</p>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        image = gr.Image(type="pil", label="Image")
                        with gr.Row():
                            model = gr.Dropdown(
                                choices=["falcon-ocr", "lighton-ocr", "paddleocr-vl", "glm-ocr", "surya-ocr", "surya-package", "dots-mocr"],
                                value="falcon-ocr",
                                label="Model",
                            )
                            threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Layout threshold")
                        run_btn = gr.Button("Run OCR through child bridge")
                    with gr.Column(scale=1):
                        gr.Markdown("### Launch child OCR UI")
                        port = gr.Number(value=7860, precision=0, label="Child UI port")
                        share = gr.Checkbox(value=False, label="Enable Gradio share")
                        inbrowser = gr.Checkbox(value=False, label="Open browser from child process")
                        launch_btn = gr.Button("Launch OptimAIze-OCR UI")
                        launch_result = gr.Code(language="json", label="Launch result")

                with gr.Row():
                    markdown = gr.Textbox(label="Markdown", lines=12)
                    html = gr.HTML(label="HTML preview")
                run_status = gr.Code(language="json", label="OCR result/status")

                launch_btn.click(launch_ocr_ui, inputs=[port, share, inbrowser], outputs=[launch_result])
                run_btn.click(parent_single_image_ocr, inputs=[image, model, threshold], outputs=[markdown, html, run_status])

            with gr.Tab("Settings / Modules"):
                gr.HTML(
                    """
                    <div class="oa-card">
                      <div class="oa-section-title">Module policy</div>
                      <h3>Optional child systems, explicit activation.</h3>
                      <p>Child modules stay independently runnable. The parent bridge resolves paths, launches child UI processes, or performs lazy imports only after user action.</p>
                    </div>
                    """
                )
                gr.Code(value=child_status_json(), language="json", label="Detected modules")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OptimAIze parent UI.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7850)
    parser.add_argument("--share", choices=["true", "false"], default="false")
    parser.add_argument("--inbrowser", choices=["true", "false"], default="false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share == "true",
        inbrowser=args.inbrowser == "true",
        theme=gr.themes.Soft(),
        css=APP_CSS,
        js=APP_JS,
    )


if __name__ == "__main__":
    main()
