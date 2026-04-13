"""
Gradio UI: left = touch/canvas handwriting, right = LTX video (OpenVINO GenAI only).

OCR: TrOCR + OpenVINO. Video: Lightricks/LTX-Video IR + Text2VideoPipeline (no non-LTX video backend).
"""
import os
import traceback
from pathlib import Path

import gradio as gr
import yaml

import ocr_ov
from ltx_ov import generate_ltx_video, load_pipeline, ltx_ov_directory

ROOT = Path(__file__).resolve().parent
_APP_CSS_PATH = ROOT / "app.css"

_JS_SKETCH_MODAL_OPEN = """() => {
    document.getElementById("sketch-modal-column")?.classList.add("sketch-modal-open");
}"""
_JS_SKETCH_MODAL_CLOSE = """() => {
    document.getElementById("sketch-modal-column")?.classList.remove("sketch-modal-open");
}"""
_JS_SKETCH_MODAL_ESC = """() => {
    if (window._sketchModalEscBound) return;
    window._sketchModalEscBound = true;
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            document.getElementById("sketch-modal-column")?.classList.remove("sketch-modal-open");
        }
    });
}"""


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
os.environ["HF_HOME"] = str(ocr_ov.hf_home_from_config(CFG))
_ltx_ready = False


def _prompt_value(v) -> str:
    """Normalize Textbox payload."""
    if v is None:
        return ""
    return str(v).strip()


def _editor_to_pil(editor_value):
    """Gradio ImageEditor may return PIL or a dict with composite/background."""
    if editor_value is None:
        return None
    if isinstance(editor_value, dict):
        if editor_value.get("composite") is not None:
            return editor_value["composite"]
        if editor_value.get("background") is not None:
            return editor_value["background"]
    return editor_value


def ensure_ltx() -> None:
    global _ltx_ready
    if _ltx_ready:
        return
    ov = ltx_ov_directory(CFG)
    if not ov.exists():
        raise FileNotFoundError(
            f"LTX OpenVINO model not found at {ov}. Run: python export_ltx.py"
        )
    load_pipeline(CFG)
    _ltx_ready = True


def on_recognize(sketch, prompt_current):
    try:
        ocr_ov.init_from_config(CFG)
        pil = _editor_to_pil(sketch)
        new_text = (ocr_ov.run_ocr(pil) or "").strip()
        prev = _prompt_value(prompt_current)
        combined = " ".join(p for p in (prev, new_text) if p).strip()
        return combined, ""
    except Exception:
        return _prompt_value(prompt_current), f"OCR error:\n{traceback.format_exc()}"


def on_generate(prompt, progress=gr.Progress()):
    text = _prompt_value(prompt)
    if not text:
        return None, "Enter a prompt (or Recognize from handwriting first)."
    try:
        ensure_ltx()

        def cb(cur: int, total: int):
            progress(cur / max(total, 1), desc="LTX (OpenVINO)")

        path = generate_ltx_video(text, progress_callback=cb)
        return str(path), ""
    except Exception:
        return None, traceback.format_exc()


def build_ui():
    with gr.Blocks(
        title="Handwriting OCR + LTX (OpenVINO)",
        fill_width=True,
    ) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="sketch-modal-column"):
                with gr.Row(elem_classes=["sketch-modal-toolbar"]):
                    b_sketch_open = gr.Button(
                        "Open full-screen sketch",
                        elem_classes=["sketch-modal-open-btn"],
                    )
                    b_sketch_close = gr.Button(
                        "Close overlay",
                        elem_classes=["sketch-modal-close-btn"],
                    )
                sketch = gr.ImageEditor(
                    elem_id="ocr-sketch-editor",
                    label="Draw text (touch or mouse)",
                    type="pil",
                    image_mode="RGB",
                    height=420,
                    width="100%",
                    canvas_size=(1000, 420),
                    fixed_canvas=True,
                    brush=gr.Brush(default_size=6, colors=["#000000"]),
                    layers=False,
                    sources=[],
                    transforms=(),
                    buttons=["fullscreen"],
                )
            with gr.Column(scale=1):
                video = gr.Video(label="LTX output (OpenVINO)", height=420, width="100%")
        prompt = gr.Textbox(
            label="Prompt (from OCR or type)",
            lines=2,
            placeholder="e.g. dolphins leaping at sunset, ocean spray",
            elem_classes=["prompt-full-width"],
        )
        with gr.Row(equal_height=True, elem_classes=["three-btn-row"]):
            with gr.Column(scale=1):
                with gr.Row(equal_height=True, elem_classes=["rec-clear-inner"]):
                    b_rec = gr.Button(
                        "Recognize (TrOCR + OpenVINO)",
                        elem_classes=["rec-clear-half"],
                    )
                    b_clear = gr.Button(
                        "Clear Prompt",
                        elem_classes=["rec-clear-half"],
                    )
            with gr.Column(scale=1):
                b_gen = gr.Button(
                    "Generate Video (LTX + OpenVINO GenAI)",
                    variant="primary",
                    elem_classes=["gen-half"],
                )

        status = gr.Markdown()

        # No JS on Recognize/Generate: click+js can fire before ImageEditor/prompt values sync (empty OCR/prompt).
        b_rec.click(on_recognize, inputs=[sketch, prompt], outputs=[prompt, status])
        b_clear.click(lambda: "", outputs=[prompt])
        b_gen.click(on_generate, inputs=[prompt], outputs=[video, status])

        b_sketch_open.click(None, None, None, js=_JS_SKETCH_MODAL_OPEN)
        b_sketch_close.click(None, None, None, js=_JS_SKETCH_MODAL_CLOSE)
        demo.load(None, None, None, js=_JS_SKETCH_MODAL_ESC)

    return demo


if __name__ == "__main__":
    app_cfg = CFG.get("app", {})
    demo = build_ui()
    demo.launch(
        server_name=app_cfg.get("host", "127.0.0.1"),
        server_port=int(app_cfg.get("port", 7860)),
        share=bool(app_cfg.get("share", False)),
        css_paths=_APP_CSS_PATH,
        footer_links=[],
    )
