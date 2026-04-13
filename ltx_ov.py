"""
Text-to-video via OpenVINO GenAI Text2VideoPipeline + LTX-Video IR only.

Uses exported Hugging Face model: Lightricks/LTX-Video (no PyTorch/CUDA inference path).
See Intel notebook: openvino_notebooks/notebooks/ltx-video/ltx-video.ipynb
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import numpy as np

_pipeline = None
_config: dict[str, Any] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def ltx_ov_directory(cfg: dict) -> Path:
    rel = cfg.get("ltx", {}).get("ov_model_dir", "models/LTX-Video/fp16")
    p = Path(rel)
    if not p.is_absolute():
        p = _project_root() / p
    return p


def _save_video_from_genai(
    video_tensor: Any,
    filename: Path,
    fps: float,
    video_crf: int | None = None,
) -> None:
    video_data = video_tensor.data
    arr = np.asarray(video_data)
    kw: dict[str, Any] = {"fps": fps}
    if video_crf is not None:
        kw["ffmpeg_params"] = ["-crf", str(int(video_crf)), "-pix_fmt", "yuv420p"]
    with imageio.get_writer(str(filename), **kw) as writer:
        for i in range(arr.shape[1]):
            frame = arr[0, i]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)


def load_pipeline(cfg: dict) -> None:
    """Load OpenVINO GenAI Text2VideoPipeline from exported LTX-Video IR directory."""
    global _pipeline, _config
    ov = ltx_ov_directory(cfg)
    if not ov.exists():
        raise FileNotFoundError(
            f"LTX OpenVINO IR not found: {ov}\n"
            "Export Lightricks/LTX-Video first:  python export_ltx.py"
        )
    import openvino_genai as ov_genai

    device = str(cfg.get("ltx", {}).get("device", "GPU"))
    _pipeline = ov_genai.Text2VideoPipeline(str(ov), device)
    _config = cfg


def generate_ltx_video(
    prompt: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Run LTX text-to-video inference (OpenVINO GenAI only).
    Returns path to the saved .mp4 under the project output/ folder.
    """
    if not (prompt or "").strip():
        raise ValueError("Prompt is empty.")
    if _pipeline is None:
        raise RuntimeError("LTX pipeline not loaded. Call load_pipeline(config) first.")

    import openvino_genai as ov_genai

    ltx = _config.get("ltx", {})
    width = int(ltx.get("width", 1024))
    height = int(ltx.get("height", 576))
    num_frames = int(ltx.get("num_frames", 25))
    num_inference_steps = int(ltx.get("num_inference_steps", 60))
    guidance_scale = float(ltx.get("guidance_scale", 4.0))
    frame_rate = float(ltx.get("frame_rate", 25))
    negative = str(
        ltx.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted",
        )
    )
    seed = int(ltx.get("seed", 42))
    _crf = ltx.get("video_crf")
    video_crf = int(_crf) if _crf is not None else None

    def callback(step: int, num_steps: int, _latent) -> bool:
        if progress_callback:
            progress_callback(step + 1, num_steps)
        return False

    out_dir = _project_root() / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"ltx_{stamp}.mp4"

    output = _pipeline.generate(
        prompt.strip(),
        negative_prompt=negative,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=ov_genai.TorchGenerator(seed),
        guidance_scale=guidance_scale,
        frame_rate=frame_rate,
        callback=callback,
    )
    _save_video_from_genai(output.video, out_path, fps=frame_rate, video_crf=video_crf)
    return out_path
