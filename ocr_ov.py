"""Handwriting OCR: TrOCR + Optimum Intel OpenVINO (OVModelForVision2Seq)."""
from pathlib import Path
from typing import Optional

from PIL import Image

_processor = None
_model = None
_model_id: str = "microsoft/trocr-large-handwritten"
_device: str = "GPU"
_cache_dir: Optional[str] = None
_hf_cache_used: Optional[str] = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def hf_home_from_config(config: dict) -> Path:
    """Hugging Face cache root under the project (hub + transformers snapshots)."""
    rel = (config.get("models") or {}).get("hf_home", "models/huggingface")
    if isinstance(rel, str) and not rel.strip():
        rel = "models/huggingface"
    p = Path(str(rel).strip())
    if not p.is_absolute():
        p = project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _ensure_rgb_white_bg(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")


def load_ocr(model_id: str, device: str = "GPU") -> None:
    global _processor, _model, _model_id, _device, _hf_cache_used
    cache = (
        _cache_dir
        if _cache_dir is not None
        else str(hf_home_from_config({}))
    )
    if (
        _model is not None
        and _model_id == model_id
        and _device == device
        and _hf_cache_used == cache
    ):
        return
    from transformers import TrOCRProcessor
    from optimum.intel import OVModelForVision2Seq

    kwargs = {"cache_dir": cache}
    _processor = TrOCRProcessor.from_pretrained(model_id, **kwargs)
    _model = OVModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        device=device,
        **kwargs,
    )
    _model_id = model_id
    _device = device
    _hf_cache_used = cache


def run_ocr(pil_image: Optional[Image.Image]) -> str:
    if pil_image is None:
        return ""
    load_ocr(_model_id, _device)
    img = _ensure_rgb_white_bg(pil_image)
    inputs = _processor(images=img, return_tensors="pt")
    generated_ids = _model.generate(**inputs)
    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return (text or "").strip().rstrip(".")


def init_from_config(config: dict) -> None:
    global _cache_dir
    o = config.get("ocr", {})
    mid = o.get("model_id", "microsoft/trocr-large-handwritten")
    dev = o.get("device", "GPU")
    _cache_dir = str(hf_home_from_config(config))
    load_ocr(mid, dev)
