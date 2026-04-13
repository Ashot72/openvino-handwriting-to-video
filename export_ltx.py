"""
Export Hugging Face Lightricks/LTX-Video to OpenVINO IR for openvino_genai.Text2VideoPipeline.

Run (venv activated):
  python export_ltx.py

Output matches config.yaml key ltx.ov_model_dir (default: models/LTX-Video/fp16).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MODEL_ID = "Lightricks/LTX-Video"
WEIGHT_FORMAT = "fp16"


def _optimum_cli() -> str:
    scripts = Path(sys.executable).resolve().parent
    win = scripts / "optimum-cli.exe"
    nix = scripts / "optimum-cli"
    if win.exists():
        return str(win)
    if nix.exists():
        return str(nix)
    found = shutil.which("optimum-cli")
    if found:
        return found
    raise FileNotFoundError(
        "optimum-cli not found. Install: pip install optimum[openvino] optimum-intel"
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg_path = root / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    from ocr_ov import hf_home_from_config

    hf_home = hf_home_from_config(cfg)
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home)

    out_dir = root / "models" / "LTX-Video" / WEIGHT_FORMAT
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = _optimum_cli()
    cmd = [
        cli,
        "export",
        "openvino",
        "-m",
        MODEL_ID,
        "--weight-format",
        WEIGHT_FORMAT,
        "--library",
        "diffusers",
        str(out_dir),
    ]
    print("HF_HOME (hub downloads):", hf_home)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(root), env=env)
    print(f"Done. OpenVINO LTX IR: {out_dir}")
    print("In config.yaml set: ltx.ov_model_dir: models/LTX-Video/fp16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
