#!/usr/bin/env python3
"""Download the main Wapic Chinese word segmentation model.

Model repo: https://huggingface.co/Ismantic/wapic-cws
Usage: python3 scripts/download_model.py
"""

import argparse
import sys
from pathlib import Path


REPO_ID = "Ismantic/wapic-cws"
MODEL_FILENAME = "model/wapic-cws.wac"
# Download into data/ so the repo's model/ subdir lands at data/model/.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Download {MODEL_FILENAME} from {REPO_ID}."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when a cached copy is available",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Missing dependency: install it with "
            "`uv pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}/{MODEL_FILENAME}")
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            local_dir=output_dir,
            force_download=args.force,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    model_path = Path(model_path).resolve()
    if not model_path.is_file() or model_path.stat().st_size == 0:
        print(f"Download failed: invalid model file at {model_path}", file=sys.stderr)
        return 1

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {model_path} ({size_mb:.1f} MiB)")
    print(f"Run: ./build/wapic -m {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
