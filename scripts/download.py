#!/usr/bin/env python3
"""Download the Wapic release model and/or evaluation data from Hugging Face.

    python3 scripts/download.py model          # model  -> data/model/
    python3 scripts/download.py data            # eval data -> data/dataset/
    python3 scripts/download.py data --full     # + training data
    python3 scripts/download.py all             # both

Files land under data/ so the repo layout is data/model/wapic-cws.wac and
data/dataset/wapic-cws-data-*.
"""

import argparse
import sys
from pathlib import Path

MODEL_REPO = "Ismantic/Wapic-CWS"
MODEL_FILE = "model/wapic-cws.wac"
DATA_REPO = "Ismantic/Wapic-CWS-Data"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data"


def download_model(out_dir, force):
    from huggingface_hub import hf_hub_download
    print(f"Downloading {MODEL_REPO}/{MODEL_FILE}")
    path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                                repo_type="model", local_dir=out_dir,
                                force_download=force)).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        print(f"Download failed: invalid file at {path}", file=sys.stderr)
        return 1
    print(f"Saved {path} ({path.stat().st_size / (1 << 20):.1f} MiB)")
    return 0


def download_data(out_dir, full, force):
    from huggingface_hub import snapshot_download
    patterns = None if full else ["dataset/wapic-cws-data-test-*"]
    scope = "complete dataset" if full else "evaluation data"
    print(f"Downloading {scope} from {DATA_REPO}")
    snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=out_dir,
                      allow_patterns=patterns, force_download=force)
    print(f"Saved to {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Download Wapic model/data from HF.")
    ap.add_argument("target", choices=("model", "data", "all"),
                    help="what to download")
    ap.add_argument("--full", action="store_true",
                    help="with 'data'/'all': also fetch the training data")
    ap.add_argument("--force", action="store_true",
                    help="re-download even when a cached copy exists")
    ap.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_DIR,
                    help=f"destination directory (default: {DEFAULT_DIR})")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Missing dependency: `uv pip install huggingface_hub`.",
              file=sys.stderr)
        return 1

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rc = 0
    try:
        if args.target in ("model", "all"):
            rc |= download_model(out, args.force)
        if args.target in ("data", "all"):
            rc |= download_data(out, args.full, args.force)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
