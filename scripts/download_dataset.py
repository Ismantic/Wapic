#!/usr/bin/env python3
"""Download Wapic evaluation data or the complete release dataset."""

import argparse
import sys
from pathlib import Path


REPO_ID = "Ismantic/wapic-cws-data"
# Download into data/ so the repo's dataset/ subdir lands at data/dataset/.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Download release data from {REPO_ID}."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="download training data in addition to evaluation files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when cached files are available",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Missing dependency: install it with "
            "`uv pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = None if args.full else ["dataset/wapic-cws-data-test-*"]

    scope = "complete dataset" if args.full else "evaluation data"
    print(f"Downloading {scope} from {REPO_ID}")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=output_dir,
            allow_patterns=patterns,
            force_download=args.force,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
