#!/usr/bin/env python3
"""Download and verify the model embedded in the PyPI model package."""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request


MODEL_REVISION = "50c205c8d3071ac9ea6ecd7a945d546df8f3843b"
MODEL_SHA256 = "b440d2efb187428cffccd0796073ed5847ba089748e67858dfe34ed600f70d8e"
MODEL_URL = (
    "https://huggingface.co/Ismantic/Wapic-CWS/resolve/"
    f"{MODEL_REVISION}/model/wapic-cws.wac"
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "packages/wapic-cws-model/src/wapic_model/data/wapic-cws.wac"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the verified Wapic model package."
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".download")
    try:
        print(f"Downloading pinned model revision {MODEL_REVISION}")
        with urllib.request.urlopen(MODEL_URL) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        actual = sha256(temporary)
        if actual != MODEL_SHA256:
            print(
                f"Model checksum mismatch: expected {MODEL_SHA256}, got {actual}",
                file=sys.stderr,
            )
            return 1
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Prepared {output} ({output.stat().st_size / (1 << 20):.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
