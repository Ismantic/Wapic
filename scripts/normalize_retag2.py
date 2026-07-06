#!/usr/bin/env python3
"""Normalize segmented data to the retag2 / PreSegment character-type convention.

Rewrites the tokenization of non-Han text so it matches exactly what
src/preprocess.cc PreSegment produces at inference, while leaving the Han
(Chinese) word segmentation untouched:

  * latin run  -> one token          (Open LLM -> OpenLLM)
  * digit run  -> one token          (full-width ８ ２ ２ ２６ -> ８２２２６)
  * latin|digit boundary -> split    (RyanTang007 -> RyanTang 007)
  * each punctuation mark -> its own token
  * whitespace -> dropped (it is a word boundary, never a labeled char)
  * Han word boundaries -> preserved exactly (the CRF's segmentation)

Reads jsonl[.gz] ({"source","cut"}) or BMES txt[.gz]. Writes both a BMES
txt file and a jsonl file (add --gzip for .gz outputs).

    python3 scripts/normalize_retag2.py IN --out-prefix OUT [--gzip]
"""

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

from retag2 import resegment, words_to_bmes


def normalize(source, orig_words):
    """Return (new_words, changed, aligned); unchanged when unaligned."""
    words, aligned = resegment(source, orig_words)
    return words, (words != orig_words), aligned


def open_read(path):
    raw = gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")
    return io.TextIOWrapper(raw, encoding="utf-8")


def open_write(path, gz):
    raw = gzip.open(path, "wb") if gz else open(path, "wb")
    return io.TextIOWrapper(raw, encoding="utf-8", newline="\n")


def iter_input(path):
    is_jsonl = ".jsonl" in Path(path).name
    with open_read(path) as fh:
        if is_jsonl:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                yield o.get("source", ""), o.get("cut", "").split()
        else:
            chars, tags = [], []
            for line in fh:
                line = line.rstrip("\n")
                if line == "":
                    if chars:
                        yield "".join(chars), _bmes_words(chars, tags)
                        chars, tags = [], []
                    continue
                c, _, t = line.rpartition(" ")
                chars.append(c)
                tags.append(t)
            if chars:
                yield "".join(chars), _bmes_words(chars, tags)


def _bmes_words(chars, tags):
    words, cur = [], ""
    for c, t in zip(chars, tags):
        if t in ("B", "S"):
            if cur:
                words.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        words.append(cur)
    return words


def main():
    ap = argparse.ArgumentParser(description="Normalize data to retag2/PreSegment.")
    ap.add_argument("input", help="jsonl[.gz] or BMES txt[.gz]")
    ap.add_argument("--out-prefix", required=True,
                    help="output path stem; writes <stem>.txt and <stem>.jsonl")
    ap.add_argument("--gzip", action="store_true", help="gzip the outputs")
    args = ap.parse_args()

    ext = ".gz" if args.gzip else ""
    txt_path = args.out_prefix + ".txt" + ext
    jsonl_path = args.out_prefix + ".jsonl" + ext

    n_sent = n_changed = n_unaligned = 0
    ftxt = open_write(txt_path, args.gzip)
    fjsonl = open_write(jsonl_path, args.gzip)
    with ftxt, fjsonl:
        for source, orig_words in iter_input(args.input):
            words, changed, aligned = normalize(source, orig_words)
            n_sent += 1
            n_changed += changed
            n_unaligned += (not aligned)
            for line in words_to_bmes(words):
                ftxt.write(line + "\n")
            ftxt.write("\n")
            fjsonl.write(json.dumps(
                {"source": source, "cut": " ".join(words)},
                ensure_ascii=False) + "\n")

    print(f"{args.input}")
    print(f"  sentences={n_sent:,}  changed={n_changed:,}  "
          f"unaligned(skipped)={n_unaligned:,}")
    print(f"  wrote {txt_path}")
    print(f"  wrote {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
