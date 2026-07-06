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


def classify(cp):
    if cp in (0x20, 0x09, 0x0A, 0x0D, 0x0C, 0x0B, 0x00A0, 0x3000):
        return "S"
    if 0x30 <= cp <= 0x39 or 0xFF10 <= cp <= 0xFF19:
        return "D"
    if (0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A or 0x00C0 <= cp <= 0x024F
            or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A):
        return "L"
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF or cp == 0x3007
            or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2EBEF):
        return "H"
    return "P"


def normalize(source, orig_words):
    """Return (new_words, changed, aligned).

    aligned is False when orig_words don't line up with source (data anomaly);
    in that case new_words == orig_words and the sentence is left untouched.
    """
    # Mark, per source position, whether an original word starts there.
    starts = [False] * len(source)
    si = 0
    for w in orig_words:
        while si < len(source) and classify(ord(source[si])) == "S":
            si += 1
        if source[si:si + len(w)] != w:
            return orig_words, False, False
        starts[si] = True
        si += len(w)

    new_words = []
    cur = ""
    prev_cat = None
    prev_space = True
    for i, ch in enumerate(source):
        cat = classify(ord(ch))
        if cat == "S":
            prev_space = True
            continue
        if cur == "":
            start_new = True
        elif prev_space or cat != prev_cat or cat == "P":
            start_new = True                 # boundary / category flip / punct
        elif cat == "H":
            start_new = starts[i]            # keep original Han segmentation
        else:
            start_new = False                # merge latin/digit run
        if start_new and cur:
            new_words.append(cur)
            cur = ""
        cur += ch
        prev_cat = cat
        prev_space = False
    if cur:
        new_words.append(cur)

    return new_words, (new_words != orig_words), True


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


def bmes_lines(words):
    for w in words:
        if len(w) == 1:
            yield w + " S"
        else:
            yield w[0] + " B"
            for c in w[1:-1]:
                yield c + " M"
            yield w[-1] + " E"


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
            for line in bmes_lines(words):
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
