#!/usr/bin/env python3
"""Check that segmented data obeys the retag2 / PreSegment character-type rules.

The Wapic model only ever labels pure-Han runs; latin / digit / punctuation are
split out by character type before the CRF sees them (see src/preprocess.cc).
This script verifies a dataset was tokenized the same way, i.e. every word is a
single-category run and punctuation is one mark per token:

  1. no word mixes character categories        e.g. "2015年", "GDP3", "IPv6"
  2. punctuation is one code point per token    e.g. "……", "3.5", "::"  (bad)
  3. latin/digit runs are not over-split        two adjacent latin (or digit)
     tokens with no separating space in source  (jsonl only — needs `source`)

Accepts `.jsonl[.gz]` ({"source","cut"}) and BMES `.txt[.gz]` files. jsonl runs
all three checks; BMES txt (no source spacing) runs checks 1-2 only.

Usage:
    python3 scripts/check_retag2.py FILE [FILE ...]
    python3 scripts/check_retag2.py                 # default: data/dataset/*
    python3 scripts/check_retag2.py --limit 100000 FILE   # first N sentences
Exit code is non-zero if any violation is found.
"""

import argparse
import gzip
import io
import json
import sys
from pathlib import Path

from retag2 import classify


def word_violation(w):
    """Return a violation tag for a single token, or None if it conforms."""
    cats = [classify(ord(c)) for c in w]
    if "S" in cats:
        return "space-in-word"
    if "P" in cats:
        # punctuation must be exactly one mark per token
        return None if len(w) == 1 else "multichar-or-mixed-punct"
    if len(set(cats)) > 1:
        return "mixed-category"
    return None


def open_text(path):
    raw = gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")
    return io.TextIOWrapper(raw, encoding="utf-8")


def iter_jsonl(fh):
    for ln, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        yield ln, obj.get("source", ""), obj.get("cut", "")


def iter_bmes(fh):
    """Yield (line_no, source, cut) reconstructed from BMES columns."""
    chars, tags, start = [], [], 1
    for ln, line in enumerate(fh, 1):
        line = line.rstrip("\n")
        if line == "":
            if chars:
                yield start, "".join(chars), _bmes_words(chars, tags)
                chars, tags = [], []
            start = ln + 1
            continue
        char, _, tag = line.rpartition(" ")
        chars.append(char)
        tags.append(tag)
    if chars:
        yield start, "".join(chars), _bmes_words(chars, tags)


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
    return " ".join(words)


def oversplit_violations(source, words):
    """Check 3: latin/digit run split across tokens with no source space.

    Walks the source, matching each word; a boundary with no intervening
    whitespace between two same-category latin/digit tokens is an over-split.
    Returns a list of violation tags (empty if OK); 'source-mismatch' if the
    words don't align to the source.
    """
    out = []
    si, prev_last_cat, prev_had_space = 0, None, True
    n = len(source)
    for w in words:
        had_space = False
        while si < n and classify(ord(source[si])) == "S":
            had_space = True
            si += 1
        if source[si:si + len(w)] != w:
            return ["source-mismatch"]
        si += len(w)
        first_cat = classify(ord(w[0])) if w else "S"
        if (prev_last_cat is not None and not had_space
                and prev_last_cat == first_cat and prev_last_cat in ("L", "D")):
            out.append("oversplit-" + prev_last_cat)
        prev_last_cat = classify(ord(w[-1])) if w else "S"
    return out


def check_file(path, limit=None, samples=3):
    is_jsonl = ".jsonl" in Path(path).name
    counts = {}
    n_sent = n_word = 0
    examples = {}

    with open_text(path) as fh:
        rows = iter_jsonl(fh) if is_jsonl else iter_bmes(fh)
        for i, (ln, source, cut) in enumerate(rows):
            if limit is not None and i >= limit:
                break
            n_sent += 1
            words = cut.split()
            n_word += len(words)

            for w in words:
                v = word_violation(w)
                if v:
                    counts[v] = counts.get(v, 0) + 1
                    examples.setdefault(v, (ln, source[:60], w))

            if is_jsonl:
                for v in oversplit_violations(source, words):
                    counts[v] = counts.get(v, 0) + 1
                    examples.setdefault(v, (ln, source[:60], cut[:60]))

    total = sum(counts.values())
    kind = "jsonl" if is_jsonl else "bmes "
    print(f"[{kind}] {path}")
    print(f"    sentences={n_sent:,}  words={n_word:,}  violations={total:,}")
    for v in sorted(counts):
        ln, src, tok = examples[v]
        print(f"    - {v}: {counts[v]:,}   e.g. line {ln}: {tok!r}  in  {src!r}")
    return total


def main():
    ap = argparse.ArgumentParser(description="Check retag2/PreSegment conformance.")
    ap.add_argument("files", nargs="*", help="jsonl[.gz] or bmes txt[.gz] files")
    ap.add_argument("--limit", type=int, default=None,
                    help="only check the first N sentences per file")
    args = ap.parse_args()

    files = args.files
    if not files:
        root = Path(__file__).resolve().parents[1] / "data" / "dataset"
        files = sorted(str(p) for p in root.rglob("*")
                       if p.is_file() and (".jsonl" in p.name or p.name.endswith(".txt")
                                           or p.name.endswith(".txt.gz")))
        if not files:
            print("No data files found under data/dataset/. Pass paths explicitly.",
                  file=sys.stderr)
            return 2

    grand = 0
    for f in files:
        grand += check_file(f, limit=args.limit)
        print()
    print(f"TOTAL violations: {grand:,}")
    return 1 if grand else 0


if __name__ == "__main__":
    sys.exit(main())
