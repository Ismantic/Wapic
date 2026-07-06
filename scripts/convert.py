#!/usr/bin/env python3
"""Convert the People's Daily 1998 (PFR `word/pos`) corpus to {source, cut} jsonl.

Part of the "train your own CWS model" tutorial (see TRAINING.md). Each PFR line
is a paragraph:

    19980101-01-001-001/m  迈向/v  充满/v  希望/n  ...  １/m  张/q  ）/w

We drop the paragraph-id token and every `/pos` tag, unwrap `[...]nt` named-
entity brackets into their inner words, merge consecutive personal-name tokens
(`江/nr 泽民/nr` -> `江泽民`, the retag2 "whole name" rule), then re-tokenize the
`cut` exactly like src/preprocess.cc PreSegment does at inference:

  * latin run -> one token, digit run -> one token (full-width included)
  * latin|digit boundary split; each punctuation mark its own token
  * whitespace dropped; the Han word boundaries from PFR are preserved

By default it reads data/199801.txt .. data/199806.txt and writes a train/test
split — the earlier months to data/PeopleDaily_1-5.jsonl and the last month to
data/PeopleDaily_6.jsonl:

    python3 scripts/convert.py
    python3 scripts/convert.py --split-names        # keep 江 泽民 unmerged
"""

import argparse
import glob
import json
import os
import re
import sys


def classify(cp):
    """Code point -> category: S space, D digit, L latin, H han, P punct.

    Mirrors ClassifyCodePoint in src/preprocess.cc — keep the two in sync if the
    character-type rules ever change.
    """
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


def resegment(source, words):
    """Retokenize `source` to the PreSegment convention.

    Han word boundaries from `words` are preserved; non-Han is merged/split by
    character type (latin/digit run -> one token, latin|digit split, each
    punctuation mark its own token) and whitespace is dropped.
    """
    starts = [False] * len(source)
    si = 0
    for w in words:
        while si < len(source) and classify(ord(source[si])) == "S":
            si += 1
        if source[si:si + len(w)] != w:
            return words                     # data anomaly: leave untouched
        starts[si] = True
        si += len(w)

    out, cur, prev_cat, prev_space = [], "", None, True
    for i, ch in enumerate(source):
        cat = classify(ord(ch))
        if cat == "S":
            prev_space = True
            continue
        if cur == "":
            start_new = True
        elif prev_space or cat != prev_cat or cat == "P":
            start_new = True
        elif cat == "H":
            start_new = starts[i]            # keep original Han segmentation
        else:
            start_new = False                # merge latin/digit run
        if start_new and cur:
            out.append(cur)
            cur = ""
        cur += ch
        prev_cat = cat
        prev_space = False
    if cur:
        out.append(cur)
    return out


def parse_pfr_line(line, merge_names=True):
    """PFR paragraph line -> list of words (id dropped, tags/brackets stripped)."""
    words = []
    name_buf = ""
    for tok in line.split()[1:]:              # [0] is the paragraph id
        if tok.startswith("["):
            tok = tok[1:]
        if "]" in tok:
            tok = tok.split("]", 1)[0]
        word, _, pos = tok.rpartition("/")
        if not word:
            word, pos = pos, ""
        if not word:
            continue
        if merge_names and pos == "nr":
            name_buf += word
            continue
        if name_buf:
            words.append(name_buf)
            name_buf = ""
        words.append(word)
    if name_buf:
        words.append(name_buf)
    return words


def convert(paths, out_path, merge_names):
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in paths:
            for line in open(path, encoding="utf-8"):
                words = parse_pfr_line(line, merge_names)
                if not words:
                    continue
                source = "".join(words)
                rec = {"source": source,
                       "cut": " ".join(resegment(source, words))}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def month_of(path):
    m = re.search(r"1998(\d\d)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    ap = argparse.ArgumentParser(description="People's Daily 1998 PFR -> jsonl")
    ap.add_argument("--data-dir", default=data_dir,
                    help="dir holding 199801.txt..199806.txt (default: repo data/)")
    ap.add_argument("--test-month", type=int, default=6,
                    help="month used as the test split (default: 6)")
    ap.add_argument("--split-names", action="store_true",
                    help="keep PFR name segmentation (江 泽民) instead of merging")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "1998*.txt")))
    if not files:
        print(f"No 1998*.txt under {args.data_dir}. See TRAINING.md for how to "
              f"fetch the People's Daily 1998 corpus.", file=sys.stderr)
        return 2

    train = [f for f in files if month_of(f) != args.test_month]
    test = [f for f in files if month_of(f) == args.test_month]
    months = sorted(month_of(f) for f in train)
    train_out = os.path.join(args.data_dir,
                             f"PeopleDaily_{months[0]}-{months[-1]}.jsonl")
    test_out = os.path.join(args.data_dir, f"PeopleDaily_{args.test_month}.jsonl")

    n1 = convert(train, train_out, not args.split_names)
    n2 = convert(test, test_out, not args.split_names)
    print(f"train: {n1:,} records -> {train_out}")
    print(f"test:  {n2:,} records -> {test_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
