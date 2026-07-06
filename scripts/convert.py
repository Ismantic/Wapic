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

from retag2 import resegment


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
                       "cut": " ".join(resegment(source, words)[0])}
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
