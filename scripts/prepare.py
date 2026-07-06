#!/usr/bin/env python3
"""Turn the {source, cut} jsonl into wapic's native BMES training format.

Part of the "train your own CWS model" tutorial (see TRAINING.md). Reads the
train/test jsonl produced by scripts/convert.py and writes BMES column files
(`<char> <B|M|E|S>`, one char per line, blank line between sentences) that
`wapic fit` / `wapic test` consume directly.

By default:
    data/PeopleDaily_1-5.jsonl -> data/PeopleDaily_1-5.txt   (training set)
    data/PeopleDaily_6.jsonl   -> data/PeopleDaily_6.txt     (test set)

    python3 scripts/prepare.py
    python3 scripts/prepare.py IN.jsonl -o OUT.txt           # single file
"""

import argparse
import glob
import json
import os
import sys


def cut_to_bmes(cut):
    """A space-separated 'cut' string -> BMES lines for its characters."""
    lines = []
    for word in cut.split():
        if len(word) == 1:
            lines.append(word + " S")
        else:
            lines.append(word[0] + " B")
            for c in word[1:-1]:
                lines.append(c + " M")
            lines.append(word[-1] + " E")
    return lines


def jsonl_to_bmes(jsonl_path, bmes_path):
    n = 0
    with open(bmes_path, "w", encoding="utf-8") as out:
        for line in open(jsonl_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            cut = json.loads(line).get("cut", "")
            for l in cut_to_bmes(cut):
                out.write(l + "\n")
            out.write("\n")
            n += 1
    return n


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    ap = argparse.ArgumentParser(description="{source,cut} jsonl -> BMES")
    ap.add_argument("input", nargs="?", help="a single jsonl to convert")
    ap.add_argument("-o", "--out", help="output BMES path (with a single input)")
    ap.add_argument("--data-dir", default=data_dir)
    args = ap.parse_args()

    if args.input:
        out = args.out or os.path.splitext(args.input)[0] + ".txt"
        n = jsonl_to_bmes(args.input, out)
        print(f"{n:,} sentences -> {out}")
        return 0

    # Default: convert every PeopleDaily_*.jsonl next to it, same stem -> .txt
    jsonls = sorted(glob.glob(os.path.join(args.data_dir, "PeopleDaily_*.jsonl")))
    if not jsonls:
        print(f"No PeopleDaily_*.jsonl under {args.data_dir}. Run "
              f"scripts/convert.py first (see TRAINING.md).", file=sys.stderr)
        return 2
    for jp in jsonls:
        out = os.path.splitext(jp)[0] + ".txt"
        n = jsonl_to_bmes(jp, out)
        role = "test " if jp.endswith("_6.jsonl") else "train"
        print(f"[{role}] {n:,} sentences -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
