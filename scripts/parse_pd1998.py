"""
Parse People's Daily 1998 raw format into (source, gold_cut) pairs.

Raw format per line:
    19980601-01-002-002/m  词1/pos1  词2/pos2  [词3/pos  词4/pos]ner  ...

Cleaning:
  - drop sentence ID (first token)
  - for each `word/pos` strip the /pos suffix
  - strip leading `[` and trailing `]<ner>` markers
  - the NER spans become just multiple separate tokens (not joined)
"""

import argparse
import json
import re
import sys
from pathlib import Path


TOKEN_RE = re.compile(r"^\[?(.+?)/[a-zA-Z]+\]?[a-zA-Z]*$")


def parse_token(tok):
    """Return word, or None if it's not a valid token."""
    if not tok:
        return None
    # strip outer [ and ]xxx markers
    t = tok
    if t.startswith("["):
        t = t[1:]
    # find last / which separates word from POS
    slash = t.rfind("/")
    if slash < 0:
        return None
    word = t[:slash]
    rest = t[slash+1:]
    # rest may be "POS" or "POS]NER" — strip ]NER
    if "]" in rest:
        rest = rest[:rest.index("]")]
    if not word:
        return None
    return word


def parse_line(line):
    """Return (source_text, [tokens]) or None."""
    line = line.strip()
    if not line:
        return None
    parts = re.split(r"\s+", line)
    if not parts:
        return None
    # parts[0] is sentence ID like 19980601-01-002-002/m — skip it
    if not parts[0].startswith("199"):
        return None
    words = []
    for p in parts[1:]:
        w = parse_token(p)
        if w:
            words.append(w)
    if not words:
        return None
    source = "".join(words)
    return source, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="199806.txt 等原始文件")
    ap.add_argument("--out", required=True, help="输出 jsonl: {source, cut}")
    args = ap.parse_args()

    n_ok = n_skip = 0
    with open(args.src, "r", encoding="utf-8", errors="replace") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            r = parse_line(line)
            if not r:
                n_skip += 1
                continue
            source, words = r
            fout.write(json.dumps({"source": source, "cut": " ".join(words)},
                                  ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"parsed: {n_ok}, skipped: {n_skip}", file=sys.stderr)


if __name__ == "__main__":
    main()
