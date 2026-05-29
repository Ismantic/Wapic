#!/usr/bin/env python3
"""Convert LTP-cut news JSONL to CRF BMES format, randomly split into train/test.

Input:  {"source": "...", "cut": "词1 词2 ..."} per line
Output: train.txt / test.txt (CRF columnar: "char tag\\n", blank line between sentences)
        + test_nolabel.txt (only chars, for `wapic test`)

Use --seed for a deterministic shuffle; default 42.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


def word_to_bmes(word):
    chars = list(word)
    if len(chars) == 1:
        return [(chars[0], "S")]
    out = []
    for i, c in enumerate(chars):
        if i == 0:
            out.append((c, "B"))
        elif i == len(chars) - 1:
            out.append((c, "E"))
        else:
            out.append((c, "M"))
    return out


def line_to_pairs(line):
    obj = json.loads(line)
    cut = obj["cut"]
    pairs = []
    for w in cut.split():
        if not w:
            continue
        pairs.extend(word_to_bmes(w))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="cut.jsonl 路径")
    ap.add_argument("--out-train", default=None, help="训练集输出 (默认 data/news_train.txt)")
    ap.add_argument("--out-test", default=None, help="测试集输出 (默认 data/news_test.txt)")
    ap.add_argument("--out-nolabel", default=None, help="测试集去标签输出 (默认 data/news_test_nolabel.txt)")
    ap.add_argument("--test-ratio", type=float, default=0.01, help="测试集比例 (默认 0.01)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--limit", type=int, default=0, help="只读前 N 行 (0=全部)，用于快速回路")
    ap.add_argument("--max-chars", type=int, default=50, help="字符数超过则丢弃 (0=不限)")
    ap.add_argument("--min-chars", type=int, default=2, help="字符数低于则丢弃")
    args = ap.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    out_train = Path(args.out_train) if args.out_train else data_dir / "news_train.txt"
    out_test = Path(args.out_test) if args.out_test else data_dir / "news_test.txt"
    out_nolabel = Path(args.out_nolabel) if args.out_nolabel else data_dir / "news_test_nolabel.txt"

    rng = random.Random(args.seed)
    n_train = n_test = n_skip = n_too_long = n_too_short = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(out_train, "w", encoding="utf-8", buffering=1 << 20) as ftr, \
         open(out_test, "w", encoding="utf-8", buffering=1 << 20) as fte, \
         open(out_nolabel, "w", encoding="utf-8", buffering=1 << 20) as fnl:
        for i, line in enumerate(fin):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                pairs = line_to_pairs(line)
            except (json.JSONDecodeError, KeyError):
                n_skip += 1
                continue
            if not pairs:
                n_skip += 1
                continue
            if len(pairs) < args.min_chars:
                n_too_short += 1
                continue
            if args.max_chars and len(pairs) > args.max_chars:
                n_too_long += 1
                continue

            is_test = rng.random() < args.test_ratio
            target = fte if is_test else ftr
            for c, t in pairs:
                target.write(f"{c} {t}\n")
            target.write("\n")
            if is_test:
                for c, _ in pairs:
                    fnl.write(f"{c}\n")
                fnl.write("\n")
                n_test += 1
            else:
                n_train += 1

            if (i + 1) % 200000 == 0:
                print(f"  processed {i + 1} lines  (train={n_train} test={n_test} "
                      f"skip={n_skip} long={n_too_long} short={n_too_short})",
                      file=sys.stderr, flush=True)

    print(f"\ntrain   : {n_train} sentences -> {out_train}")
    print(f"test    : {n_test} sentences -> {out_test}")
    print(f"nolabel : {n_test} sentences -> {out_nolabel}")
    print(f"skip    : {n_skip}")
    print(f"too long: {n_too_long} (> {args.max_chars} chars)")
    print(f"too short: {n_too_short} (< {args.min_chars} chars)")


if __name__ == "__main__":
    main()
