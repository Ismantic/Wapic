#!/usr/bin/env python3
"""Evaluate trained model(s) on BMES gold test sets (span-level P/R/F1).

    python3 scripts/test.py MODEL [MODEL ...]
        # the two published test sets under data/dataset/
        #   wapic-cws-data-test-2 (PD-1998) and -test-1 (12M)
    python3 scripts/test.py MODEL --gold data/PeopleDaily_6.txt [GOLD ...]
        # your own BMES gold file(s)

For each gold set it derives the char-only input, runs `wapic test`, and reports
span-level precision / recall / F1. Replaces the old evaluate.sh.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_nolabel(gold, out):
    """gold BMES -> char-only input (`awk '{print $1}'`, blank lines kept)."""
    with open(out, "w", encoding="utf-8") as f:
        for line in open(gold, encoding="utf-8"):
            parts = line.split()
            f.write((parts[0] if parts else "") + "\n")


def read_gold(path):
    sents, tags = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "":
            if tags:
                sents.append(tags); tags = []
            continue
        tags.append(line.split()[-1])
    if tags:
        sents.append(tags)
    return sents


def read_pred(path):
    sents, tags = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "":
            if tags:
                sents.append(tags); tags = []
            continue
        if line.startswith("score="):
            continue
        tags.append(line.split()[0])
    if tags:
        sents.append(tags)
    return sents


def spans(tags):
    res, start = set(), 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] in ("B", "S"):
            res.add((start, i)); start = i
    return res


def evaluate(binpath, model, gold):
    tmp = tempfile.mkdtemp()
    nolbl, pred = os.path.join(tmp, "in.txt"), os.path.join(tmp, "pred.txt")
    make_nolabel(gold, nolbl)
    subprocess.run([binpath, "test", "-m", model, nolbl, pred],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    g_sents, p_sents = read_gold(gold), read_pred(pred)
    os.unlink(nolbl); os.unlink(pred); os.rmdir(tmp)

    tp = fp = fn = 0
    for gt, pt in zip(g_sents, p_sents):
        if len(gt) != len(pt):
            continue
        g, p = spans(gt), spans(pt)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P * 100, R * 100, F * 100


def main():
    ap = argparse.ArgumentParser(description="Span-F1 evaluation for wapic models.")
    ap.add_argument("models", nargs="+", help="model .wac file(s)")
    ap.add_argument("--gold", nargs="+", help="custom BMES gold file(s)")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--bin", default=str(ROOT / "build" / "wapic"))
    args = ap.parse_args()

    if args.gold:
        golds = [(Path(g).name, g) for g in args.gold]
    else:
        golds = [("PD-1998", os.path.join(args.data_dir, "wapic-cws-data-test-2.txt")),
                 ("12M", os.path.join(args.data_dir, "wapic-cws-data-test-1.txt"))]

    for model in args.models:
        if not os.path.isfile(model):
            print(f"Model not found: {model}", file=sys.stderr)
            return 1
        print(f"{os.path.basename(model)}")
        for name, gold in golds:
            if not os.path.isfile(gold):
                print(f"  {name:12} gold not found: {gold}", file=sys.stderr)
                continue
            P, R, F = evaluate(args.bin, model, gold)
            print(f"  {name:12} F1={F:.2f}  P={P:.2f}  R={R:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
