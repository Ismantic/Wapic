"""Evaluate a wapic model on the structured bad-case eval set.

Pass criterion: the LTP-NER ground-truth name appears as a single token in wapic's cut.
Reports overall pass rate + per-category pass rate.

Usage: python scripts/eval_badcase_set.py --model PATH [--input data/badcase_eval.jsonl]
"""
import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict


def wapic_cut_batch(model, srcs):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in srcs:
            for c in s:
                fin.write(c + "\n")
            fin.write("\n")
        ip = fin.name
    op = ip + ".out"
    subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                   capture_output=True, check=False)
    out_tags = []
    cur = []
    with open(op, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if cur:
                    out_tags.append(cur)
                    cur = []
                continue
            if line.startswith("score="): continue
            p = line.split()
            if p and p[0] in ("B", "M", "E", "S"):
                cur.append(p[0])
    if cur:
        out_tags.append(cur)
    os.unlink(ip); os.unlink(op)
    word_lists = []
    for s, tags in zip(srcs, out_tags):
        w = []; cw = ""
        for c, t in zip(s, tags):
            if t in ("B", "S"):
                if cw: w.append(cw)
                cw = c
            else:
                cw += c
        if cw: w.append(cw)
        word_lists.append(w)
    return word_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default="data/badcase_eval.jsonl")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--show-fail", type=int, default=10,
                    help="Print first N failures per category")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.input)]
    total = len(cases)

    by_cat_total = defaultdict(int)
    by_cat_pass = defaultdict(int)
    fail_samples = defaultdict(list)

    import re
    for i in range(0, len(cases), args.batch):
        chunk = cases[i:i + args.batch]
        srcs = [c["source"] for c in chunk]
        wls = wapic_cut_batch(args.model, srcs)
        for c, wl in zip(chunk, wls):
            cat = c["category"]
            by_cat_total[cat] += 1
            # Punctuation like · or - is split out as independent tokens by PD standard.
            # So check each non-empty part of the name (after split on · or -) appears in cut.
            parts = re.split(r'[·・•\-]', c["name"])
            parts = [p for p in parts if p]
            if all(p in wl for p in parts):
                by_cat_pass[cat] += 1
            else:
                fail_samples[cat].append((c["name"], c["source"], " ".join(wl)))

    overall_pass = sum(by_cat_pass.values())
    print(f"=== {os.path.basename(args.model)} ===")
    print(f"OVERALL: {overall_pass}/{total} = {100 * overall_pass / total:.1f}%\n")
    print("By category:")
    for cat in sorted(by_cat_total, key=lambda c: -by_cat_total[c]):
        n_tot = by_cat_total[cat]
        n_pass = by_cat_pass[cat]
        print(f"  {cat:20s}: {n_pass}/{n_tot} = {100 * n_pass / n_tot:5.1f}%")
    if args.show_fail > 0:
        print("\nSample failures (per category, first N):")
        for cat in sorted(fail_samples):
            print(f"  --- {cat} ---")
            for nm, src, cut in fail_samples[cat][:args.show_fail]:
                print(f"    {nm}: {src[:50]} ... | cut: {cut[:60]}")


if __name__ == "__main__":
    main()
