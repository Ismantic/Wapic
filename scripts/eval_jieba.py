"""Evaluate jieba on the same test sets as wapic for direct comparison."""
import argparse
import json
import re
import sys

import jieba

PUNCT_SPLIT = re.compile(r'[·・•\-]')


def jieba_cut(s):
    return list(jieba.cut(s))


def spans_from_bmes(chars, tags):
    sp = []
    p = 0
    cur = ""
    for c, t in zip(chars, tags):
        if t in ("B", "S"):
            if cur: sp.append((p - len(cur), p))
            cur = c
        else:
            cur += c
        p += 1
    if cur: sp.append((p - len(cur), p))
    return set(sp)


def spans_from_words(words):
    sp = []
    p = 0
    for w in words:
        sp.append((p, p + len(w)))
        p += len(w)
    return set(sp)


def eval_f1(gold_path):
    # gold: char-tab-tag per line, blank between sentences
    sents = []
    chars, tags = [], []
    for line in open(gold_path, encoding="utf-8"):
        line = line.rstrip()
        if not line:
            if chars:
                sents.append((chars, tags))
                chars, tags = [], []
            continue
        x = line.split()
        chars.append(x[0])
        tags.append(x[-1])
    if chars: sents.append((chars, tags))

    tp = fp = fn = 0
    for chars, gtags in sents:
        src = "".join(chars)
        gold_sp = spans_from_bmes(chars, gtags)
        pred_words = jieba_cut(src)
        pred_sp = spans_from_words(pred_words)
        tp += len(gold_sp & pred_sp)
        fp += len(pred_sp - gold_sp)
        fn += len(gold_sp - pred_sp)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r) * 100


def eval_15case():
    sys.path.insert(0, "scripts")
    from test_name_cases import CASES
    n_ok = 0
    total = 0
    for nm, sents in CASES:
        for s in sents:
            total += 1
            words = jieba_cut(s)
            if nm in words:
                n_ok += 1
    return n_ok, total


def eval_badcase(path):
    cases = [json.loads(l) for l in open(path)]
    n_ok = 0
    for c in cases:
        words = jieba_cut(c["source"])
        parts = [p for p in PUNCT_SPLIT.split(c["name"]) if p]
        if all(p in words for p in parts):
            n_ok += 1
    return n_ok, len(cases)


def main():
    print("=== jieba vs wapic-20260601 ===\n")

    print("F1_pdmp:")
    f1 = eval_f1("data/pd_mp_test.txt")
    print(f"  jieba: {f1:.2f}")

    print("\nF1_12m:")
    f1 = eval_f1("data/all12m_test.txt")
    print(f"  jieba: {f1:.2f}")

    print("\n15-case:")
    n, t = eval_15case()
    print(f"  jieba: {n}/{t}")

    print("\nbadcase_v2 (200):")
    n, t = eval_badcase("data/badcase_eval_v2.jsonl")
    print(f"  jieba: {n}/{t} = {100*n/t:.1f}%")

    print("\nbadcase_v3 (500):")
    n, t = eval_badcase("data/badcase_eval_v3.jsonl")
    print(f"  jieba: {n}/{t} = {100*n/t:.1f}%")


if __name__ == "__main__":
    main()
