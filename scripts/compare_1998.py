"""
Compare Wapic-12M vs LTP/base1 on 1998 People's Daily June (1998-06.txt).
Both evaluated against the 1998 human gold standard.

用法:
    python scripts/compare_1998.py --limit 2000
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def words_to_spans(words):
    spans = set()
    pos = 0
    for w in words:
        spans.add((pos, pos + len(w)))
        pos += len(w)
    return spans


def tags_to_words(text, tags):
    words = []
    cur = ""
    for c, t in zip(text, tags):
        if t == "B":
            if cur: words.append(cur)
            cur = c
        elif t == "M":
            cur += c
        elif t == "E":
            cur += c; words.append(cur); cur = ""
        elif t == "S":
            if cur: words.append(cur)
            words.append(c); cur = ""
        else:
            cur += c
    if cur: words.append(cur)
    return words


def f1(pred_words, gold_words):
    pred = words_to_spans(pred_words)
    gold = words_to_spans(gold_words)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    P = tp / (tp + fp) * 100 if (tp + fp) else 0
    R = tp / (tp + fn) * 100 if (tp + fn) else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    return P, R, F, pred == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/1998-06.txt")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--wapic-model", default="data/all12m_model.wac")
    ap.add_argument("--wapic-bin", default="./build/wapic")
    ap.add_argument("--ltp-model", default="LTP/base1")
    ap.add_argument("--sample-diff", type=int, default=5)
    args = ap.parse_args()

    # 1. Load source + gold, clean NER bracket marker "[" so it's its own token
    def clean_gold(tokens):
        out = []
        for w in tokens:
            if w.startswith("[") and len(w) > 1:
                out.append("[")
                out.append(w[1:])
            else:
                out.append(w)
        return out

    sentences = []
    golds = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = obj["source"]
            cut = obj["cut"]
            if not src or len(src) > args.max_chars or len(src) < 2:
                continue
            sentences.append(src)
            golds.append(clean_gold(cut.split()))
            if len(sentences) >= args.limit:
                break
    print(f"Loaded {len(sentences)} sentences with gold from {args.src}")

    # 2. Run LTP
    print(f"\nRunning LTP/{args.ltp_model.split('/')[-1]} ...", flush=True)
    t0 = time.time()
    from ltp import LTP
    try:
        ltp = LTP(args.ltp_model, local_files_only=True)
    except Exception:
        ltp = LTP(args.ltp_model)
    ltp.to("cuda"); ltp.half(); ltp.eval()
    ltp_cws = []
    for s in range(0, len(sentences), 64):
        batch = sentences[s:s+64]
        out = ltp.pipeline(batch, tasks=["cws"])
        ltp_cws.extend(out.cws)
    ltp_time = time.time() - t0
    print(f"  done in {ltp_time:.1f}s")

    # 3. Run Wapic (keep input/output files for inspection)
    print("\nRunning Wapic ...", flush=True)
    t0 = time.time()
    in_path = Path("/tmp/wapic_1998_in.txt")
    out_path = Path("/tmp/wapic_1998_out.txt")
    with open(in_path, "w", encoding="utf-8") as f:
        for s in sentences:
            for ch in s:
                f.write(ch + "\n")
            f.write("\n")
    subprocess.run([args.wapic_bin, "test", "-m", args.wapic_model,
                    str(in_path), str(out_path)],
                   check=True, capture_output=True)
    wapic_cws = []
    cur_tags = []
    sent_idx = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("score="):
                continue
            if not line:
                if cur_tags:
                    wapic_cws.append(tags_to_words(sentences[sent_idx], cur_tags))
                    sent_idx += 1
                    cur_tags = []
                continue
            parts = line.split()
            if parts:
                cur_tags.append(parts[0])
    if cur_tags:
        wapic_cws.append(tags_to_words(sentences[sent_idx], cur_tags))
    wapic_time = time.time() - t0
    print(f"  done in {wapic_time:.1f}s")

    # Write per-sentence Wapic cut to a readable file
    wapic_cut_path = Path("/tmp/wapic_1998_cut.txt")
    with open(wapic_cut_path, "w", encoding="utf-8") as f:
        for src, words in zip(sentences, wapic_cws):
            f.write(f"原: {src}\n")
            f.write(f"切: {' '.join(words)}\n\n")
    print(f"  Wapic 分词结果保存到: {wapic_cut_path}")

    # Speed
    print(f"\nSpeed: LTP/base1={len(sentences)/ltp_time:.0f} sent/s (GPU), "
          f"Wapic={len(sentences)/wapic_time:.0f} sent/s (CPU single-thread)")

    # 4. F1 each vs gold
    ltp_TP = ltp_FP = ltp_FN = 0; ltp_exact = 0
    wap_TP = wap_FP = wap_FN = 0; wap_exact = 0
    for gold, lc, wc in zip(golds, ltp_cws, wapic_cws):
        for tokens, exact_var, TP_var, FP_var, FN_var, label in [
            (lc, "ltp_exact", "ltp_TP", "ltp_FP", "ltp_FN", "ltp"),
            (wc, "wap_exact", "wap_TP", "wap_FP", "wap_FN", "wap"),
        ]:
            pred_spans = words_to_spans(tokens)
            gold_spans = words_to_spans(gold)
            tp = len(pred_spans & gold_spans)
            fp = len(pred_spans - gold_spans)
            fn = len(gold_spans - pred_spans)
            if label == "ltp":
                ltp_TP += tp; ltp_FP += fp; ltp_FN += fn
                if pred_spans == gold_spans: ltp_exact += 1
            else:
                wap_TP += tp; wap_FP += fp; wap_FN += fn
                if pred_spans == gold_spans: wap_exact += 1

    def stats(tp, fp, fn, exact, total):
        P = tp / (tp + fp) * 100 if (tp + fp) else 0
        R = tp / (tp + fn) * 100 if (tp + fn) else 0
        F = 2 * P * R / (P + R) if (P + R) else 0
        return P, R, F, exact / total * 100

    print(f"\n=== Both vs 1998 PD gold ({len(sentences)} sentences) ===\n")
    lp, lr, lf, le = stats(ltp_TP, ltp_FP, ltp_FN, ltp_exact, len(sentences))
    wp, wr, wf, we = stats(wap_TP, wap_FP, wap_FN, wap_exact, len(sentences))
    print(f"{'Model':<20s} {'P':>7s} {'R':>7s} {'F1':>7s} {'ExactMatch':>12s}")
    print(f"{'LTP/base1':<20s} {lp:>7.2f} {lr:>7.2f} {lf:>7.2f} {le:>11.2f}%")
    print(f"{'Wapic-12M':<20s} {wp:>7.2f} {wr:>7.2f} {wf:>7.2f} {we:>11.2f}%")

    # 5. Sample diffs
    print(f"\n=== Sample sentences (first {args.sample_diff} with any diff) ===\n")
    shown = 0
    for src, gold, lc, wc in zip(sentences, golds, ltp_cws, wapic_cws):
        if " ".join(gold) == " ".join(lc) == " ".join(wc):
            continue
        ok_l = "✓" if gold == lc else "✗"
        ok_w = "✓" if gold == wc else "✗"
        print(f"原文: {src}")
        print(f"Gold: {' '.join(gold)}")
        print(f"LTP {ok_l}: {' '.join(lc)}")
        print(f"Wap {ok_w}: {' '.join(wc)}")
        print()
        shown += 1
        if shown >= args.sample_diff:
            break


if __name__ == "__main__":
    main()
