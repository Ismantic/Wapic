"""
Compare Wapic vs LTP/base1 on raw sentences.

输入: 一份 raw 句子文本 (每行一句, 比如 4-9M 行之后没参与训练的 OpenNews)
流程:
  1) LTP/base1 切词作为参考（base1 风格的"gold"，注意：训练集本来就用 base1 标的，
     这里只是看 wapic 模仿 base1 有多准）
  2) Wapic 二进制 REPL 模式或 fit 模式切词（用 wapic test 流程）
  3) 比较两者输出的 token 边界 F1

用法:
    python scripts/compare_ltp.py \
        --raw /home/tfbao/Data/data/OpenNews.100M.sentences.txt \
        --skip 4000000 --limit 5000 \
        --wapic-model data/all12m_model.wac \
        --wapic-bin ./build/wapic
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Raw text, one sentence per line")
    ap.add_argument("--skip", type=int, default=4_000_000,
                    help="Skip first N lines (avoid training overlap)")
    ap.add_argument("--limit", type=int, default=5000, help="Use N lines after skip")
    ap.add_argument("--max-chars", type=int, default=50, help="Drop sentences longer than this")
    ap.add_argument("--wapic-model", default="data/all12m_model.wac")
    ap.add_argument("--wapic-bin", default="./build/wapic")
    ap.add_argument("--ltp-model", default="LTP/base1")
    args = ap.parse_args()

    # 1. Sample sentences
    print(f"Sampling {args.limit} sentences from {args.raw} (skip {args.skip})...", flush=True)
    sentences = []
    with open(args.raw, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < args.skip:
                continue
            s = line.strip()
            if not s or len(s) > args.max_chars or len(s) < 2:
                continue
            sentences.append(s)
            if len(sentences) >= args.limit:
                break
    print(f"  got {len(sentences)} sentences")

    # 2. LTP cut
    print(f"\nRunning LTP/{args.ltp_model.split('/')[-1]} ...", flush=True)
    t0 = time.time()
    from ltp import LTP
    try:
        ltp = LTP(args.ltp_model, local_files_only=True)
    except Exception:
        ltp = LTP(args.ltp_model)
    ltp.to("cuda")
    ltp.half()
    ltp.eval()

    ltp_cws = []
    batch_size = 64
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        out = ltp.pipeline(batch, tasks=["cws"])
        ltp_cws.extend(out.cws)
    ltp_time = time.time() - t0
    print(f"  LTP done in {ltp_time:.1f}s ({len(sentences)/ltp_time:.0f} sent/s)")

    # 3. Wapic cut: write input as BMES-style char-per-line (test_nolabel format), run wapic test
    print("\nRunning Wapic ...", flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "wapic_in.txt"
        out_path = Path(td) / "wapic_out.txt"
        with open(in_path, "w", encoding="utf-8") as f:
            for s in sentences:
                for ch in s:
                    f.write(ch + "\n")
                f.write("\n")
        subprocess.run([args.wapic_bin, "test", "-m", args.wapic_model,
                        str(in_path), str(out_path)],
                       check=True, capture_output=True)
        # Parse: per sentence -> "score=" line, then "tag score" per char, then blank.
        wapic_cws = []
        cur_tags = []
        sent_idx = 0
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("score="):
                    continue
                if not line:
                    if cur_tags and sent_idx < len(sentences):
                        words = tags_to_words(sentences[sent_idx], cur_tags)
                        wapic_cws.append(words)
                        sent_idx += 1
                    cur_tags = []
                    continue
                parts = line.split()
                if parts:
                    cur_tags.append(parts[0])
        if cur_tags and sent_idx < len(sentences):
            words = tags_to_words(sentences[sent_idx], cur_tags)
            wapic_cws.append(words)
    wapic_time = time.time() - t0
    print(f"  Wapic done in {wapic_time:.1f}s ({len(sentences)/wapic_time:.0f} sent/s)")
    print(f"\nSpeed ratio: Wapic is {ltp_time/wapic_time:.1f}x faster than LTP/base1")

    # 4. F1 comparison (LTP as reference; how well does Wapic match)
    print("\n=== Wapic vs LTP/base1 (LTP is reference) ===\n")
    tp = fp = fn = 0
    exact_match = 0
    for ltp_tokens, wap_tokens in zip(ltp_cws, wapic_cws):
        ltp_spans = words_to_spans(ltp_tokens)
        wap_spans = words_to_spans(wap_tokens)
        tp += len(ltp_spans & wap_spans)
        fp += len(wap_spans - ltp_spans)
        fn += len(ltp_spans - wap_spans)
        if ltp_spans == wap_spans:
            exact_match += 1

    P = tp / (tp + fp) * 100 if (tp + fp) else 0
    R = tp / (tp + fn) * 100 if (tp + fn) else 0
    F = 2 * P * R / (P + R) if (P + R) else 0
    print(f"  Token boundary Precision: {P:.2f}%")
    print(f"  Token boundary Recall:    {R:.2f}%")
    print(f"  Token boundary F1:        {F:.2f}%")
    print(f"  Sentence exact match:     {exact_match}/{len(ltp_cws)} = {exact_match/len(ltp_cws)*100:.2f}%")

    # 5. Sample diffs
    print("\n=== Sample diffs (first 5 mismatches) ===\n")
    shown = 0
    for sent, ltp_w, wap_w in zip(sentences, ltp_cws, wapic_cws):
        if " ".join(ltp_w) == " ".join(wap_w):
            continue
        print(f"原: {sent}")
        print(f"LTP: {' '.join(ltp_w)}")
        print(f"Wap: {' '.join(wap_w)}")
        print()
        shown += 1
        if shown >= 5:
            break


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
            if cur:
                words.append(cur)
            cur = c
        elif t == "M":
            cur += c
        elif t == "E":
            cur += c
            words.append(cur)
            cur = ""
        elif t == "S":
            if cur:
                words.append(cur)
            words.append(c)
            cur = ""
        else:
            cur += c
    if cur:
        words.append(cur)
    return words


if __name__ == "__main__":
    main()
