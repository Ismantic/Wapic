"""
对 OpenNews 跑 LTP/base1 NER。
输出: data/raw/opennews_ner*.jsonl
  每行 {"source": str, "cut": str, "ner": [[tag, text, start, end], ...]}
  cut 字段已应用 NER Nh 合并（人名作为单 token）+ 标点拆分

支持：
  - 纯文本（每行一句）或 cut.jsonl 两种输入格式
  - --filter-nh-only: 只输出含 Nh 实体的句子
  - --limit-out: 输出达到 N 句后停止
  - --start N: 跳过前 N 行（断点续跑）
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_punct import split_word


def merge_nh(cut_words, ner_entities):
    """把 NER Nh 实体覆盖到的 cut 词合并为一个 token.

    LTP NER 返回 (tag, text, start_word_idx, end_word_idx) — start/end 都是 cut 词的下标。
    """
    if not ner_entities:
        return list(cut_words)
    nh_ranges = []  # (start, end) inclusive in cut-word idx
    for tag, text, sw, ew in ner_entities:
        if tag == "Nh":
            nh_ranges.append((sw, ew))
    if not nh_ranges:
        return list(cut_words)
    nh_ranges.sort()
    out = []
    i = 0
    ri = 0
    while i < len(cut_words):
        if ri < len(nh_ranges) and i == nh_ranges[ri][0]:
            s, e = nh_ranges[ri]
            merged = "".join(cut_words[s:e + 1])
            out.append(merged)
            i = e + 1
            ri += 1
        else:
            out.append(cut_words[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/ltp_opennews_3m.jsonl")
    ap.add_argument("--output", default="data/raw/opennews_ner.jsonl")
    ap.add_argument("--start", type=int, default=0,
                    help="跳过前 N 行（断点续跑）")
    ap.add_argument("--limit", type=int, default=0,
                    help="0=处理到文件结尾")
    ap.add_argument("--limit-out", type=int, default=0,
                    help="输出达到 N 句后早停（0=不限）")
    ap.add_argument("--filter-nh-only", action="store_true",
                    help="只保存含 Nh 实体的句子")
    ap.add_argument("--input-format", choices=["jsonl", "txt"], default="jsonl",
                    help="jsonl=每行 {source,cut} / txt=每行一句")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=120,
                    help="超长句过滤防 OOM")
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument("--model", default="LTP/base1")
    ap.add_argument("--progress-every", type=int, default=200,
                    help="每 N 个 batch 打印一次")
    args = ap.parse_args()

    from ltp import LTP
    print(f"Loading {args.model} on CUDA half...", flush=True)
    try:
        ltp = LTP(args.model, local_files_only=True)
    except Exception:
        ltp = LTP(args.model)
    ltp.to("cuda"); ltp.half(); ltp.eval()
    print("Model loaded.", flush=True)

    mode = "ab" if args.start > 0 and os.path.exists(args.output) else "wb"
    fout = open(args.output, mode)

    batch = []
    n_in = 0
    n_out = 0
    n_skip = 0
    t0 = time.time()
    last_print = t0

    def flush_batch():
        nonlocal n_out
        if not batch:
            return
        srcs = [o["source"] for o in batch]
        try:
            out = ltp.pipeline(srcs, tasks=["cws", "ner"])
        except Exception as e:
            print(f"NER error on batch: {e}", file=sys.stderr, flush=True)
            return
        for obj, cwords, ner_ents in zip(batch, out.cws, out.ner):
            if args.filter_nh_only and not any(t == "Nh" for (t, *_rest) in ner_ents):
                continue
            merged = merge_nh(cwords, ner_ents)
            split_words = []
            for w in merged:
                split_words.extend(split_word(w))
            rec = {
                "source": obj["source"],
                "cut": " ".join(split_words),
                "ner": [[tag, text, sw, ew] for tag, text, sw, ew in ner_ents],
            }
            fout.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            n_out += 1

    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < args.start:
                continue
            if args.limit and (i - args.start) >= args.limit:
                break
            if args.input_format == "jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n_skip += 1
                    continue
                src = obj.get("source", "")
            else:
                src = line.rstrip("\n")
                obj = {"source": src}
            if not src or len(src) < args.min_chars or len(src) > args.max_chars:
                n_skip += 1
                continue
            batch.append(obj)
            n_in += 1
            if len(batch) >= args.batch:
                flush_batch()
                batch = []
                if args.limit_out and n_out >= args.limit_out:
                    print(f"[EARLY STOP] reached --limit-out {args.limit_out}", flush=True)
                    break
                if (n_in // args.batch) % args.progress_every == 0:
                    now = time.time()
                    elapsed = now - t0
                    rate = n_in / elapsed if elapsed > 0 else 0
                    print(f"[{i:>10}] in={n_in} out={n_out} skip={n_skip} "
                          f"rate={rate:.1f}/s",
                          flush=True)
                    fout.flush()
                    last_print = now
    flush_batch()
    fout.close()
    elapsed = time.time() - t0
    print(f"DONE. in={n_in} out={n_out} skip={n_skip} elapsed={elapsed/3600:.2f}h "
          f"→ {args.output}", flush=True)


if __name__ == "__main__":
    main()
