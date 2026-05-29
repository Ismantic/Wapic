"""
从 LTP 标注的 cut.jsonl 里提取含人名的句子。
用 LTP NER 标 source，把含 Nh 实体的句子留下，并保证 cut 已是 LTP 风格的 token。

输入: cut.jsonl (来自 cut_corpus.py 的输出，已用 LTP/base1 切好 cut 字段)
输出: 同格式 jsonl，但只保留含人名的句子（且 cut 经过 punct split）
"""

import argparse
import json
import sys
sys.path.insert(0, "scripts")
from split_punct import split_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-in", type=int, default=300000,
                    help="只扫前 N 行寻找候选")
    ap.add_argument("--limit-out", type=int, default=80000,
                    help="最多保留 N 条")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # 先收集候选
    candidates = []
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit_in:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = obj["source"]
            if len(src) < args.min_chars or len(src) > args.max_chars:
                continue
            candidates.append(obj)

    print(f"Loaded {len(candidates)} candidates", flush=True)

    # 跑 LTP NER 找含 Nh 的句子
    from ltp import LTP
    try:
        ltp = LTP("LTP/base1", local_files_only=True)
    except Exception:
        ltp = LTP("LTP/base1")
    ltp.to("cuda"); ltp.half(); ltp.eval()

    kept = []
    for s in range(0, len(candidates), args.batch):
        if len(kept) >= args.limit_out:
            break
        batch = candidates[s:s + args.batch]
        srcs = [o["source"] for o in batch]
        try:
            out = ltp.pipeline(srcs, tasks=["cws", "ner"])
        except Exception as e:
            print(f"NER error: {e}", file=sys.stderr)
            continue
        for obj, ner in zip(batch, out.ner):
            if any(tag == "Nh" for (tag, *_) in ner):
                # split punct on cut
                new_words = []
                for w in obj["cut"].split():
                    new_words.extend(split_word(w))
                kept.append({"source": obj["source"], "cut": " ".join(new_words)})
                if len(kept) >= args.limit_out:
                    break
        if (s // args.batch) % 50 == 0:
            print(f"  scanned {s + len(batch)}, kept {len(kept)}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"DONE. kept {len(kept)} name-rich sentences -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
