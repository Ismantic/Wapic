"""
用人名列表快速过滤包含人名的句子（不用 LTP）。
然后再用 LTP 对这些句子做精确 cws+ner，输出 cut.jsonl。
"""

import argparse
import json
import sys
sys.path.insert(0, "scripts")
from split_punct import split_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="人名列表 txt，一行一个")
    ap.add_argument("--input", required=True, help="原始文本 txt 每行一句")
    ap.add_argument("--output", required=True, help="输出 cut.jsonl")
    ap.add_argument("--limit-in", type=int, default=2000000)
    ap.add_argument("--limit-out", type=int, default=200000)
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=80)
    ap.add_argument("--name-min-len", type=int, default=2)
    ap.add_argument("--name-max-len", type=int, default=3)
    args = ap.parse_args()

    print("Loading names ...", flush=True)
    names = set()
    with open(args.names, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if args.name_min_len <= len(n) <= args.name_max_len:
                names.add(n)
    print(f"Loaded {len(names)} names of length [{args.name_min_len}, {args.name_max_len}]",
          flush=True)

    print(f"Scanning {args.input} (up to {args.limit_in} lines) ...", flush=True)
    matched = []
    n_scanned = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            n_scanned += 1
            if n_scanned > args.limit_in:
                break
            s = line.strip()
            if not (args.min_chars <= len(s) <= args.max_chars):
                continue
            # 检查 2-char 和 3-char 子串是否在 names 中
            found = False
            for L in range(args.name_min_len, args.name_max_len + 1):
                for i in range(len(s) - L + 1):
                    sub = s[i:i + L]
                    if sub in names:
                        found = True
                        break
                if found:
                    break
            if found:
                matched.append(s)
                if len(matched) >= args.limit_out:
                    break
            if n_scanned % 100000 == 0:
                print(f"  scanned {n_scanned}, matched {len(matched)}", flush=True)

    print(f"Total matched: {len(matched)}", flush=True)

    # 对 matched 跑 LTP cws → 切词
    print("Loading LTP ...", flush=True)
    from ltp import LTP
    try:
        ltp = LTP("LTP/base1", local_files_only=True)
    except Exception:
        ltp = LTP("LTP/base1")
    ltp.to("cuda"); ltp.half(); ltp.eval()

    print("Cutting matched sentences ...", flush=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for s in range(0, len(matched), 64):
            batch = matched[s:s + 64]
            try:
                out = ltp.pipeline(batch, tasks=["cws"])
                for src, words in zip(batch, out.cws):
                    # split_punct on every word
                    new_words = []
                    for w in words:
                        new_words.extend(split_word(w))
                    fout.write(json.dumps({"source": src, "cut": " ".join(new_words)},
                                          ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"LTP err: {e}", file=sys.stderr)
            if (s // 64) % 100 == 0:
                print(f"  cut {s + len(batch)}/{len(matched)}", flush=True)

    print(f"DONE -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
