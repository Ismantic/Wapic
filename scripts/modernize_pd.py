"""
用 LTP/base1 的 NER 修正 PD-1998 分词，仅做「合并」，永不拆分。

输入: PD-1998 JSONL (source + cut)
输出: 同格式 JSONL，但 cut 已现代化

规则:
  对每条句子，先用 LTP 做 NER (cws + ner)
  若一个 NER 实体 (人名/地名/机构名) 在原文中的字符范围
  完整覆盖了 PD 切分中的 N 个连续 token 且字符边界吻合，
  就把这 N 个 token 合并成 1 个。

例:
  原: 李瑞环今天回到北京
  PD cut: 李 瑞环 今天 回到 北京
  LTP NER: Nh(李瑞环, 0-2), Ns(北京, 7-8)
  → 合并: 李瑞环 今天 回到 北京
"""

import argparse
import json
import sys
from pathlib import Path


def merge_pd_with_ner(pd_words, ner_entities):
    """
    pd_words: ['李', '瑞环', '今天', '回到', '北京']
    ner_entities: [('Nh', '李瑞环', 0, 2), ('Ns', '北京', 7, 8)]
                  (tag, text, start_char_inclusive, end_char_inclusive)
    返回: 合并后的 token 列表 ['李瑞环', '今天', '回到', '北京']
    """
    # 计算每个 PD token 的字符位置
    pos = 0
    pd_spans = []  # (start, end_inclusive)
    for w in pd_words:
        pd_spans.append((pos, pos + len(w) - 1))
        pos += len(w)
    pd_end_char = pos

    # 把 ner_entities 转成 char-range set，按 start 排序
    merges = []  # list of (token_start_idx, token_end_idx_inclusive)
    for tag, text, ent_start, ent_end in ner_entities:
        if ent_start < 0 or ent_end >= pd_end_char:
            continue
        # 找哪些 PD token 完整落在这个区间
        first_tok = None
        last_tok = None
        for i, (s, e) in enumerate(pd_spans):
            if s >= ent_start and e <= ent_end:
                if first_tok is None:
                    first_tok = i
                last_tok = i
            elif s > ent_end:
                break
        if first_tok is None or last_tok is None or last_tok == first_tok:
            continue
        # 边界必须吻合：第一个 token 起始 == 实体起始，最后一个 token 终点 == 实体终点
        if pd_spans[first_tok][0] != ent_start: continue
        if pd_spans[last_tok][1] != ent_end: continue
        merges.append((first_tok, last_tok))

    # 应用合并
    if not merges:
        return pd_words[:]

    merges.sort()
    out = []
    i = 0
    midx = 0
    while i < len(pd_words):
        if midx < len(merges) and merges[midx][0] == i:
            s, e = merges[midx]
            out.append("".join(pd_words[s:e+1]))
            i = e + 1
            midx += 1
        else:
            out.append(pd_words[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="PD JSONL 输入")
    ap.add_argument("--output", required=True, help="modernized JSONL 输出")
    ap.add_argument("--ltp-model", default="LTP/base1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ner-tags", default="Nh,Ni,Ns",
                    help="哪些 NER 标签触发合并")
    args = ap.parse_args()
    keep_tags = set(args.ner_tags.split(","))

    print(f"Loading LTP {args.ltp_model} ...", flush=True)
    from ltp import LTP
    try:
        ltp = LTP(args.ltp_model, local_files_only=True)
    except Exception:
        ltp = LTP(args.ltp_model)
    ltp.to("cuda"); ltp.half(); ltp.eval()

    # 1) 读 PD JSONL
    sources = []
    pd_cuts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sources.append(obj["source"])
            pd_cuts.append(obj["cut"].split())

    print(f"Loaded {len(sources)} sentences", flush=True)

    # 2) 跑 LTP NER
    n_merge_sentences = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for batch_start in range(0, len(sources), args.batch):
            batch_srcs = sources[batch_start:batch_start + args.batch]
            batch_pds = pd_cuts[batch_start:batch_start + args.batch]
            try:
                out = ltp.pipeline(batch_srcs, tasks=["cws", "ner"])
                cws_batch = out.cws
                ner_batch = out.ner
            except Exception as e:
                print(f"NER error at batch {batch_start}: {e}", file=sys.stderr)
                cws_batch = [[] for _ in batch_srcs]
                ner_batch = [[] for _ in batch_srcs]
            for src, pd_w, words, ner in zip(batch_srcs, batch_pds, cws_batch, ner_batch):
                # words: LTP 的分词
                # ner: list of (tag, text, word_start, word_end)  -- LTP 的 word 索引！
                # 计算每个 LTP word 的 char 范围
                p = 0
                word_char = []
                for w in words:
                    word_char.append((p, p + len(w) - 1))
                    p += len(w)

                ents_char = []
                for tag, text, ws, we in ner:
                    if tag not in keep_tags:
                        continue
                    if ws < 0 or we >= len(word_char):
                        continue
                    cs = word_char[ws][0]
                    ce = word_char[we][1]
                    ents_char.append((tag, text, cs, ce))

                merged = merge_pd_with_ner(pd_w, ents_char)
                if merged != pd_w:
                    n_merge_sentences += 1
                fout.write(json.dumps(
                    {"source": src, "cut": " ".join(merged)},
                    ensure_ascii=False) + "\n")

            if (batch_start // args.batch) % 50 == 0:
                print(f"  processed {batch_start + len(batch_srcs)}/{len(sources)} "
                      f"(merged in {n_merge_sentences} sentences)", flush=True)

    print(f"DONE. Merged in {n_merge_sentences}/{len(sources)} sentences", flush=True)


if __name__ == "__main__":
    main()
