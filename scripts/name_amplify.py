"""
从 cut.jsonl 抽出常见人名，造模板句强化"姓名整体"模式。
输出: cut.jsonl 格式的合成数据。

模板里 {N} 是人名，{N0} 是别的人名，{X1}/{X2} 是占位中文。
所有标点已分开。
"""

import argparse
import json
import random
import sys
from collections import Counter

TEMPLATES = [
    "{N} 表示 ， 这 是 一 个 好 消息 。",
    "{N} 在 现场 接受 了 采访 。",
    "据 {N} 介绍 ， 项目 进展 顺利 。",
    "记者 {N} 报道",
    "记者 {N} 从 北京 发回 报道",
    "{N} 说 ， 我 完全 同意 这个 看法 。",
    "{N} 表示 支持 ， 并 提出 建议 。",
    "{N} 出席 会议 并 发表 讲话 。",
    "{N} 等 一 行 人 抵达 现场 。",
    "{N} 与 {N0} 共同 出席 仪式 。",
    "{N} 担任 该 公司 的 总经理 。",
    "{N} 获得 比赛 冠军 。",
    "{N} 来 自 北京 ， 今年 30 岁 。",
    "{N} 是 这 部 电影 的 导演 。",
    "{N} 接受 了 本报 记者 的 采访 。",
    "{N} 强调 ， 必须 加强 监管 。",
    "{N} 主持 了 今天 的 会议 。",
    "{N} 透露 ， 双方 已 达成 一致 。",
    "在 现场 ， {N} 与 群众 亲切 交流 。",
    "本报 讯 记者 {N} 报道 。",
    "在 这 件 事 上 ， {N} 的 看法 是 这样 的 。",
    "{N} 称 ， 这 是 一 次 重要 的 尝试 。",
    "{N} 任 该 项目 的 总 负责人 。",
    "中央 领导 {N} 同志 发表 讲话 。",
    "据 悉 ， {N} 已 抵达 北京 。",
    "{N} 是 一 名 优秀 的 运动员 。",
    "{N} 的 新书 即将 出版 。",
    "{N} 是 著名 的 学者 。",
]


def looks_chinese_name(s):
    """简单判定：2-3 个汉字。"""
    if not (2 <= len(s) <= 3):
        return False
    for c in s:
        if not ('一' <= c <= '鿿'):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="cut.jsonl with LTP labels")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-in", type=int, default=300000)
    ap.add_argument("--min-freq", type=int, default=2,
                    help="人名在原数据中至少出现 min_freq 次才用")
    ap.add_argument("--per-name", type=int, default=8,
                    help="每个名字用多少个模板")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 1. 收集候选句子 + 用 LTP NER 抽人名
    print(f"Scanning {args.limit_in} sentences for person names ...", flush=True)
    from ltp import LTP
    try:
        ltp = LTP("LTP/base1", local_files_only=True)
    except Exception:
        ltp = LTP("LTP/base1")
    ltp.to("cuda"); ltp.half(); ltp.eval()

    name_counter = Counter()
    batch_src = []
    n_scanned = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            n_scanned += 1
            if n_scanned > args.limit_in:
                break
            try:
                src = json.loads(line)["source"]
            except Exception:
                continue
            batch_src.append(src)
            if len(batch_src) >= 64:
                try:
                    out = ltp.pipeline(batch_src, tasks=["cws", "ner"])
                    for ner in out.ner:
                        for tag, text, ws, we in ner:
                            if tag == "Nh" and looks_chinese_name(text):
                                name_counter[text] += 1
                except Exception as e:
                    print(f"NER err: {e}", file=sys.stderr)
                batch_src = []
                if n_scanned % 20000 == 0:
                    print(f"  scanned {n_scanned}, unique names: {len(name_counter)}",
                          flush=True)
    # tail
    if batch_src:
        out = ltp.pipeline(batch_src, tasks=["cws", "ner"])
        for ner in out.ner:
            for tag, text, ws, we in ner:
                if tag == "Nh" and looks_chinese_name(text):
                    name_counter[text] += 1

    names = [n for n, c in name_counter.items() if c >= args.min_freq]
    print(f"Found {len(names)} unique names (freq >= {args.min_freq})", flush=True)

    # 2. 用模板造句
    print("Generating template sentences ...", flush=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for name in names:
            tmpls = random.sample(TEMPLATES, min(args.per_name, len(TEMPLATES)))
            for t in tmpls:
                # 需要的话提供第二个名字
                other = random.choice(names) if "{N0}" in t else ""
                while other == name and "{N0}" in t:
                    other = random.choice(names)
                cut = t.format(N=name, N0=other)
                src = cut.replace(" ", "")
                fout.write(json.dumps({"source": src, "cut": cut},
                                      ensure_ascii=False) + "\n")
    print(f"DONE. wrote {len(names) * args.per_name} synthetic sentences -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
