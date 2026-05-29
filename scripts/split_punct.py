"""
强制把所有标点符号当作独立 token。
对 JSONL 的 cut 字段做处理，source 不变。

输入: {"source": "...", "cut": "13:10 ..."}
输出: {"source": "...", "cut": "13 : 10 ..."}
"""

import argparse
import json
import unicodedata


# 半角标点 + 常见数学/标记符号
ASCII_PUNCT = set(r"""/:;.,()[]{}<>"'`-_=+*&|\?!#@$%^~""")

def is_punct(c):
    if c in ASCII_PUNCT:
        return True
    cat = unicodedata.category(c)
    # P* = 各种标点；S* = symbol（不一定都拆，但 / 等会落到 So）
    if cat[0] == "P":
        return True
    if cat in ("Sm",):  # 数学符号如 ＝
        return True
    return False


def split_word(w):
    out = []
    cur = ""
    for c in w:
        if is_punct(c):
            if cur:
                out.append(cur)
                cur = ""
            out.append(c)
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    n_changed = 0
    n_total = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_words = []
            for w in obj["cut"].split():
                parts = split_word(w)
                new_words.extend(parts)
            new_cut = " ".join(new_words)
            if new_cut != obj["cut"]:
                n_changed += 1
            n_total += 1
            obj["cut"] = new_cut
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"processed {n_total}, modified {n_changed}")


if __name__ == "__main__":
    main()
