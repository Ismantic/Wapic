"""Build structured bad-case eval set from mined H2.3 failures.

Each case: {id, category, source, name, h23_fail_cut}
Categories are auto-assigned by heuristic for grouping/reporting.
"""
import json
import re


def categorize(name, source):
    # 1) Foreign name with dot
    if "·" in name or "・" in name:
        return "foreign_dot"
    # 2) Pure foreign transliteration (contains rare-for-Chinese-names chars or len >= 4)
    foreign_chars = set("拉夫罗斯基洛娃妮特莉亚耶娃尼克娜诺索西雅图迪马尔德")
    if len(name) >= 4 and sum(1 for c in name if c in foreign_chars) >= 2:
        return "foreign_translit"
    # 3) Name at start of sentence
    if source.startswith(name):
        return "sentence_start"
    # 4) 3-char Chinese name in mid sentence
    if len(name) == 3 and all('一' <= c <= '鿿' for c in name):
        return "mid_three_char"
    # 5) 2-char Chinese name
    if len(name) == 2:
        return "two_char"
    # default
    return "other"


def main():
    inp = "data/badcases_h23_clean.jsonl"
    outp = "data/badcase_eval.jsonl"
    by_cat = {}
    cases = []
    with open(inp) as fin:
        for i, line in enumerate(fin, 1):
            o = json.loads(line)
            cat = categorize(o["ner_name"], o["source"])
            case = {
                "id": f"BC{i:03d}",
                "category": cat,
                "source": o["source"],
                "name": o["ner_name"],
                "h23_fail_cut": o["wapic_cut"],
            }
            cases.append(case)
            by_cat.setdefault(cat, 0)
            by_cat[cat] += 1
    with open(outp, "w") as fout:
        for c in cases:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases → {outp}")
    print("\nCategory breakdown:")
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {n}")


if __name__ == "__main__":
    main()
