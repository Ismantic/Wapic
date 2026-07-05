"""Quick person-name consistency test.

For each (name, sentences) pair, runs wapic on each sentence and reports
whether the name was kept as a single token.

Usage:
    python3 scripts/test_name_cases.py --model data/model/wapic-20260605.wac
"""
import argparse
import subprocess
import tempfile
import os


CASES = [
    ("李镇全", [
        "李镇全是著名的学者。",
        "据李镇全介绍，项目进展顺利。",
        "李镇全担任该公司的总经理。",
        "记者李镇全报道。",
        "中央领导李镇全同志发表讲话。",
    ]),
    ("林强峰", [
        "缉毒警林强峰牺牲时大家都很悲伤。",
        "林强峰是这部电视剧的主角。",
        "据林强峰介绍，毒贩已经落网。",
        "林强峰与同事配合默契。",
        "记者采访了林强峰的家人。",
    ]),
    ("张子健", [
        "张子健今天接受了采访。",
        "据张子健介绍，事情进展顺利。",
        "张子健担任公司总经理。",
        "记者张子健报道。",
        "张子健与同事一起加班。",
    ]),
]


def tags_to_words(chars, tags):
    words = []
    cur = ""
    for c, t in zip(chars, tags):
        if t in ("B", "S"):
            if cur: words.append(cur)
            cur = c
        else:
            cur += c
    if cur: words.append(cur)
    return words


def run_wapic(model, sentences, wapic_bin="./build/wapic"):
    """Return list of segmented sentences (list of words)."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as f:
        in_path = f.name
        for s in sentences:
            for ch in s:
                f.write(ch + "\n")
            f.write("\n")
    out_path = in_path + ".out"
    try:
        subprocess.run(
            [wapic_bin, "test", "-m", model, in_path, out_path],
            check=True,
            capture_output=True,
        )
        results = []
        cur_tags = []
        sent_idx = 0
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("score="):
                    continue
                if not line:
                    if cur_tags:
                        results.append(tags_to_words(sentences[sent_idx], cur_tags))
                        sent_idx += 1
                        cur_tags = []
                    continue
                cur_tags.append(line.split()[0])
        if cur_tags:
            results.append(tags_to_words(sentences[sent_idx], cur_tags))
        return results
    finally:
        for path in (in_path, out_path):
            if os.path.exists(path):
                os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wapic", default="./build/wapic")
    args = ap.parse_args()

    total = ok = 0
    for name, sents in CASES:
        print(f"\n=== {name} ({len(sents)} sentences) ===")
        results = run_wapic(args.model, sents, args.wapic)
        name_ok = 0
        for sent, words in zip(sents, results):
            kept = name in words
            mark = "✓" if kept else "✗"
            print(f"  {mark} {sent}")
            print(f"     → {' '.join(words)}")
            if kept:
                name_ok += 1
                ok += 1
            total += 1
        print(f"  --- {name}: {name_ok}/{len(sents)} kept ---")
    print(f"\n=== TOTAL: {ok}/{total} = {ok*100/total:.1f}% ===")


if __name__ == "__main__":
    main()
