"""Real-world tokenization bad-case tracker.

For each known bug pattern, capture the expected vs actual segmentation.
This is a regression / progress tracker — not a strict pass/fail test.

Usage:
    python scripts/test_badcase.py --model data/exp_h2_3_*.wac
"""
import argparse
import subprocess
import tempfile
import os


# Each case: (source, expected_words_loose_pattern_to_check)
# expected_words = list of substrings that should appear as separate tokens
# i.e. "波·索提拉克" should be one token, "部长" should be one token
BAD_CASES = [
    {
        "id": "BC001_minister_foreign_name",
        "source": "对话会期间，柬埔寨区域研究中心高级顾问、柬埔寨前工业、矿业和能源部长波·索提拉克在接受采访时表示。",
        # By PD-1998 standard, · is independent token, so parts of name should each be tokens.
        "must_keep_together": ["部长", "波", "索提拉克"],
        "must_not_token": ["长波", "能源部长"],
        "notes": "部长 + 外国人名（音节·音节）混合 boundary",
    },
]


def cut_with(model, source):
    chars = list(source)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for c in chars:
            fin.write(c + "\n")
        fin.write("\n")
        in_path = fin.name
    out_path = in_path + ".out"
    try:
        subprocess.run(
            ["./build/wapic", "test", "-m", model, in_path, out_path],
            capture_output=True, check=False
        )
        # wapic test output format: "tag score" per line, with a "score=X" header line.
        tags = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line: break
                if line.startswith("score="): continue
                parts = line.split()
                if parts and parts[0] in ("B", "M", "E", "S"):
                    tags.append(parts[0])
        # tags to words
        words, cur = [], ""
        for c, t in zip(chars, tags):
            if t in ("B", "S"):
                if cur: words.append(cur)
                cur = c
            else:
                cur += c
        if cur: words.append(cur)
        return words
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path): os.unlink(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    n_ok = 0
    for case in BAD_CASES:
        words = cut_with(args.model, case["source"])
        cut_str = " ".join(words)

        keep_ok = all(tok in words for tok in case["must_keep_together"])
        not_token_ok = all(tok not in words for tok in case["must_not_token"])
        ok = keep_ok and not_token_ok

        status = "✓" if ok else "✗"
        print(f"{status} {case['id']}: {case['notes']}")
        print(f"  src: {case['source']}")
        print(f"  cut: {cut_str}")
        if not keep_ok:
            missing = [t for t in case['must_keep_together'] if t not in words]
            print(f"  ✗ missing tokens: {missing}")
        if not not_token_ok:
            bad = [t for t in case['must_not_token'] if t in words]
            print(f"  ✗ unwanted tokens: {bad}")
        print()
        if ok: n_ok += 1
    print(f"=== TOTAL: {n_ok}/{len(BAD_CASES)} ===")


if __name__ == "__main__":
    main()
