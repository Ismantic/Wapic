"""Simple JSONL -> BMES converter (no train/test split, no random)."""
import argparse
import json


def word_to_bmes(word):
    chars = list(word)
    if len(chars) == 1: return [(chars[0], "S")]
    out = []
    for i, c in enumerate(chars):
        if i == 0: out.append((c, "B"))
        elif i == len(chars) - 1: out.append((c, "E"))
        else: out.append((c, "M"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-bmes", required=True)
    ap.add_argument("--out-nolabel", default=None)
    ap.add_argument("--max-chars", type=int, default=0, help="0=no limit")
    ap.add_argument("--min-chars", type=int, default=2)
    args = ap.parse_args()

    n_ok = n_skip = 0
    fnl = open(args.out_nolabel, "w", encoding="utf-8") if args.out_nolabel else None
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.out_bmes, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pairs = []
            for w in obj["cut"].split():
                w = w.lstrip("[")
                if w:
                    pairs.extend(word_to_bmes(w))
            if len(pairs) < args.min_chars:
                n_skip += 1
                continue
            if args.max_chars and len(pairs) > args.max_chars:
                n_skip += 1
                continue
            for c, t in pairs:
                fout.write(f"{c} {t}\n")
            fout.write("\n")
            if fnl:
                for c, _ in pairs:
                    fnl.write(f"{c}\n")
                fnl.write("\n")
            n_ok += 1
    if fnl: fnl.close()
    print(f"ok={n_ok} skip={n_skip}")


if __name__ == "__main__":
    main()
