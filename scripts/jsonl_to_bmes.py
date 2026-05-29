"""Simple JSONL -> BMES converter (no train/test split, no random).

Output format: ``char\ttype\tlabel`` per line, blank line per sentence.
The type column lets the CRF use character-class features (h/d/a/p/o).
``--no-type`` falls back to legacy ``char\tlabel``.
"""
import argparse
import json
import unicodedata


def char_type(c):
    """h=Han, d=ASCII digit, a=ASCII letter, p=punct/symbol, o=other."""
    if '一' <= c <= '鿿':
        return 'h'
    if '豈' <= c <= '﫿':  # CJK compatibility
        return 'h'
    o = ord(c)
    if ('0' <= c <= '9') or ('０' <= c <= '９'):  # ASCII + 全角 digit
        return 'd'
    if ('A' <= c <= 'Z') or ('a' <= c <= 'z'):
        return 'a'
    if ('Ａ' <= c <= 'Ｚ') or ('ａ' <= c <= 'ｚ'):  # 全角字母
        return 'a'
    cat = unicodedata.category(c)
    if cat[0] in ('P', 'S'):
        return 'p'
    return 'o'


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
    ap.add_argument("--no-type", action="store_true",
                    help="legacy 2-col output (char\\tlabel), no type column")
    args = ap.parse_args()

    use_type = not args.no_type
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
            for c, lbl in pairs:
                if use_type:
                    fout.write(f"{c} {char_type(c)} {lbl}\n")
                else:
                    fout.write(f"{c} {lbl}\n")
            fout.write("\n")
            if fnl:
                for c, _ in pairs:
                    if use_type:
                        fnl.write(f"{c} {char_type(c)}\n")
                    else:
                        fnl.write(f"{c}\n")
                fnl.write("\n")
            n_ok += 1
    if fnl: fnl.close()
    print(f"ok={n_ok} skip={n_skip}")


if __name__ == "__main__":
    main()
