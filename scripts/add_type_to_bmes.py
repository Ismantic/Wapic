"""In-place augment legacy BMES files (char\tlabel) with type column.

Reads ``char\tlabel`` (legacy) or ``char\ttype\tlabel`` (already typed).
Writes ``char\ttype\tlabel`` either way; existing type column is recomputed
to ensure consistency.
"""
import argparse
import sys

sys.path.insert(0, "scripts")
from jsonl_to_bmes import char_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-label", action="store_true",
                    help="input is char-only (e.g. test_nolabel), output is char\\ttype")
    args = ap.parse_args()

    n_lines = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.rstrip("\n")
            if not s:
                fout.write("\n")
                continue
            parts = s.split()
            if not parts:
                fout.write("\n")
                continue
            c = parts[0]
            t = char_type(c)
            if args.no_label:
                fout.write(f"{c} {t}\n")
            else:
                lbl = parts[-1]
                fout.write(f"{c} {t} {lbl}\n")
            n_lines += 1
    print(f"wrote {n_lines} non-blank lines -> {args.output}")


if __name__ == "__main__":
    main()
