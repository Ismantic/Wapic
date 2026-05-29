"""Add character-type column to a wapic inference input file.

Input file: one character per line, blank lines separate sentences.
Output: ``char\ttype`` per line, blank lines preserved.

Type codes match scripts/jsonl_to_bmes.py: h/d/a/p/o.
"""
import argparse
import sys

sys.path.insert(0, "scripts")
from jsonl_to_bmes import char_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.rstrip("\n")
            if not s:
                fout.write("\n")
                continue
            # input may already be "c\ttype" or just "c"
            c = s.split("\t")[0] if "\t" in s else s.split()[0]
            fout.write(f"{c}\t{char_type(c)}\n")


if __name__ == "__main__":
    main()
