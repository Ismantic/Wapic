"""
Sentence-level random sample from BMES file (each sentence = char-line block + blank line).
Usage: python sample_sentences.py --input src.txt --output dst.txt --n 100000 --seed 42
"""
import argparse
import random
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # First pass: collect sentence byte offsets
    sents = []  # list of (start_offset, end_offset)
    start = 0
    pos = 0
    in_sent = False
    with open(args.input, "rb") as f:
        for line in f:
            if line.strip() == b"":
                if in_sent:
                    sents.append((start, pos))
                    in_sent = False
                pos += len(line)
                start = pos
            else:
                in_sent = True
                pos += len(line)
        if in_sent:
            sents.append((start, pos))
    print(f"Found {len(sents)} sentences in {args.input}", file=sys.stderr, flush=True)

    if args.n >= len(sents):
        idx = list(range(len(sents)))
    else:
        random.seed(args.seed)
        idx = random.sample(range(len(sents)), args.n)
    idx.sort()

    with open(args.input, "rb") as fin, open(args.output, "wb") as fout:
        for s, e in (sents[i] for i in idx):
            fin.seek(s)
            fout.write(fin.read(e - s))
            fout.write(b"\n")  # blank sep
    print(f"Wrote {len(idx)} sentences to {args.output}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
