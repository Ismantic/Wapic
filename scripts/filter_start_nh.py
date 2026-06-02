"""Filter NER jsonl for sentences where word index 0 is an N-char Nh entity."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--name-len", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    n_in = n_out = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for tag, text, sw, ew in obj.get("ner", []):
                if tag == "Nh" and sw == 0 and len(text) == args.name_len:
                    fout.write(line)
                    n_out += 1
                    break
            if args.limit and n_out >= args.limit:
                break
    print(f"in={n_in} out={n_out}")


if __name__ == "__main__":
    main()
