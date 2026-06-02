"""Mine real-world person-name bad cases by comparing wapic vs LTP NER ground truth.

Strategy:
- Read opennews_full_nh.jsonl (LTP cut + NER Nh entities)
- Run wapic on each source sentence
- For each Nh entity that LTP cut kept as 1 word: check if wapic also kept it as 1 word
- If wapic SPLIT it: record as a bad case

Outputs jsonl: {source, ner_name, wapic_cut, ltp_cut}
"""
import argparse
import json
import os
import subprocess
import tempfile


def cut_with_wapic_batch(model, sentences):
    """Cut N sentences in one wapic call. Returns list of word lists."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in sentences:
            for c in s:
                fin.write(c + "\n")
            fin.write("\n")
        in_path = fin.name
    out_path = in_path + ".out"
    try:
        subprocess.run(
            ["./build/wapic", "test", "-m", model, in_path, out_path],
            capture_output=True, check=False
        )
        # Parse: tag + score per char, blank line between sentences
        results = []
        cur_tags = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    if cur_tags:
                        results.append(cur_tags)
                        cur_tags = []
                    continue
                if line.startswith("score="): continue
                parts = line.split()
                if parts and parts[0] in ("B", "M", "E", "S"):
                    cur_tags.append(parts[0])
        if cur_tags:
            results.append(cur_tags)
        # Convert tags to words for each
        word_lists = []
        for s, tags in zip(sentences, results):
            words = []
            cur = ""
            for c, t in zip(s, tags):
                if t in ("B", "S"):
                    if cur: words.append(cur)
                    cur = c
                else:
                    cur += c
            if cur: words.append(cur)
            word_lists.append(words)
        return word_lists
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path): os.unlink(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/opennews_full_nh.jsonl")
    ap.add_argument("--model", default="data/wapic-20260529.wac")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-in", type=int, default=100000)
    ap.add_argument("--limit-out", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--min-name-len", type=int, default=3)
    ap.add_argument("--max-name-len", type=int, default=4)
    ap.add_argument("--max-src-len", type=int, default=100)
    args = ap.parse_args()

    n_in = n_out = 0
    fout = open(args.output, "w", encoding="utf-8")
    buf = []  # batch of (source, ner_names)
    with open(args.input, encoding="utf-8") as fin:
        for line in fin:
            n_in += 1
            if n_in > args.limit_in: break
            try: obj = json.loads(line)
            except: continue
            src = obj.get("source", "")
            if not src or len(src) > args.max_src_len: continue
            ner = obj.get("ner", [])
            names = [text for (tag, text, *_) in ner
                     if tag == "Nh" and args.min_name_len <= len(text) <= args.max_name_len]
            if not names: continue
            buf.append((src, names))
            if len(buf) >= args.batch:
                # process batch
                srcs = [b[0] for b in buf]
                word_lists = cut_with_wapic_batch(args.model, srcs)
                for (s, names), words in zip(buf, word_lists):
                    for nm in names:
                        if nm in s and nm not in words:
                            # wapic split this name — bad case!
                            rec = {"source": s, "ner_name": nm, "wapic_cut": " ".join(words)}
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            n_out += 1
                            if n_out >= args.limit_out: break
                    if n_out >= args.limit_out: break
                buf = []
                if n_out >= args.limit_out: break
                if n_in % 5000 < args.batch:
                    print(f"  scanned {n_in}, found {n_out} bad cases", flush=True)

    # final batch
    if buf and n_out < args.limit_out:
        srcs = [b[0] for b in buf]
        word_lists = cut_with_wapic_batch(args.model, srcs)
        for (s, names), words in zip(buf, word_lists):
            for nm in names:
                if nm in s and nm not in words:
                    rec = {"source": s, "ner_name": nm, "wapic_cut": " ".join(words)}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1

    fout.close()
    print(f"DONE. scanned {n_in}, found {n_out} bad cases → {args.output}", flush=True)


if __name__ == "__main__":
    main()
