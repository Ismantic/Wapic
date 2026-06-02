"""Mine real H2.3 vs LTP-NER disagreements with single-sentence verification.

Batch process for speed, but VERIFY each candidate failure with single-sentence wapic call.
Only keeps cases that fail in single-sentence mode (true H2.3 errors).
"""
import argparse
import json
import os
import subprocess
import tempfile


def wapic_cut_single(model, src):
    chars = list(src)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for c in chars:
            fin.write(c + "\n")
        fin.write("\n")
        ip = fin.name
    op = ip + ".out"
    try:
        subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                       capture_output=True, check=False)
        tags = []
        with open(op, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line: break
                if line.startswith("score="): continue
                p = line.split()
                if p and p[0] in ("B", "M", "E", "S"):
                    tags.append(p[0])
        w, cw = [], ""
        for c, t in zip(chars, tags):
            if t in ("B", "S"):
                if cw: w.append(cw)
                cw = c
            else:
                cw += c
        if cw: w.append(cw)
        return w
    finally:
        os.unlink(ip)
        if os.path.exists(op): os.unlink(op)


def wapic_cut_batch(model, srcs):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in srcs:
            for c in s:
                fin.write(c + "\n")
            fin.write("\n")
        ip = fin.name
    op = ip + ".out"
    try:
        subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                       capture_output=True, check=False)
        out_tags = []
        cur = []
        with open(op, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    if cur:
                        out_tags.append(cur); cur = []
                    continue
                if line.startswith("score="):
                    if cur:
                        out_tags.append(cur); cur = []
                    continue
                p = line.split()
                if p and p[0] in ("B", "M", "E", "S"):
                    cur.append(p[0])
        if cur: out_tags.append(cur)
        word_lists = []
        for s, tags in zip(srcs, out_tags):
            w, cw = [], ""
            for c, t in zip(s, tags):
                if t in ("B", "S"):
                    if cw: w.append(cw)
                    cw = c
                else:
                    cw += c
            if cw: w.append(cw)
            word_lists.append(w)
        return word_lists
    finally:
        os.unlink(ip)
        if os.path.exists(op): os.unlink(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/opennews_full_nh.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-in", type=int, default=500000)
    ap.add_argument("--limit-out", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-name-len", type=int, default=3)
    ap.add_argument("--max-name-len", type=int, default=6)
    ap.add_argument("--max-src-len", type=int, default=80)
    args = ap.parse_args()

    n_in = n_cand = n_out = 0
    fout = open(args.output, "w", encoding="utf-8")
    buf = []
    print(f"mining with --batch {args.batch}, single-sentence verify each candidate", flush=True)

    with open(args.input, encoding="utf-8") as fin:
        for line in fin:
            n_in += 1
            if n_in > args.limit_in: break
            try: obj = json.loads(line)
            except: continue
            src = obj.get("source", "")
            if not src or len(src) > args.max_src_len: continue
            if " " in src: continue  # skip sentences with spaces (parse hazard)
            ner = obj.get("ner", [])
            names = [text for (tag, text, *_) in ner
                     if tag == "Nh" and args.min_name_len <= len(text) <= args.max_name_len]
            if not names: continue
            buf.append((src, names))
            if len(buf) >= args.batch:
                srcs = [b[0] for b in buf]
                word_lists = wapic_cut_batch(args.model, srcs)
                if len(word_lists) != len(srcs):
                    # batch mismatch — skip this batch
                    buf = []
                    continue
                for (s, names), words in zip(buf, word_lists):
                    for nm in names:
                        if nm not in s: continue
                        if nm not in words:
                            n_cand += 1
                            # VERIFY single sentence
                            verify_words = wapic_cut_single(args.model, s)
                            if nm not in verify_words:
                                # confirmed bad case
                                rec = {"source": s, "ner_name": nm,
                                       "cut": " ".join(verify_words)}
                                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                n_out += 1
                                if n_out >= args.limit_out: break
                    if n_out >= args.limit_out: break
                buf = []
                if n_out >= args.limit_out: break
                if n_in % 10000 < args.batch:
                    print(f"  scanned {n_in}  cands {n_cand}  confirmed {n_out}", flush=True)

    fout.close()
    print(f"DONE. scanned {n_in}  candidates {n_cand}  confirmed {n_out} → {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
