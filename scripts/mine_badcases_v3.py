"""V3 Mining: find cases where wapic CORRECTLY disagrees with LTP/base1 NER.

Correct criterion:
- For each sentence with LTP Nh entity, split name by ·/・/•/- into parts (punct is independent token by PD standard)
- Check if EACH non-empty part appears as a token in wapic cut
- If wapic FAILS to have any part as token → confirmed bad case

Single-sentence verification: each candidate is re-tested in single-sentence wapic call to filter batch artifacts.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

PUNCT_SPLIT = re.compile(r'[·・•\-]')


def wapic_single(model, src):
    chars = list(src)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as f:
        for c in chars: f.write(c + "\n")
        f.write("\n")
        ip = f.name
    op = ip + ".out"
    try:
        subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                       capture_output=True, check=False)
        tags = []
        with open(op, encoding="utf-8") as fo:
            for line in fo:
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


def wapic_batch(model, srcs):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as f:
        for s in srcs:
            for c in s: f.write(c + "\n")
            f.write("\n")
        ip = f.name
    op = ip + ".out"
    try:
        subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                       capture_output=True, check=False)
        results, cur = [], []
        with open(op, encoding="utf-8") as fo:
            for line in fo:
                line = line.rstrip()
                if not line:
                    if cur: results.append(cur); cur = []
                    continue
                if line.startswith("score="):
                    if cur: results.append(cur); cur = []
                    continue
                p = line.split()
                if p and p[0] in ("B", "M", "E", "S"):
                    cur.append(p[0])
        if cur: results.append(cur)
        out = []
        for s, tags in zip(srcs, results):
            w, cw = [], ""
            for c, t in zip(s, tags):
                if t in ("B", "S"):
                    if cw: w.append(cw)
                    cw = c
                else:
                    cw += c
            if cw: w.append(cw)
            out.append(w)
        return out
    finally:
        os.unlink(ip)
        if os.path.exists(op): os.unlink(op)


def parts_of(name):
    return [p for p in PUNCT_SPLIT.split(name) if p]


def is_clean_name(name):
    """Filter to pure Chinese characters (no punct, no foreign script)."""
    if not name: return False
    for c in name:
        if '一' <= c <= '鿿': continue
        if '豈' <= c <= '﫿': continue
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/opennews_full_nh.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit-in", type=int, default=500000)
    ap.add_argument("--limit-out", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-name-len", type=int, default=3)
    ap.add_argument("--max-name-len", type=int, default=8)
    ap.add_argument("--max-src-len", type=int, default=80)
    args = ap.parse_args()

    n_in = n_cand = n_out = 0
    fout = open(args.output, "w", encoding="utf-8")
    buf = []

    def process_buf():
        nonlocal n_cand, n_out
        if not buf: return False
        srcs = [b[0] for b in buf]
        wls = wapic_batch(args.model, srcs)
        if len(wls) != len(srcs):
            return False
        for (s, names), wl in zip(buf, wls):
            for nm in names:
                if nm not in s: continue
                parts = parts_of(nm)
                if not parts: continue
                # batch check (may have artifacts)
                if not all(p in wl for p in parts):
                    n_cand += 1
                    # SINGLE-SENTENCE VERIFY
                    verify_wl = wapic_single(args.model, s)
                    missing = [p for p in parts if p not in verify_wl]
                    if missing:
                        rec = {"source": s, "ner_name": nm, "parts": parts,
                               "missing_parts": missing,
                               "wapic_cut": " ".join(verify_wl)}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_out += 1
                        if n_out % 10 == 0:
                            print(f"  found {n_out} confirmed (cands={n_cand} scanned={n_in})", flush=True)
                        if n_out >= args.limit_out: return True
        return False

    with open(args.input, encoding="utf-8") as fin:
        for line in fin:
            n_in += 1
            if n_in > args.limit_in: break
            try: obj = json.loads(line)
            except: continue
            src = obj.get("source", "")
            if not src or len(src) > args.max_src_len or " " in src: continue
            ner = obj.get("ner", [])
            names = [text for (tag, text, *_) in ner
                     if tag == "Nh" and args.min_name_len <= len(text) <= args.max_name_len
                     and is_clean_name(text)]
            if not names: continue
            buf.append((src, names))
            if len(buf) >= args.batch:
                done = process_buf()
                buf = []
                if done: break
                if n_in % 1000 < args.batch:
                    print(f"  scanned {n_in}  candidates {n_cand}  confirmed {n_out}", flush=True)

    if buf and n_out < args.limit_out:
        process_buf()

    fout.close()
    print(f"DONE. scanned {n_in}  candidates {n_cand}  confirmed {n_out} → {args.output}", flush=True)


if __name__ == "__main__":
    main()
