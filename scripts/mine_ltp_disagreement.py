"""Mine sentences where wapic CRF and LTP/base1 disagree on token boundaries.

Uses pre-computed LTP cuts from data/raw/opennews_full_nh.jsonl (3.4 GB).

Pipeline:
  1. Build dedup set (SHA1) from Stage 1 + Stage 2 training sentence text.
  2. Stream the jsonl source, skip if sentence hash in dedup_set.
  3. For each candidate, run wapic test, compare against LTP's cut from jsonl.
  4. Filter: wapic produces a >=4 char CJK token where LTP splits inside,
     AND LTP's splits in that span are all <=3 char CJK (avoids LTP errors).
  5. Emit BMES tagged with LTP's cut, retag2-normalized.

Run modes:
  --dry-run   : measure dedup overlap rate, no wapic call
  --mine      : full mining pipeline, write to output

Usage:
  python scripts/mine_ltp_disagreement.py --dry-run -n 1000000
  python scripts/mine_ltp_disagreement.py --mine -n 1000000 \\
    --wapic-model data/wapic-20260605-h25_20.wac \\
    --out data/mined_ltp_disagree.txt
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

STAGE1 = "data/all12m_train_retag2.txt"
STAGE2 = "data/h24_1.txt"
DEFAULT_SOURCE = "data/raw/opennews_full_nh.jsonl"


def stage_sentences(path):
    """Yield each sentence text from a BMES file."""
    chars = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if chars:
                    yield "".join(chars)
                    chars = []
            else:
                parts = line.split()
                if parts:
                    chars.append(parts[0])
        if chars:
            yield "".join(chars)


def sent_hash(s: str) -> bytes:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()[:16]


def build_dedup_set(stage1, stage2):
    s = set()
    t0 = time.time()
    n = 0
    for path in [stage1, stage2]:
        print(f"  hashing {path}...", file=sys.stderr, flush=True)
        for sent in stage_sentences(path):
            s.add(sent_hash(sent))
            n += 1
            if n % 1_000_000 == 0:
                print(f"    {n} sents, |set|={len(s)}, t={time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
    print(f"  done: {n} sents read, |set|={len(s)} unique, t={time.time()-t0:.0f}s",
          file=sys.stderr, flush=True)
    return s


def is_cjk(c):
    return ('一' <= c <= '鿿') or ('㐀' <= c <= '䶿')


def is_pure_cjk_token(w):
    return bool(w) and all(is_cjk(c) for c in w)


def find_long_token_disagree(wapic_words, ltp_words, min_len=4, max_ltp_inside=3):
    """Find wapic long-CJK tokens that LTP splits cleanly inside."""
    text = "".join(wapic_words)
    text_ltp = "".join(ltp_words)
    if text != text_ltp:
        return []

    out = []
    char_pos = 0
    # Build LTP token char-positions
    ltp_pos = []
    p = 0
    for w in ltp_words:
        ltp_pos.append((p, p + len(w), w))
        p += len(w)

    for w in wapic_words:
        wlen = len(w)
        if is_pure_cjk_token(w) and wlen >= min_len:
            # LTP segs fully inside [char_pos, char_pos+wlen)
            segs = [lw for (a, b, lw) in ltp_pos
                    if a >= char_pos and b <= char_pos + wlen]
            if (len(segs) >= 2
                and sum(len(s) for s in segs) == wlen
                and all(len(s) <= max_ltp_inside and is_pure_cjk_token(s) for s in segs)):
                out.append((char_pos, char_pos + wlen, w, segs))
        char_pos += wlen
    return out


def words_to_bmes(words):
    out = []
    for w in words:
        if len(w) == 1:
            out.append((w, 'S'))
        else:
            out.append((w[0], 'B'))
            for c in w[1:-1]:
                out.append((c, 'M'))
            out.append((w[-1], 'E'))
    return out


def char_type(c):
    if c.isdigit(): return 'D'
    if c.isalpha():
        if c.isascii(): return 'L'
        co = ord(c)
        if 0x4E00 <= co <= 0x9FFF: return 'C'
        if 0x3400 <= co <= 0x4DBF: return 'C'
        return 'L'
    return 'P'


def retag2_words(words):
    """Apply retag2 char-type boundary split."""
    out = []
    for w in words:
        cur = ''
        cur_t = None
        for c in w:
            t = char_type(c)
            if t == 'P':
                if cur:
                    out.append(cur); cur = ''
                cur_t = None
                out.append(c)
            else:
                if cur_t is None or t == cur_t:
                    cur += c
                    cur_t = t
                else:
                    out.append(cur); cur = c; cur_t = t
        if cur:
            out.append(cur)
    return out


def write_bmes(words, fp):
    for c, t in words_to_bmes(words):
        fp.write(f"{c} {t}\n")
    fp.write("\n")


def wapic_cut_batch(model, srcs):
    """Run wapic test on batch. Returns list aligned with srcs (empty list
    for sentences that couldn't be cut cleanly).

    Bug fix: source chars containing whitespace become blank lines in the
    wapic input, which wapic interprets as sentence boundaries, fragmenting
    the input. Skip such sentences (return [] for them).
    """
    safe_idx = []  # indices into srcs that we actually send to wapic
    safe_srcs = []
    for i, s in enumerate(srcs):
        if any(c.isspace() for c in s):
            continue
        safe_idx.append(i)
        safe_srcs.append(s)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in safe_srcs:
            for c in s:
                fin.write(c + "\n")
            fin.write("\n")
        ip = fin.name
    op = ip + ".out"
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
            if line.startswith("score="): continue
            p = line.split()
            if p and p[0] in ("B", "M", "E", "S"):
                cur.append(p[0])
    if cur:
        out_tags.append(cur)
    os.unlink(ip); os.unlink(op)

    # Sanity: did wapic emit the expected number of sentences?
    # If counts mismatch, fall back to empty for all (alignment broken).
    if len(out_tags) != len(safe_srcs):
        return [[] for _ in srcs]

    # Build aligned result indexed by original srcs order
    word_lists = [[] for _ in srcs]
    for safe_i, (s, tags) in enumerate(zip(safe_srcs, out_tags)):
        if len(s) != len(tags):
            continue  # leave as []
        words = []; cw = ""
        for c, tg in zip(s, tags):
            cw += c
            if tg in ("E", "S"):
                words.append(cw); cw = ""
        if cw:
            words.append(cw)
        word_lists[safe_idx[safe_i]] = words
    return word_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--wapic-model", default="data/wapic-20260605-h25_20.wac")
    ap.add_argument("--out", default="data/mined_ltp_disagree.txt")
    ap.add_argument("-n", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--min-cjk-len", type=int, default=4)
    ap.add_argument("--max-ltp-piece", type=int, default=3)
    args = ap.parse_args()

    if not args.dry_run and not args.mine:
        print("Specify --dry-run or --mine", file=sys.stderr)
        sys.exit(1)

    print("Building dedup set from Stage 1 + Stage 2...", file=sys.stderr)
    dedup = build_dedup_set(STAGE1, STAGE2)

    if args.dry_run:
        print(f"\n=== DRY RUN: scanning first {args.n} jsonl entries ===", file=sys.stderr)
        n_total = 0; n_dup = 0
        with open(args.source, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                except: continue
                s = d.get("source")
                if not s: continue
                n_total += 1
                if sent_hash(s) in dedup:
                    n_dup += 1
                if n_total >= args.n:
                    break
        print(f"  total scanned: {n_total}")
        print(f"  duplicates (already in Stage 1/2): {n_dup} ({100*n_dup/max(n_total,1):.1f}%)")
        print(f"  novel candidates: {n_total - n_dup} ({100*(n_total-n_dup)/max(n_total,1):.1f}%)")
        return

    # Full mining
    print(f"\n=== MINING from {args.source} ===", file=sys.stderr)
    print(f"  wapic model: {args.wapic_model}", file=sys.stderr)
    print(f"  filter: wapic CJK token >= {args.min_cjk_len}, LTP pieces <= {args.max_ltp_piece}",
          file=sys.stderr)
    print(f"  target novel candidates: {args.n}", file=sys.stderr)

    out_fp = open(args.out, "w", encoding="utf-8")
    n_read = 0; n_kept = 0; n_yield = 0
    src_batch = []; ltp_batch = []
    t0 = time.time()

    def flush():
        nonlocal n_yield
        if not src_batch:
            return
        wapic_out = wapic_cut_batch(args.wapic_model, src_batch)
        for sent, lw_str, ww in zip(src_batch, ltp_batch, wapic_out):
            if not ww:
                continue
            lw = lw_str.split()  # LTP cut field is space-separated
            disagrees = find_long_token_disagree(
                ww, lw, min_len=args.min_cjk_len, max_ltp_inside=args.max_ltp_piece)
            if not disagrees:
                continue
            # Use LTP's full cut as "correct" label, then retag2
            ltp_retag2 = retag2_words(lw)
            write_bmes(ltp_retag2, out_fp)
            n_yield += 1

    with open(args.source, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
            except: continue
            s = d.get("source"); c = d.get("cut")
            if not s or not c: continue
            n_read += 1
            if sent_hash(s) in dedup:
                continue
            n_kept += 1
            src_batch.append(s)
            ltp_batch.append(c)
            if len(src_batch) >= args.batch:
                flush()
                src_batch = []; ltp_batch = []
                if n_kept % (args.batch * 10) == 0:
                    print(f"  read={n_read} novel={n_kept} mined={n_yield} t={time.time()-t0:.0f}s",
                          file=sys.stderr, flush=True)
            if n_kept >= args.n:
                break
    flush()
    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  source read:    {n_read}")
    print(f"  novel (dedup):  {n_kept}")
    print(f"  mined yields:   {n_yield}")
    print(f"  output:         {args.out}")


if __name__ == "__main__":
    main()
