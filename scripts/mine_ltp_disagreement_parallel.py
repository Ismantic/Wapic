"""Parallel version of mine_ltp_disagreement.py.

Uses multiprocessing.Pool to run multiple wapic test subprocess in parallel.
Each worker maintains its own tempfile + wapic test invocation.

Speedup: ~N× linear with #workers (wapic test is single-thread CPU-bound).

Usage:
  python scripts/mine_ltp_disagreement_parallel.py --mine \\
    -n 1000000 -w 8 \\
    --wapic-model data/wapic-20260603-h24_2.wac \\
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
from multiprocessing import Pool

STAGE1 = "data/all12m_train_retag2.txt"
STAGE2 = "data/h24_1.txt"
DEFAULT_SOURCE = "data/raw/opennews_full_nh.jsonl"


# ----- duplicate basic utilities (avoid heavy import) ---------------------

def stage_sentences(path):
    chars = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if chars:
                    yield "".join(chars); chars = []
            else:
                p = line.split()
                if p:
                    chars.append(p[0])
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
    text = "".join(wapic_words)
    text_ltp = "".join(ltp_words)
    if text != text_ltp:
        return []
    ltp_pos = []
    p = 0
    for w in ltp_words:
        ltp_pos.append((p, p + len(w), w))
        p += len(w)
    out = []
    char_pos = 0
    for w in wapic_words:
        wlen = len(w)
        if is_pure_cjk_token(w) and wlen >= min_len:
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
    out = []
    for w in words:
        cur = ''; cur_t = None
        for c in w:
            t = char_type(c)
            if t == 'P':
                if cur: out.append(cur); cur = ''
                cur_t = None
                out.append(c)
            else:
                if cur_t is None or t == cur_t:
                    cur += c; cur_t = t
                else:
                    out.append(cur); cur = c; cur_t = t
        if cur: out.append(cur)
    return out


def bmes_text(words):
    """Return BMES-formatted string for one sentence (with trailing blank)."""
    lines = []
    for c, t in words_to_bmes(words):
        lines.append(f"{c} {t}")
    lines.append("")  # blank separator
    return "\n".join(lines) + "\n"


def wapic_cut_batch(model, srcs):
    """Run wapic test on batch, returning list aligned with srcs."""
    safe_idx = []; safe_srcs = []
    for i, s in enumerate(srcs):
        if any(c.isspace() for c in s):
            continue
        safe_idx.append(i); safe_srcs.append(s)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in safe_srcs:
            for c in s: fin.write(c + "\n")
            fin.write("\n")
        ip = fin.name
    op = ip + ".out"
    try:
        result = subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                               capture_output=True, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        try: os.unlink(ip)
        except: pass
        return [[] for _ in srcs]
    if not os.path.exists(op):
        # wapic test failed (e.g., transient resource limit). Return empty.
        try: os.unlink(ip)
        except: pass
        return [[] for _ in srcs]
    out_tags = []; cur = []
    with open(op, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if cur: out_tags.append(cur); cur = []
                continue
            if line.startswith("score="): continue
            p = line.split()
            if p and p[0] in ("B", "M", "E", "S"):
                cur.append(p[0])
    if cur: out_tags.append(cur)
    os.unlink(ip); os.unlink(op)

    if len(out_tags) != len(safe_srcs):
        return [[] for _ in srcs]
    word_lists = [[] for _ in srcs]
    for safe_i, (s, tags) in enumerate(zip(safe_srcs, out_tags)):
        if len(s) != len(tags): continue
        words = []; cw = ''
        for c, tg in zip(s, tags):
            cw += c
            if tg in ('E', 'S'):
                words.append(cw); cw = ''
        if cw: words.append(cw)
        word_lists[safe_idx[safe_i]] = words
    return word_lists


# ----- worker function ----------------------------------------------------

# Per-worker constants set via initializer
_WAPIC_MODEL = None
_MIN_LEN = 4
_MAX_LTP_PIECE = 3


def _init_worker(model, min_len, max_ltp_piece):
    global _WAPIC_MODEL, _MIN_LEN, _MAX_LTP_PIECE
    _WAPIC_MODEL = model
    _MIN_LEN = min_len
    _MAX_LTP_PIECE = max_ltp_piece


def _process_batch(args):
    """Worker: given (src_batch, ltp_batch), return list of BMES strings to write."""
    src_batch, ltp_batch = args
    wapic_out = wapic_cut_batch(_WAPIC_MODEL, src_batch)
    mined_bmes = []
    for sent, lw_str, ww in zip(src_batch, ltp_batch, wapic_out):
        if not ww: continue
        lw = lw_str.split()
        disagrees = find_long_token_disagree(ww, lw, _MIN_LEN, _MAX_LTP_PIECE)
        if not disagrees: continue
        ltp_retag2 = retag2_words(lw)
        mined_bmes.append(bmes_text(ltp_retag2))
    return mined_bmes


# ----- main ---------------------------------------------------------------

def batch_iter(source_path, dedup, batch_size, target_n):
    """Yield (src_batch, ltp_batch) tuples after dedup, up to target_n novel."""
    src_batch = []; ltp_batch = []
    n_read = 0; n_kept = 0
    for line in open(source_path, encoding="utf-8", errors="ignore"):
        if not line.strip(): continue
        try:
            d = json.loads(line)
        except: continue
        s = d.get("source"); c = d.get("cut")
        if not s or not c: continue
        n_read += 1
        if sent_hash(s) in dedup: continue
        n_kept += 1
        src_batch.append(s); ltp_batch.append(c)
        if len(src_batch) >= batch_size:
            yield (src_batch, ltp_batch), n_read, n_kept
            src_batch = []; ltp_batch = []
        if n_kept >= target_n:
            break
    if src_batch:
        yield (src_batch, ltp_batch), n_read, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true", required=True)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--wapic-model", default="data/wapic-20260603-h24_2.wac")
    ap.add_argument("--out", default="data/mined_ltp_disagree.txt")
    ap.add_argument("-n", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--min-cjk-len", type=int, default=4)
    ap.add_argument("--max-ltp-piece", type=int, default=3)
    args = ap.parse_args()

    print("Building dedup set...", file=sys.stderr)
    dedup = build_dedup_set(STAGE1, STAGE2)

    print(f"\n=== PARALLEL MINING ===", file=sys.stderr)
    print(f"  workers: {args.workers}", file=sys.stderr)
    print(f"  model:   {args.wapic_model}", file=sys.stderr)
    print(f"  target:  {args.n} novel", file=sys.stderr)
    print(f"  batch:   {args.batch}", file=sys.stderr)
    print(f"  filter:  wapic CJK >= {args.min_cjk_len}, LTP piece <= {args.max_ltp_piece}",
          file=sys.stderr)

    out_fp = open(args.out, "w", encoding="utf-8")

    n_yield = 0
    t0 = time.time()

    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.wapic_model, args.min_cjk_len, args.max_ltp_piece)) as pool:

        # Generator that yields batches; we'll feed them through imap_unordered.
        # We need to track progress; use a list of (n_read, n_kept) tuples alongside batches.
        # Simpler: collect batches first then map. But that uses memory.
        # Use a buffered streaming approach.

        # Collect batches in chunks, dispatch, collect results, repeat.
        gen = batch_iter(args.source, dedup, args.batch, args.n)

        batches_for_pool = []
        last_n_read = 0; last_n_kept = 0

        def drain_pool():
            nonlocal n_yield
            if not batches_for_pool: return
            for result_bmes_list in pool.imap_unordered(_process_batch, batches_for_pool):
                for bmes_str in result_bmes_list:
                    out_fp.write(bmes_str)
                    n_yield += 1
                out_fp.flush()
            batches_for_pool.clear()

        DISPATCH_CHUNK = args.workers * 4  # how many batches to gather before dispatch

        for batch_tuple, n_read, n_kept in gen:
            last_n_read = n_read; last_n_kept = n_kept
            batches_for_pool.append(batch_tuple)
            if len(batches_for_pool) >= DISPATCH_CHUNK:
                drain_pool()
                print(f"  read={n_read} novel={n_kept} mined={n_yield} t={time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
        drain_pool()

    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  source read:    {last_n_read}")
    print(f"  novel (dedup):  {last_n_kept}")
    print(f"  mined yields:   {n_yield}")
    print(f"  output:         {args.out}")


if __name__ == "__main__":
    main()
