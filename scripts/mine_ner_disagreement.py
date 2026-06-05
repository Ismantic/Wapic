"""NER-guided mining: find sentences where wapic splits/merges LTP's Nh (person)
entities incorrectly.

Uses LTP's pre-computed NER tags from opennews_full_nh.jsonl which has:
  {"source": "...", "cut": "tok1 tok2 ...", "ner": [["Nh", "罗愕莹", 3, 3], ...]}
where the integer indices are LTP word indices (0-based) into the cut.

Filter logic:
  For each NER entity (default: Nh person), compute its char range in source.
  Run wapic on source. Check if wapic produces the entity as a single token
  at the matching char range.
    - If yes: agreement, skip
    - If no: disagreement (wapic over-splits or over-merges) → mine this case

Output: BMES with LTP's cut, retag2-normalized (LTP cut treated as gold).

Usage:
  python scripts/mine_ner_disagreement.py --mine \\
    -n 2000000 -w 12 \\
    --entity-types Nh \\
    --wapic-model data/wapic-20260603-h24_2.wac \\
    --out data/mined_ner_disagree.txt
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


# ----- duplicate shared utilities ----------------------------------------

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
                if p: chars.append(p[0])
        if chars:
            yield "".join(chars)


def sent_hash(s: str) -> bytes:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()[:16]


def build_dedup_set(stage1, stage2):
    s = set()
    t0 = time.time(); n = 0
    for path in [stage1, stage2]:
        print(f"  hashing {path}...", file=sys.stderr, flush=True)
        for sent in stage_sentences(path):
            s.add(sent_hash(sent)); n += 1
            if n % 1_000_000 == 0:
                print(f"    {n} sents, |set|={len(s)}, t={time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
    print(f"  done: {n} sents, |set|={len(s)}, t={time.time()-t0:.0f}s",
          file=sys.stderr, flush=True)
    return s


def words_to_bmes(words):
    out = []
    for w in words:
        if len(w) == 1: out.append((w, 'S'))
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
                cur_t = None; out.append(c)
            else:
                if cur_t is None or t == cur_t:
                    cur += c; cur_t = t
                else:
                    out.append(cur); cur = c; cur_t = t
        if cur: out.append(cur)
    return out


def bmes_text(words):
    lines = []
    for c, t in words_to_bmes(words):
        lines.append(f"{c} {t}")
    lines.append("")
    return "\n".join(lines) + "\n"


def wapic_cut_batch(model, srcs):
    safe_idx = []; safe_srcs = []
    for i, s in enumerate(srcs):
        if any(c.isspace() for c in s): continue
        safe_idx.append(i); safe_srcs.append(s)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".in") as fin:
        for s in safe_srcs:
            for c in s: fin.write(c + "\n")
            fin.write("\n")
        ip = fin.name
    op = ip + ".out"
    try:
        subprocess.run(["./build/wapic", "test", "-m", model, ip, op],
                       capture_output=True, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        try: os.unlink(ip)
        except: pass
        return [[] for _ in srcs]
    if not os.path.exists(op):
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
            if p and p[0] in ("B","M","E","S"):
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
            if tg in ('E','S'):
                words.append(cw); cw = ''
        if cw: words.append(cw)
        word_lists[safe_idx[safe_i]] = words
    return word_lists


# ----- NER-specific filter -----------------------------------------------

def ner_entity_char_range(ltp_words, start_word_idx, end_word_idx):
    """Given LTP cut + entity word indices, return char [start, end) in source."""
    pos = 0
    for i, w in enumerate(ltp_words):
        if i == start_word_idx:
            char_start = pos
        if i == end_word_idx:
            char_end = pos + len(w)
            return char_start, char_end
        pos += len(w)
    return None


def wapic_has_token_at(wapic_words, char_start, char_end):
    """Return True if wapic_words has exactly one token covering [char_start, char_end)."""
    pos = 0
    for w in wapic_words:
        wlen = len(w)
        if pos == char_start and pos + wlen == char_end:
            return True
        pos += wlen
    return False


def find_ner_disagreements(wapic_words, ltp_words, ner_entities, entity_types,
                            min_entity_len=2):
    """Find NER entities where wapic doesn't match. Returns list of (char_start,
    char_end, entity_text, entity_type)."""
    text_w = "".join(wapic_words)
    text_l = "".join(ltp_words)
    if text_w != text_l:
        return []
    out = []
    for ent in ner_entities:
        if not isinstance(ent, list) or len(ent) < 4:
            continue
        tag, entity_text, start_idx, end_idx = ent[0], ent[1], ent[2], ent[3]
        if tag not in entity_types: continue
        if len(entity_text) < min_entity_len: continue
        rng = ner_entity_char_range(ltp_words, start_idx, end_idx)
        if rng is None: continue
        cs, ce = rng
        if cs >= len(text_w) or ce > len(text_w): continue
        if text_w[cs:ce] != entity_text:
            # mismatch — skip (data inconsistency)
            continue
        if not wapic_has_token_at(wapic_words, cs, ce):
            out.append((cs, ce, entity_text, tag))
    return out


# ----- worker -----------------------------------------------------------

_WAPIC_MODEL = None
_ENTITY_TYPES = ("Nh",)
_MIN_ENTITY_LEN = 2


def _init_worker(model, entity_types, min_entity_len):
    global _WAPIC_MODEL, _ENTITY_TYPES, _MIN_ENTITY_LEN
    _WAPIC_MODEL = model
    _ENTITY_TYPES = entity_types
    _MIN_ENTITY_LEN = min_entity_len


def _process_batch(args):
    src_batch, ltp_batch, ner_batch = args
    wapic_out = wapic_cut_batch(_WAPIC_MODEL, src_batch)
    mined = []
    for sent, lw_str, ner_list, ww in zip(src_batch, ltp_batch, ner_batch, wapic_out):
        if not ww: continue
        lw = lw_str.split()
        dis = find_ner_disagreements(ww, lw, ner_list, _ENTITY_TYPES, _MIN_ENTITY_LEN)
        if not dis: continue
        ltp_retag2 = retag2_words(lw)
        mined.append(bmes_text(ltp_retag2))
    return mined


# ----- main -------------------------------------------------------------

def batch_iter(source_path, dedup, batch_size, target_n, entity_types):
    src_batch = []; ltp_batch = []; ner_batch = []
    n_read = 0; n_kept = 0
    for line in open(source_path, encoding="utf-8", errors="ignore"):
        if not line.strip(): continue
        try: d = json.loads(line)
        except: continue
        s = d.get("source"); c = d.get("cut"); ner = d.get("ner")
        if not s or not c: continue
        # Quick filter: require at least one entity of requested types
        if not ner: continue
        has_target = any(isinstance(e, list) and len(e) >= 1 and e[0] in entity_types
                         for e in ner)
        if not has_target: continue
        n_read += 1
        if sent_hash(s) in dedup: continue
        n_kept += 1
        src_batch.append(s); ltp_batch.append(c); ner_batch.append(ner)
        if len(src_batch) >= batch_size:
            yield (src_batch, ltp_batch, ner_batch), n_read, n_kept
            src_batch = []; ltp_batch = []; ner_batch = []
        if n_kept >= target_n: break
    if src_batch:
        yield (src_batch, ltp_batch, ner_batch), n_read, n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true", required=True)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--wapic-model", default="data/wapic-20260603-h24_2.wac")
    ap.add_argument("--out", default="data/mined_ner_disagree.txt")
    ap.add_argument("-n", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--entity-types", default="Nh",
                    help="comma-separated LTP NER types to mine (Nh=person, "
                         "Ni=org, Ns=location)")
    ap.add_argument("--min-entity-len", type=int, default=2,
                    help="skip entities shorter than this (chars)")
    args = ap.parse_args()

    entity_types = tuple(args.entity_types.split(","))

    print("Building dedup set...", file=sys.stderr)
    dedup = build_dedup_set(STAGE1, STAGE2)

    print(f"\n=== NER-GUIDED MINING ===", file=sys.stderr)
    print(f"  workers: {args.workers}", file=sys.stderr)
    print(f"  entity:  {entity_types}", file=sys.stderr)
    print(f"  model:   {args.wapic_model}", file=sys.stderr)
    print(f"  target:  {args.n} novel (with target NER)", file=sys.stderr)

    out_fp = open(args.out, "w", encoding="utf-8")
    n_yield = 0
    t0 = time.time()

    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.wapic_model, entity_types, args.min_entity_len)) as pool:
        gen = batch_iter(args.source, dedup, args.batch, args.n, entity_types)
        batches_for_pool = []
        last_read = 0; last_kept = 0

        def drain_pool():
            nonlocal n_yield
            if not batches_for_pool: return
            for result in pool.imap_unordered(_process_batch, batches_for_pool):
                for bmes_str in result:
                    out_fp.write(bmes_str); n_yield += 1
                out_fp.flush()
            batches_for_pool.clear()

        DISPATCH_CHUNK = args.workers * 4
        for batch_tuple, n_read, n_kept in gen:
            last_read = n_read; last_kept = n_kept
            batches_for_pool.append(batch_tuple)
            if len(batches_for_pool) >= DISPATCH_CHUNK:
                drain_pool()
                print(f"  read={n_read} novel={n_kept} mined={n_yield} t={time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
        drain_pool()

    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  source read:    {last_read}")
    print(f"  novel:          {last_kept}")
    print(f"  mined yields:   {n_yield}")
    print(f"  output:         {args.out}")


if __name__ == "__main__":
    main()
