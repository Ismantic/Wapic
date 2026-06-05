"""NER-guided mining via wapic Python binding (cut_smart).

Major upgrades over the subprocess version:
  - 20× faster (no fork/exec per batch, model in memory)
  - Handles whitespace in source correctly (cut_smart splits on whitespace
    and treats non-Chinese segments as single tokens)

Usage:
  PYTHONPATH=build_py/python python3 scripts/mine_ner_pybind.py \\
    -n 2000000 --entity-types Nh \\
    --wapic-model data/wapic-20260603-h24_2.wac \\
    --out data/mined_ner_pybind.txt
"""
import argparse
import hashlib
import json
import os
import sys
import time
sys.path.insert(0, "build_py/python")
import wapic

STAGE1 = "data/all12m_train_retag2.txt"
STAGE2 = "data/h24_1.txt"
DEFAULT_SOURCE = "data/raw/opennews_full_nh.jsonl"


# --- shared utilities ---

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
        if chars: yield "".join(chars)


def sent_hash(s):
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


def words_to_bmes_lines(words):
    out = []
    for w in words:
        if len(w) == 1:
            out.append(f"{w} S")
        else:
            out.append(f"{w[0]} B")
            for c in w[1:-1]:
                out.append(f"{c} M")
            out.append(f"{w[-1]} E")
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


# --- NER mining logic ---

def ner_entity_char_range(ltp_words, start_word_idx, end_word_idx):
    pos = 0; char_start = None
    for i, w in enumerate(ltp_words):
        if i == start_word_idx:
            char_start = pos
        if i == end_word_idx:
            return char_start, pos + len(w)
        pos += len(w)
    return None


def wapic_has_token_at(wapic_words, char_start, char_end):
    pos = 0
    for w in wapic_words:
        wlen = len(w)
        if pos == char_start and pos + wlen == char_end:
            return True
        pos += wlen
    return False


def find_ner_disagreements(wapic_words, ltp_words, ner_entities, entity_types,
                            min_entity_len=2):
    text_w = "".join(wapic_words)
    text_l = "".join(ltp_words)
    if text_w != text_l:
        return []
    out = []
    for ent in ner_entities:
        if not isinstance(ent, list) or len(ent) < 4: continue
        tag, entity_text, start_idx, end_idx = ent[0], ent[1], ent[2], ent[3]
        if tag not in entity_types: continue
        if len(entity_text) < min_entity_len: continue
        rng = ner_entity_char_range(ltp_words, start_idx, end_idx)
        if rng is None: continue
        cs, ce = rng
        if cs >= len(text_w) or ce > len(text_w): continue
        if text_w[cs:ce] != entity_text: continue
        if not wapic_has_token_at(wapic_words, cs, ce):
            out.append((cs, ce, entity_text, tag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--wapic-model", default="data/wapic-20260603-h24_2.wac")
    ap.add_argument("--out", default="data/mined_ner_pybind.txt")
    ap.add_argument("-n", type=int, default=2000000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--entity-types", default="Nh")
    ap.add_argument("--min-entity-len", type=int, default=2)
    args = ap.parse_args()

    entity_types = tuple(args.entity_types.split(","))

    print("Building dedup set...", file=sys.stderr)
    dedup = build_dedup_set(STAGE1, STAGE2)

    print(f"\n=== NER MINING (pybind) ===", file=sys.stderr)
    print(f"  model:   {args.wapic_model}", file=sys.stderr)
    print(f"  entity:  {entity_types}", file=sys.stderr)
    print(f"  target:  {args.n} novel", file=sys.stderr)

    print(f"  loading wapic model...", file=sys.stderr)
    seg = wapic.Segmenter(args.wapic_model)
    print(f"  loaded.", file=sys.stderr)

    out_fp = open(args.out, "w", encoding="utf-8")
    n_yield = 0
    t0 = time.time()

    src_batch = []; ltp_batch = []; ner_batch = []
    n_read = 0; n_kept = 0

    def flush():
        nonlocal n_yield
        if not src_batch: return
        # Use cut_smart_batch
        wapic_outs = seg.cut_smart_batch(src_batch)
        for sent, lw_str, ner, ww in zip(src_batch, ltp_batch, ner_batch, wapic_outs):
            if not ww: continue
            lw = lw_str.split()
            dis = find_ner_disagreements(ww, lw, ner, entity_types, args.min_entity_len)
            if not dis: continue
            ltp_retag2 = retag2_words(lw)
            for line in words_to_bmes_lines(ltp_retag2):
                out_fp.write(line + "\n")
            out_fp.write("\n")
            n_yield += 1
        out_fp.flush()

    with open(args.source, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            try: d = json.loads(line)
            except: continue
            s = d.get("source"); c = d.get("cut"); ner = d.get("ner")
            if not s or not c or not ner: continue
            has_target = any(isinstance(e, list) and len(e) >= 1 and e[0] in entity_types
                             for e in ner)
            if not has_target: continue
            n_read += 1
            if sent_hash(s) in dedup: continue
            n_kept += 1
            src_batch.append(s); ltp_batch.append(c); ner_batch.append(ner)
            if len(src_batch) >= args.batch:
                flush()
                src_batch = []; ltp_batch = []; ner_batch = []
                if n_kept % (args.batch * 4) == 0:
                    rate = n_kept / max(time.time() - t0, 0.01)
                    print(f"  read={n_read} novel={n_kept} mined={n_yield} "
                          f"rate={rate:.0f}/s t={time.time()-t0:.0f}s",
                          file=sys.stderr, flush=True)
            if n_kept >= args.n: break
    flush()

    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  source read:    {n_read}")
    print(f"  novel:          {n_kept}")
    print(f"  mined yields:   {n_yield}")
    print(f"  rate:           {n_kept/max(dt,0.01):.0f} novel/s")
    print(f"  output:         {args.out}")


if __name__ == "__main__":
    main()
