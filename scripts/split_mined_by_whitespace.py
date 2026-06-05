"""For each mined sentence, look up the original jsonl source (which may have
whitespace), then SPLIT it into multiple training sentences along whitespace
boundaries. Each whitespace-separated segment becomes its own BMES sentence
with the corresponding LTP cut as label.

This matches cut_smart's runtime behavior: at deploy time, cut_smart pre-splits
on whitespace and runs CRF per segment. If we train wapic on the segments,
the model learns to handle them properly.

Usage:
  python scripts/split_mined_by_whitespace.py \\
    --mined data/mined_ner_pybind_2m.txt \\
    --source data/raw/opennews_full_nh.jsonl \\
    --out data/mined_split_by_ws.txt
"""
import argparse
import hashlib
import json
import sys
import time


def sent_hash(s):
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()[:16]


def read_bmes(path):
    chars = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if chars:
                    yield "".join(chars)
                    chars = []
            else:
                p = line.split()
                if p:
                    chars.append(p[0])
        if chars: yield "".join(chars)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", required=True)
    ap.add_argument("--source", default="data/raw/opennews_full_nh.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Read mined sentences → hashes (key = whitespace-stripped source)
    print("Indexing mined sentences...", file=sys.stderr, flush=True)
    mined_hashes = set()
    for src in read_bmes(args.mined):
        mined_hashes.add(sent_hash(src))
    print(f"  {len(mined_hashes)} unique mined hashes", file=sys.stderr)

    # Stream jsonl, match by stripping whitespace from source
    print("\nProcessing jsonl source...", file=sys.stderr)
    out_fp = open(args.out, "w", encoding="utf-8")
    n_jsonl_match = 0
    n_segments_written = 0
    t0 = time.time()
    n_read = 0
    for line in open(args.source, encoding="utf-8", errors="ignore"):
        if not line.strip(): continue
        try:
            d = json.loads(line)
        except: continue
        s = d.get("source"); c = d.get("cut")
        if not s or not c: continue
        n_read += 1
        # Strip whitespace and check hash
        s_stripped = "".join(s.split())
        if sent_hash(s_stripped) not in mined_hashes:
            continue
        n_jsonl_match += 1

        # Now split by whitespace into segments. For each segment, find
        # corresponding LTP words.
        # LTP cut tokens may span across whitespace boundaries — but normally
        # LTP tokens don't contain whitespace, so we can match by char counts.
        ltp_words = c.split()
        # Build LTP positions in stripped text
        # Reconstruct mapping from stripped position to original position
        # Split source by whitespace
        segments = []  # list of (start_in_stripped, end_in_stripped)
        char_idx = 0
        seg_start = None
        for ch in s:
            if ch.isspace():
                if seg_start is not None:
                    segments.append((seg_start, char_idx))
                    seg_start = None
            else:
                if seg_start is None:
                    seg_start = char_idx
                char_idx += 1
        if seg_start is not None:
            segments.append((seg_start, char_idx))

        # For each segment, find LTP words fully within it
        ltp_pos = []  # cumulative char positions in stripped text
        p = 0
        for w in ltp_words:
            ltp_pos.append((p, p + len(w), w))
            p += len(w)

        for seg_start, seg_end in segments:
            seg_words = [w for (a, b, w) in ltp_pos
                         if a >= seg_start and b <= seg_end]
            if not seg_words: continue
            # Verify total length matches
            if sum(len(w) for w in seg_words) != seg_end - seg_start: continue
            # Write BMES for this segment
            for line in words_to_bmes_lines(seg_words):
                out_fp.write(line + "\n")
            out_fp.write("\n")
            n_segments_written += 1

        if n_jsonl_match % 5000 == 0:
            print(f"  jsonl read={n_read} matched={n_jsonl_match} "
                  f"segments={n_segments_written} t={time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)

    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  jsonl read:        {n_read}")
    print(f"  mined matched:     {n_jsonl_match}")
    print(f"  segments written:  {n_segments_written}")
    print(f"  avg segments/sent: {n_segments_written/max(n_jsonl_match,1):.1f}")
    print(f"  output:            {args.out}")


if __name__ == "__main__":
    main()
