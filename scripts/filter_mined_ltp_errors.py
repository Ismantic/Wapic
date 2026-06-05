"""Filter LTP-error samples out of mined data.

Heuristic: for each mined sentence, identify the wapic-merged token that
disagrees with LTP. If that token appears as a *whole word* >= N times in
the existing Stage 2 training data (h24_1.txt), LTP is likely wrong
(it fragmented an established compound/idiom), so discard the sample.

Usage:
  python scripts/filter_mined_ltp_errors.py \\
    --in data/mined_ltp_disagree_v5_parallel.txt \\
        data/mined_ltp_disagree_v6_2m.txt \\
        data/mined_ltp_disagree_v7_2m.txt \\
    --out data/mined_filtered.txt \\
    --train data/h24_1.txt \\
    --min-freq 5
"""
import argparse
import sys
import time
from collections import Counter


def stage_tokens(path):
    """Yield (sentence_idx, token_list) from a BMES file."""
    cur_chars = []; cur_tags = []
    sent_idx = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if cur_chars:
                    words = []; cw = ""
                    for c, t in zip(cur_chars, cur_tags):
                        cw += c
                        if t in ("E", "S"):
                            words.append(cw); cw = ""
                    if cw: words.append(cw)
                    yield sent_idx, words
                    sent_idx += 1
                    cur_chars = []; cur_tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    cur_chars.append(parts[0])
                    cur_tags.append(parts[-1])
        if cur_chars:
            words = []; cw = ""
            for c, t in zip(cur_chars, cur_tags):
                cw += c
                if t in ("E", "S"):
                    words.append(cw); cw = ""
            if cw: words.append(cw)
            yield sent_idx, words


def is_cjk(c):
    return ('一' <= c <= '鿿') or ('㐀' <= c <= '䶿')


def is_pure_cjk_token(w):
    return bool(w) and all(is_cjk(c) for c in w)


def build_token_freq(train_path):
    """Count word frequency over training corpus."""
    print(f"Building token freq from {train_path}...", file=sys.stderr, flush=True)
    freq = Counter()
    t0 = time.time()
    n_sents = 0
    for _, words in stage_tokens(train_path):
        for w in words:
            if is_pure_cjk_token(w) and len(w) >= 4:
                freq[w] += 1
        n_sents += 1
        if n_sents % 1_000_000 == 0:
            print(f"  {n_sents} sents, |freq|={len(freq)}, t={time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
    print(f"  done: {n_sents} sents, |freq|={len(freq)}, t={time.time()-t0:.0f}s",
          file=sys.stderr, flush=True)
    return freq


def find_candidate_merges(words, min_len=4):
    """Yield (start_pos, length, merged_text) for >=min_len pure-CJK adjacent
    sequences in the LTP cut. A mined sentence's wapic-merged token is the
    concatenation of consecutive 2-3-char LTP segments. We approximate by
    looking at all runs of consecutive small (1-3 char) pure-CJK tokens
    whose total length >= min_len.
    """
    pos = 0
    n = len(words)
    i = 0
    while i < n:
        w = words[i]
        if (is_pure_cjk_token(w) and len(w) <= 3):
            # gather consecutive small-CJK
            j = i
            tot_len = 0
            while j < n and is_pure_cjk_token(words[j]) and len(words[j]) <= 3:
                tot_len += len(words[j])
                j += 1
            if tot_len >= min_len:
                merged = "".join(words[i:j])
                yield pos, tot_len, merged
            pos += sum(len(words[k]) for k in range(i, j))
            i = j
        else:
            pos += len(w)
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="Input mined BMES files")
    ap.add_argument("--out", required=True, help="Filtered output BMES")
    ap.add_argument("--train", default="data/h24_1.txt",
                    help="Reference training corpus for frequency check")
    ap.add_argument("--min-freq", type=int, default=5,
                    help="If wapic-merged token appears >= this many times in "
                         "training corpus, the sample is rejected (LTP error)")
    args = ap.parse_args()

    freq = build_token_freq(args.train)

    n_in = n_kept = n_rejected = 0
    rejected_samples = []
    out_fp = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    for path in args.inputs:
        print(f"\nProcessing {path} ...", file=sys.stderr)
        for sent_idx, words in stage_tokens(path):
            n_in += 1
            # Identify possible wapic-merge candidates from this LTP cut
            should_reject = False
            for pos, tlen, merged in find_candidate_merges(words, min_len=4):
                if freq.get(merged, 0) >= args.min_freq:
                    should_reject = True
                    if len(rejected_samples) < 20:
                        rejected_samples.append((merged, freq[merged], words))
                    break
            if should_reject:
                n_rejected += 1
                continue
            # Write to filtered output (BMES format)
            text = ""
            for w in words:
                if len(w) == 1:
                    text += f"{w} S\n"
                else:
                    text += f"{w[0]} B\n"
                    for c in w[1:-1]:
                        text += f"{c} M\n"
                    text += f"{w[-1]} E\n"
            text += "\n"
            out_fp.write(text)
            n_kept += 1
    out_fp.close()

    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    print(f"  input:    {n_in}")
    print(f"  kept:     {n_kept} ({100*n_kept/max(n_in,1):.1f}%)")
    print(f"  rejected: {n_rejected} ({100*n_rejected/max(n_in,1):.1f}%)")
    print(f"  output:   {args.out}")
    print(f"\n=== Sample rejected (first 20) ===")
    for merged, f, words in rejected_samples:
        print(f"  '{merged}' (h24_1 freq={f})  ctx: {' / '.join(words[:10])}...")


if __name__ == "__main__":
    main()
