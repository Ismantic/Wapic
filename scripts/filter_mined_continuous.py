"""Filter mined NER disagreement data: keep only true disagreements where
wapic.cut (continuous text, NOT cut_smart) still disagrees with LTP.

Background: mining used cut_smart which pre-splits on whitespace. Some
disagreements only happen because cut_smart breaks small segments. When the
text is treated as continuous (real-world bcv3 case), wapic may actually
agree with LTP. Filter those out.

Usage:
  PYTHONPATH=build_py/python python3 scripts/filter_mined_continuous.py \\
    --in data/mined_ner_pybind_2m.txt \\
    --out data/mined_filtered_cont.txt \\
    --wapic-model data/wapic-20260603-h24_2.wac
"""
import argparse
import sys
import time

sys.path.insert(0, "build_py/python")
import wapic


def read_bmes(path):
    """Yield (chars_text, ltp_words) for each sentence."""
    chars = []; words = []; cw = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if chars:
                    if cw:
                        words.append(cw); cw = ""
                    yield "".join(chars), words
                    chars = []; words = []
                continue
            p = line.split()
            if len(p) >= 2:
                c, t = p[0], p[-1]
                chars.append(c); cw += c
                if t in ("E", "S"):
                    words.append(cw); cw = ""
    if chars:
        if cw: words.append(cw)
        yield "".join(chars), words


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
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wapic-model", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    print("Loading wapic model...", file=sys.stderr, flush=True)
    seg = wapic.Segmenter(args.wapic_model)

    out_fp = open(args.out, "w", encoding="utf-8")
    n_in = n_kept = 0
    t0 = time.time()

    src_batch = []; ltp_batch = []

    def flush():
        nonlocal n_kept
        if not src_batch:
            return
        # Use regular cut (continuous text mode)
        wapic_outs = seg.cut_batch(src_batch)
        for src, ltp_w, wapic_w in zip(src_batch, ltp_batch, wapic_outs):
            if wapic_w == ltp_w:
                # Identical → false positive, discard
                continue
            # Still disagrees on continuous text → keep
            for line in words_to_bmes_lines(ltp_w):
                out_fp.write(line + "\n")
            out_fp.write("\n")
            n_kept += 1
        out_fp.flush()

    for path in args.inputs:
        print(f"\nProcessing {path}...", file=sys.stderr, flush=True)
        for src, ltp_w in read_bmes(path):
            n_in += 1
            src_batch.append(src); ltp_batch.append(ltp_w)
            if len(src_batch) >= args.batch:
                flush()
                src_batch = []; ltp_batch = []
                if n_in % (args.batch * 4) == 0:
                    print(f"  read={n_in} kept={n_kept} t={time.time()-t0:.0f}s",
                          file=sys.stderr, flush=True)
    flush()

    out_fp.close()
    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.0f}s ===")
    print(f"  input:  {n_in}")
    print(f"  kept:   {n_kept} ({100*n_kept/max(n_in,1):.1f}%)")
    print(f"  drop:   {n_in - n_kept} ({100*(n_in-n_kept)/max(n_in,1):.1f}%)")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
