"""
用 LTP/base1 对大规模语料做分词，输出 JSONL（每行 {"source": "...", "cut": "x x x"}）。

用法:
    source ~/.venv310/bin/activate
    python scripts/cut_corpus.py \
        -i /home/tfbao/Data/data/OpenNews.100M.sentences.txt \
        -o /home/tfbao/Data/data/OpenNews.100M.cut.jsonl

特性:
  - 走 GPU（默认 cuda），批处理
  - 断点续跑：输出文件已有 N 行就跳过输入前 N 行
  - 进度条 + 速率统计
  - 跳过空行
  - Ctrl-C 安全：随时中断，下次接着跑
  - --fp16: 半精度推理 (Turing/Ampere/Ada 上一般 1.5-2x 提速)
  - --sort-chunk N: 读 N 行后按长度排序再批处理，减少 padding 浪费；
    输出会按输入原顺序写出
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="输入文本，每行一句")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL")
    parser.add_argument("-m", "--model", default="LTP/base1")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-d", "--device", default="cuda", help="cuda | cpu | cuda:0 ...")
    parser.add_argument("--max-len", type=int, default=510, help="超长句子按字符截断")
    parser.add_argument("--no-count", action="store_true", help="跳过总行数统计（快但没 ETA）")
    parser.add_argument("--flush-every", type=int, default=1000, help="每多少批 flush 一次磁盘")
    parser.add_argument("--fp16", action="store_true", help="半精度推理")
    parser.add_argument("--sort-chunk", type=int, default=0,
                        help="按长度排序的块大小（0=关闭）。例如 4096 表示每 4096 行内按长度排序后批处理")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：数已写出的行数
    skip = 0
    if out_path.exists():
        with open(out_path, "rb") as f:
            for _ in f:
                skip += 1
        print(f"[resume] 输出已有 {skip} 行，跳过输入前 {skip} 行", flush=True)

    total = None
    if not args.no_count:
        print("统计总行数...", flush=True)
        total = count_lines(in_path)
        print(f"总行数: {total}", flush=True)

    print(f"加载模型 {args.model} -> {args.device} (fp16={args.fp16}) ...", flush=True)
    from ltp import LTP
    import torch

    try:
        ltp = LTP(args.model, local_files_only=True)
    except Exception:
        print("本地缓存未命中，从 HF 下载...", flush=True)
        ltp = LTP(args.model)
    ltp.to(args.device)
    if args.fp16:
        ltp.half()
    ltp.eval()

    try:
        from tqdm import tqdm
    except ImportError:
        os.system(f"{sys.executable} -m pip install tqdm -q")
        from tqdm import tqdm

    fin = open(in_path, "r", encoding="utf-8", errors="replace")
    fout = open(out_path, "a", encoding="utf-8", buffering=1024 * 1024)

    # 跳过断点之前的行
    for _ in range(skip):
        fin.readline()

    bs = args.batch_size
    max_len = args.max_len

    pbar = tqdm(total=total - skip if total else None, initial=0, unit="line", smoothing=0.05)
    n_done = 0
    n_skip = 0
    n_batches = 0
    t0 = time.time()

    def write_record(src, toks):
        fout.write(json.dumps({"source": src, "cut": " ".join(toks)}, ensure_ascii=False))
        fout.write("\n")

    def predict(batch_sents):
        prepped = [s if len(s) <= max_len else s[:max_len] for s in batch_sents]
        with torch.no_grad():
            res = ltp.pipeline(prepped, tasks=["cws"])
        return res.cws

    def process_batch_in_order(batch_sents):
        """直接按输入顺序跑一批，写出。出错降级到单条。"""
        nonlocal n_done, n_batches
        if not batch_sents:
            return
        try:
            cws = predict(batch_sents)
            for s, toks in zip(batch_sents, cws):
                write_record(s, toks)
        except Exception as e:
            tqdm.write(f"[batch err] {type(e).__name__}: {e} — 改单条重试")
            for s in batch_sents:
                try:
                    toks = predict([s])[0]
                    write_record(s, toks)
                except Exception as e2:
                    tqdm.write(f"  [skip line] {type(e2).__name__}: {str(e2)[:80]}")
        pbar.update(len(batch_sents))
        n_done += len(batch_sents)
        n_batches += 1
        if n_batches % args.flush_every == 0:
            fout.flush()
            os.fsync(fout.fileno())

    def process_chunk_sorted(chunk):
        """对 chunk(list[(idx, sent)]) 按长度排序批处理，最后按原 idx 顺序写出。"""
        nonlocal n_done, n_batches
        if not chunk:
            return
        order = sorted(range(len(chunk)), key=lambda i: len(chunk[i][1]))
        results: list = [None] * len(chunk)
        for start in range(0, len(order), bs):
            idxs = order[start:start + bs]
            sents = [chunk[i][1] for i in idxs]
            try:
                cws = predict(sents)
                for i, toks in zip(idxs, cws):
                    results[i] = toks
            except Exception as e:
                tqdm.write(f"[batch err] {type(e).__name__}: {e} — 改单条重试")
                for i in idxs:
                    try:
                        results[i] = predict([chunk[i][1]])[0]
                    except Exception as e2:
                        tqdm.write(f"  [skip line] {type(e2).__name__}: {str(e2)[:80]}")
                        results[i] = None
            n_batches += 1
            if n_batches % args.flush_every == 0:
                fout.flush()
                os.fsync(fout.fileno())
        for i in range(len(chunk)):
            _, s = chunk[i]
            toks = results[i]
            if toks is None:
                continue
            write_record(s, toks)
        pbar.update(len(chunk))
        n_done += len(chunk)

    try:
        if args.sort_chunk and args.sort_chunk > 0:
            chunk: list = []
            for line in fin:
                s = line.rstrip("\n\r")
                if not s.strip():
                    n_skip += 1
                    pbar.update(1)
                    continue
                chunk.append((len(chunk), s))
                if len(chunk) >= args.sort_chunk:
                    process_chunk_sorted(chunk)
                    chunk = []
            process_chunk_sorted(chunk)
        else:
            batch: list = []
            for line in fin:
                s = line.rstrip("\n\r")
                if not s.strip():
                    n_skip += 1
                    pbar.update(1)
                    continue
                batch.append(s)
                if len(batch) >= bs:
                    process_batch_in_order(batch)
                    batch = []
            process_batch_in_order(batch)
    except KeyboardInterrupt:
        print("\n[interrupt] 中断；磁盘已 flush，下次重启自动续跑。", flush=True)
    finally:
        fout.flush()
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass
        fout.close()
        fin.close()
        pbar.close()

    dt = time.time() - t0
    rate = n_done / dt if dt > 0 else 0
    print(f"\n完成: 写出 {n_done} 行 | 跳过空行 {n_skip} | 用时 {dt:.1f}s | 速率 {rate:.1f} 行/s")


if __name__ == "__main__":
    main()
