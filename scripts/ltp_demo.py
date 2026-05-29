"""
LTP REPL — usage:

    source ~/.venv310/bin/activate
    python scripts/ltp_demo.py                # 默认 LTP/small
    python scripts/ltp_demo.py -m LTP/base1   # 换模型
    python scripts/ltp_demo.py -t cws,pos     # 只跑部分任务

启动后逐行输入句子回车即可分析。命令：
    :q / :quit / Ctrl-D    退出
    :tasks cws,pos,ner     切换任务
    :model LTP/base2       切换模型 (会重新加载)
    :help                  显示帮助

任务说明：
  cws 分词 / pos 词性 / ner 命名实体 / dep 依存句法 / srl 语义角色 / sdp 语义依存
"""

import argparse
import os
import sys

# 默认走本地缓存，避免代理 HEAD 检查卡住；模型不在本地时再去网络
os.environ.setdefault("HF_HUB_OFFLINE", "0")

from ltp import LTP

ALL_TASKS = ["cws", "pos", "ner", "dep", "srl", "sdp"]


def load_model(name):
    print(f"加载模型: {name} ...", flush=True)
    try:
        ltp = LTP(name, local_files_only=True)
        print(f"  (来自本地缓存) tokenizer={type(ltp.tokenizer).__name__}")
    except Exception:
        print("  本地缓存未命中，从 HuggingFace 下载...", flush=True)
        ltp = LTP(name)
        print(f"  下载完成 tokenizer={type(ltp.tokenizer).__name__}")
    return ltp


def render(sent, result, tasks):
    lines = []
    cws = result.cws[0] if "cws" in tasks else None
    if "cws" in tasks:
        lines.append(f"  cws: {' / '.join(cws)}")
    if "pos" in tasks:
        pairs = " ".join(f"{t}/{p}" for t, p in zip(cws, result.pos[0]))
        lines.append(f"  pos: {pairs}")
    if "ner" in tasks:
        ents = result.ner[0]
        if ents:
            lines.append("  ner: " + ", ".join(f"{text}({tag})" for tag, text, _, _ in ents))
        else:
            lines.append("  ner: (无)")
    if "dep" in tasks:
        dep = result.dep[0]
        edges = [
            f"{cws[i]}--{lab}-->{('ROOT' if h == 0 else cws[h - 1])}"
            for i, (h, lab) in enumerate(zip(dep["head"], dep["label"]))
        ]
        lines.append("  dep: " + " | ".join(edges))
    if "srl" in tasks:
        srl = result.srl[0]
        if srl:
            for frame in srl:
                args = ", ".join(f"{role}={text}" for role, text, _, _ in frame["arguments"])
                lines.append(f"  srl: 谓词={frame['predicate']}  {args}")
        else:
            lines.append("  srl: (无)")
    if "sdp" in tasks:
        sdp = result.sdp[0]
        edges = [
            f"{cws[i]}--{lab}-->{('ROOT' if h == 0 else cws[h - 1])}"
            for i, (h, lab) in enumerate(zip(sdp["head"], sdp["label"]))
        ]
        lines.append("  sdp: " + " | ".join(edges))
    return "\n".join(lines)


HELP = """命令:
  :q / :quit             退出
  :tasks <列表>          切换任务，逗号分隔，例如 :tasks cws,pos,ner
  :model <name>          切换模型，例如 :model LTP/base2
  :help                  显示本帮助
任务可选: cws pos ner dep srl sdp"""


def main():
    parser = argparse.ArgumentParser(description="LTP REPL")
    parser.add_argument("-m", "--model", default="LTP/small")
    parser.add_argument("-t", "--tasks", default=",".join(ALL_TASKS))
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in ALL_TASKS]
    if bad:
        sys.exit(f"未知任务: {bad}，可选: {ALL_TASKS}")

    try:
        import readline  # noqa: F401  让方向键和历史可用
    except ImportError:
        pass

    ltp = load_model(args.model)
    print(f"任务: {tasks}")
    print("输入句子回车分析；:help 查看命令；Ctrl-D 退出。\n")

    while True:
        try:
            line = input("ltp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in (":q", ":quit", ":exit"):
            break
        if line == ":help":
            print(HELP)
            continue
        if line.startswith(":tasks"):
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("用法: :tasks cws,pos,ner")
                continue
            new = [t.strip() for t in parts[1].split(",") if t.strip()]
            bad = [t for t in new if t not in ALL_TASKS]
            if bad:
                print(f"未知任务: {bad}")
                continue
            tasks = new
            print(f"已切到任务: {tasks}")
            continue
        if line.startswith(":model"):
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("用法: :model LTP/base2")
                continue
            try:
                ltp = load_model(parts[1])
            except Exception as e:
                print(f"加载失败: {e}")
            continue

        try:
            result = ltp.pipeline([line], tasks=tasks)
        except Exception as e:
            print(f"  [错误] {e}")
            continue
        print(render(line, result, tasks))


if __name__ == "__main__":
    main()
