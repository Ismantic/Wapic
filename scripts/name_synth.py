"""
合成"姓+名"组合 + 多模板，让 CRF 深度学到"姓+名 = 一个 token"的通用规律。

不依赖具体人名数据，而是用：
- 常见 100 姓 × 常见 ~500 名字字 → 50000 名字模式
- 每个名字塞进 20+ 多样模板
- 共 100w+ 训练样本

输出 cut.jsonl 格式。
"""

import argparse
import json
import random

# 百家姓常见 100 个（覆盖大部分）
SURNAMES_100 = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚")

# 常见名字字（双字名通常 2 个字）
GIVEN_CHARS = list("伟芳娜秀英敏静丽强磊洋艳勇军杰娟涛明超秀兰霞平刚桂英文华建国玲桂兰艳红玉梅萍秀梅秀珍秀华小芳玉兰玉珍秀荣秀清玉芬玉芳玉华玉英桂芬桂华桂荣桂珍丽华丽萍丽珍丽芳丽英丽君雪梅雪芳雪英素芳素珍素华素英惠英惠芳惠珍惠君美华美玲美英美珍美兰春华春英春兰桂梅桂芳桂珍桂华志国志强志明志华志军建华建国建军建民建勇建平建文建斌建荣建华学军学文学华学明学斌志勇志斌国强国华国军国民国良国华国文国斌国荣春生春明春华春晖春雨夏雨秋雨秋华秋明海军海涛海明海波海军立志立国立军立明立华立志东华东明东亮东军东海东方东升志远志强志成志清志超晓东晓明晓华晓军晓鹏晓波晓东晓兵晓林晓辉小军小明小华小强小波小峰小林小红小燕小芳小玉小英小敏小丽小梅小娟小亮小刚小磊小宁小敏小欣小娜小慧小颖")

# 至少 25 个不同语义/语法位置的模板
TEMPLATES = [
    "{N} 是 这 部 电影 的 主演 。",
    "{N} 接受 了 记者 的 采访 。",
    "据 {N} 介绍 ， 项目 进展 顺利 。",
    "{N} 在 比赛 中 表现 出色 。",
    "{N} 等 一 行 人 抵达 北京 。",
    "{N} 与 {N0} 共同 出席 了 仪式 。",
    "{N} 表示 ， 一切 都 会 好 起来 。",
    "{N} 透露 ， 双方 已 达成 一致 。",
    "{N} 担任 公司 总经理 一 职 。",
    "{N} 出席 会议 并 发言 。",
    "{N} 的 父亲 是 一 名 工程师 。",
    "{N} 在 现场 接受 采访 。",
    "{N} 说 ， 我 完全 同意 这个 看法 。",
    "记者 {N} 从 北京 发回 报道 。",
    "本报 讯 记者 {N} 报道 。",
    "{N} 强调 ， 必须 加强 监管 。",
    "{N} 称 ， 这 是 一 次 重要 的 尝试 。",
    "{N} 主持 了 今天 的 会议 。",
    "{N} 出生 于 1985 年 。",
    "{N} 来 自 山东 省 。",
    "{N} 担任 主席 已 多 年 。",
    "{N} 获得 一等奖 。",
    "在 现场 ， {N} 与 群众 亲切 交流 。",
    "{N} 与 同事 一起 加班 到 深夜 。",
    "据 悉 ， {N} 已 抵达 北京 。",
    "中央 领导 {N} 同志 发表 讲话 。",
    "{N} 任 该 项目 总 负责人 。",
    "{N} 是 著名 的 学者 。",
    "{N} 接受 了 表彰 。",
    "{N} 是 中国 足球 运动员 。",
    "教练 {N} 表示 ， 状态 很 好 。",
    "{N} 的 看法 是 这样 的 。",
    "{N} 是 一 名 优秀 的 医生 。",
    "{N} 的 同事 {N0} 也 在 现场 。",
    "记者 {N} 摄",
    "（ 本报 记者 {N} 摄 ）",
    "{N} 告诉 记者 ， 一切 顺利 。",
    "{N} 透露 ， 已 完成 任务 。",
    "{N} 来到 现场 ， 与 大家 交谈 。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-names", type=int, default=20000,
                    help="生成多少个不同名字")
    ap.add_argument("--per-name", type=int, default=10,
                    help="每个名字塞多少模板")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 生成名字：随机 姓 × 给定字 1-2 个
    names = set()
    while len(names) < args.n_names:
        s = random.choice(SURNAMES_100)
        g1 = random.choice(GIVEN_CHARS)
        if random.random() < 0.6:
            g2 = random.choice(GIVEN_CHARS)
            if g2 != g1:
                names.add(s + g1 + g2)
            else:
                names.add(s + g1)
        else:
            names.add(s + g1)

    names = list(names)
    print(f"Generated {len(names)} unique synthetic names")

    n_sentences = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for name in names:
            tmpls = random.sample(TEMPLATES, min(args.per_name, len(TEMPLATES)))
            for t in tmpls:
                other = random.choice(names)
                while other == name:
                    other = random.choice(names)
                cut = t.format(N=name, N0=other)
                src = cut.replace(" ", "")
                fout.write(json.dumps({"source": src, "cut": cut},
                                      ensure_ascii=False) + "\n")
                n_sentences += 1
    print(f"DONE. wrote {n_sentences} synthetic sentences -> {args.output}")


if __name__ == "__main__":
    main()
