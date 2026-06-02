"""
合成 antimerge 训练数据：教 CRF "相邻两个 3 字名不要合并" + 其它边界分组。

输出: jsonl with {"source": str, "cut": str}  (cut 用空格分词)

模板类别：
1. 双 3 字名直接相邻 ("AAA BBB" 紧靠)
2. 双 3 字名 + 连接虚词 ("AAA 与/和/及 BBB")
3. 双 3 字名 在括号里 ("(AAA BBB 报道)")
4. 名 + 全角/半角缩写 (AI/CPU/DNA/GDP/ABO/RNA)
5. 地名串 (省+市+区)
6. 颜色复合 (蓝红色 等)
7. 数量+币 (三元硬币 等)

显式排除评估 case 子串保证干净。
"""

import argparse
import json
import random

# 评估 case 子串黑名单 - 任何含这些子串的名字都过滤掉
CASE_BLACKLIST = ["镇全", "强峰", "子健", "李镇", "林强", "张子", "镇李", "强林"]

# 百家姓（避免低频姓）
SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑罗宋谢唐韩"
                "曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜"
                "范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦"
                "阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛")

# 常用名字字
GIVEN = list("伟芳娜秀英敏静丽强磊洋艳勇军杰娟涛明超兰霞平刚桂华建"
             "国玲文化梅萍珍清芬芳荣志学斌春夏秋冬海立东西南北方"
             "晓小红玉雪素惠美丽君正长生奇思齐心仁义礼智信忠孝礼"
             "新中良民勤俭温良恭让国华国军国民国良国文国斌国荣")

PROVINCES = ["北京", "天津", "上海", "重庆", "河北省", "山西省", "辽宁省",
             "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省", "福建省",
             "江西省", "山东省", "河南省", "湖北省", "湖南省", "广东省",
             "广西", "海南省", "四川省", "贵州省", "云南省", "西藏",
             "陕西省", "甘肃省", "青海省", "宁夏", "新疆"]
CITIES = ["北京市", "上海市", "天津市", "广州市", "深圳市", "杭州市",
          "南京市", "武汉市", "成都市", "西安市", "重庆市", "苏州市"]
DISTRICTS = ["海淀区", "朝阳区", "东城区", "西城区", "丰台区", "石景山区",
             "通州区", "顺义区", "大兴区", "昌平区", "怀柔区", "门头沟区"]

COLORS = ["红", "黄", "蓝", "绿", "紫", "白", "黑", "灰", "棕", "粉"]
COLOR_COMPOUND = ["红色", "黄色", "蓝色", "绿色", "紫色", "白色", "黑色"]

NUM_WORDS = ["一", "二", "三", "五", "十", "百", "千"]
CURRENCY = ["元", "分", "角", "毫"]
COINAGES = ["硬币", "纸币", "钱币", "券"]

ACRONYMS = ["AI", "CPU", "GPU", "DNA", "RNA", "GDP", "ABO",
            "ＡＢＯ", "ＤＮＡ", "ＡＩ"]
ACRO_FOLLOW = ["试剂", "检测", "分析", "数据", "技术", "血型", "应用",
               "研发", "标准", "技术指标"]

# 名+名 衔接虚词
JOIN_WORDS = ["和", "与", "及", "、"]

# 双 3 字名 句模板
TEMPLATES_PAIR = [
    "{N1}和{N2}一同出席仪式。",
    "{N1}与{N2}担任评委。",
    "{N1}及{N2}共同主持会议。",
    "{N1}、{N2}发表联合声明。",
    "他邀请{N1}与{N2}共进晚餐。",
    "{N1}和{N2}是好朋友。",
    "{N1}、{N2}两人都获奖。",
]
TEMPLATES_ADJACENT = [
    "记者{N1}{N2}联合报道。",
    "（摄影：{N1}{N2}）",
    "（{N1}{N2}）",
    "本报讯({N1}{N2}报道)",
    "本报讯（{N1}{N2}）",
]
TEMPLATES_GEO = [
    "他来自{P}{C}{D}。",
    "{P}{C}是个好地方。",
    "请联系{P}{C}{D}办事处。",
    "{C}{D}发生一起事件。",
    "前往{P}{C}考察。",
]
TEMPLATES_COLOR = [
    "{C1}{C2}的旗帜飘扬。",
    "天空是{C1}{C2}的。",
    "{C1}{C2}的头发。",
    "{C1}{C2}很漂亮。",
]
TEMPLATES_COIN = [
    "这是{N}{U}{T}。",
    "面值{N}{U}的{T}；",
    "{C}{N}{U}券；",
]
TEMPLATES_ACRO = [
    "采用{A}{F}进行检测。",
    "{A}{F}的发展。",
    "({F}：{A})",
]


def gen_name(rng, length=3):
    """生成 length 字人名，过滤 case 黑名单子串"""
    for _ in range(20):
        s = rng.choice(SURNAMES)
        rest = "".join(rng.choice(GIVEN) for _ in range(length - 1))
        name = s + rest
        if not any(b in name for b in CASE_BLACKLIST):
            return name
    return None  # give up


def to_cut(words):
    return " ".join(words)


def synth_pair_sep(rng):
    n1, n2 = gen_name(rng), gen_name(rng)
    if not n1 or not n2:
        return None
    tmpl = rng.choice(TEMPLATES_PAIR)
    s = tmpl.format(N1=n1, N2=n2)
    # cut: 让相邻名 + 中间虚词 + 后续词 分开
    # build cut by replacing back
    import re
    # crude split: words split by Chinese chars heuristics — simpler: rewrite template manually
    cut_tmpl_map = {
        "{N1}和{N2}一同出席仪式。": [n1, "和", n2, "一同", "出席", "仪式", "。"],
        "{N1}与{N2}担任评委。": [n1, "与", n2, "担任", "评委", "。"],
        "{N1}及{N2}共同主持会议。": [n1, "及", n2, "共同", "主持", "会议", "。"],
        "{N1}、{N2}发表联合声明。": [n1, "、", n2, "发表", "联合", "声明", "。"],
        "他邀请{N1}与{N2}共进晚餐。": ["他", "邀请", n1, "与", n2, "共进", "晚餐", "。"],
        "{N1}和{N2}是好朋友。": [n1, "和", n2, "是", "好", "朋友", "。"],
        "{N1}、{N2}两人都获奖。": [n1, "、", n2, "两", "人", "都", "获奖", "。"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


def synth_adjacent(rng):
    n1, n2 = gen_name(rng), gen_name(rng)
    if not n1 or not n2:
        return None
    tmpl = rng.choice(TEMPLATES_ADJACENT)
    s = tmpl.format(N1=n1, N2=n2)
    cut_tmpl_map = {
        "记者{N1}{N2}联合报道。": ["记者", n1, n2, "联合", "报道", "。"],
        "（摄影：{N1}{N2}）": ["（", "摄影", "：", n1, n2, "）"],
        "（{N1}{N2}）": ["（", n1, n2, "）"],
        "本报讯({N1}{N2}报道)": ["本报", "讯", "(", n1, n2, "报道", ")"],
        "本报讯（{N1}{N2}）": ["本报", "讯", "（", n1, n2, "）"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


def synth_geo(rng):
    p = rng.choice(PROVINCES)
    c = rng.choice(CITIES)
    d = rng.choice(DISTRICTS)
    tmpl = rng.choice(TEMPLATES_GEO)
    s = tmpl.format(P=p, C=c, D=d)
    cut_tmpl_map = {
        "他来自{P}{C}{D}。": ["他", "来", "自", p, c, d, "。"],
        "{P}{C}是个好地方。": [p, c, "是", "个", "好", "地方", "。"],
        "请联系{P}{C}{D}办事处。": ["请", "联系", p, c, d, "办事处", "。"],
        "{C}{D}发生一起事件。": [c, d, "发生", "一", "起", "事件", "。"],
        "前往{P}{C}考察。": ["前往", p, c, "考察", "。"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


def synth_color(rng):
    c1 = rng.choice(COLORS)
    c2 = rng.choice(COLOR_COMPOUND)
    if c1 == c2[0]:
        c2 = rng.choice([x for x in COLOR_COMPOUND if x[0] != c1])
    tmpl = rng.choice(TEMPLATES_COLOR)
    s = tmpl.format(C1=c1, C2=c2)
    cut_tmpl_map = {
        "{C1}{C2}的旗帜飘扬。": [c1, c2, "的", "旗帜", "飘扬", "。"],
        "天空是{C1}{C2}的。": ["天空", "是", c1, c2, "的", "。"],
        "{C1}{C2}的头发。": [c1, c2, "的", "头发", "。"],
        "{C1}{C2}很漂亮。": [c1, c2, "很", "漂亮", "。"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


def synth_coin(rng):
    n = rng.choice(NUM_WORDS)
    u = rng.choice(CURRENCY)
    t = rng.choice(COINAGES)
    c = rng.choice(COLOR_COMPOUND)
    tmpl = rng.choice(TEMPLATES_COIN)
    s = tmpl.format(N=n, U=u, T=t, C=c)
    cut_tmpl_map = {
        "这是{N}{U}{T}。": ["这", "是", n, u, t, "。"],
        "面值{N}{U}的{T}；": ["面值", n, u, "的", t, "；"],
        "{C}{N}{U}券；": [c, n, u, "券", "；"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


def synth_acro(rng):
    a = rng.choice(ACRONYMS)
    f = rng.choice(ACRO_FOLLOW)
    tmpl = rng.choice(TEMPLATES_ACRO)
    s = tmpl.format(A=a, F=f)
    cut_tmpl_map = {
        "采用{A}{F}进行检测。": ["采用", a, f, "进行", "检测", "。"],
        "{A}{F}的发展。": [a, f, "的", "发展", "。"],
        "({F}：{A})": ["(", f, "：", a, ")"],
    }
    if tmpl not in cut_tmpl_map:
        return None
    return {"source": s, "cut": to_cut(cut_tmpl_map[tmpl])}


SYNTHESIZERS = [
    (synth_pair_sep, 25),
    (synth_adjacent, 25),
    (synth_geo, 20),
    (synth_color, 10),
    (synth_coin, 10),
    (synth_acro, 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=6300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    weighted = []
    for fn, w in SYNTHESIZERS:
        weighted.extend([fn] * w)
    n_ok = 0
    with open(args.output, "w", encoding="utf-8") as out:
        while n_ok < args.n:
            fn = rng.choice(weighted)
            r = fn(rng)
            if r is None:
                continue
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} → {args.output}")


if __name__ == "__main__":
    main()
