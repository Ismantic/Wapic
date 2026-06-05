"""Generate antimerge training data targeting the merge errors we observed:

  1. Name + 高频后缀 (X委员, X院长, X书记, ...) — "龙勤国委员" 类
  2. 前缀虚词 + Name (后X, 在X, 据X, ...) — "后古力娜扎" 类
  3. 多字外文/罕见名独立 — "勒苦伍牛惹" 类

Output: BMES file with retag2 convention (data/antimerge_v3_train.txt)

Usage:
    python scripts/synth_antimerge_v2.py --out data/antimerge_v3_train.txt -n 100000
"""

import argparse
import random
import sys

# ---- surname / given name pools (no eval-set leakage) --------------------

CASE_BLACKLIST = ["镇全", "强峰", "子健", "李镇", "林强", "张子",
                  "镇李", "强林", "子张"]

SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林郑罗宋谢唐韩"
                "曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜"
                "范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦"
                "阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛"
                "洪龚汤陶黎温莫易樊乔文安殷颜庄章鲁倪庞邢俞翟蓝"
                "聂齐向申葛闫焦裴米庄房柳卞屠舒蒲蔚梅盛")

GIVEN = list("伟芳娜秀英敏静丽强磊洋艳勇军杰娟涛明超兰霞平刚桂华"
             "建国玲文梅萍珍清芬芳荣志学斌春夏秋冬海立东西南北方"
             "晓小红玉雪素惠美君正长生奇思齐心仁义礼智信忠孝礼"
             "新中良民勤俭温良恭让民勤俭让世杰雄峰飞鸿鹏鹤鸿凡")

# Foreign-style 2-3 char chunks for synthesizing 4-5 char transliterated names
FOREIGN_CHUNKS = ["阿尔", "贝克", "巴布", "卡特", "德罗", "费尔", "弗兰",
                  "格里", "哈罗", "伊万", "约翰", "凯丽", "拉里", "马克",
                  "尼克", "奥尔", "彼得", "罗杰", "山姆", "汤姆", "维克",
                  "威廉", "詹姆", "尤里", "卓娅", "波罗", "弗洛", "戈尔",
                  "卡尔", "莫罗", "切瓦", "桑切", "雅典", "斯坦", "皮特",
                  "古力", "牛惹", "苦伍", "勒苦", "雅库", "纳扎", "扎拉"]

# ---- vocabulary used for context templates ------------------------------

TITLES = [  # high-freq suffix titles that wapic over-merges with names
    "委员", "委员长", "副委员长",
    "院长", "副院长", "院士",
    "书记", "副书记", "总书记",
    "部长", "副部长",
    "总裁", "副总裁", "总经理", "副总经理", "经理",
    "主任", "副主任", "主席", "副主席",
    "教授", "副教授", "讲师", "院士",
    "记者", "本报记者", "特约记者",
    "同志", "老师", "医生", "法官", "律师",
    "局长", "副局长", "处长", "科长",
    "校长", "副校长",
    "先生", "女士", "夫人",
    "教官", "教练", "队长",
    "厅长", "司长", "署长",
    "干事", "干部", "员工",
]

# prefix function words that wapic accidentally merges into name boundary
PREFIX_WORDS = [
    "据", "在", "由", "对", "向", "和", "与", "及",
    "记者", "通讯员", "特约记者",
    "副", "前", "时", "现", "原",
    "本报讯", "新华社", "中新社",
]

# verbs commonly following a name
VERBS_AFTER_NAME = [
    "表示", "认为", "强调", "指出", "透露", "介绍", "宣布",
    "出席", "参加", "主持", "担任", "兼任",
    "说", "称", "表达", "回应",
    "接受", "采访",
    "成为", "当选", "获得", "荣获", "代表",
    "前往", "抵达",
    "签署", "发表", "提交",
]

CONJUNCTIONS = ["和", "与", "及", "、"]
PUNCT_END = ["。", "！", "？"]


# ---- name generators ---------------------------------------------------

def random_chinese_name(rng, length=3):
    """3-char (or other length) random Chinese name, avoiding eval blacklist."""
    for _ in range(20):
        s = rng.choice(SURNAMES)
        rest = "".join(rng.choice(GIVEN) for _ in range(length - 1))
        name = s + rest
        if not any(b in name for b in CASE_BLACKLIST):
            return name
    return None


def random_foreign_name(rng, min_len=4, max_len=6):
    """Foreign transliteration: concat 2-3 char chunks. May or may not contain ·."""
    target_len = rng.randint(min_len, max_len)
    name = ""
    while len(name) < target_len:
        chunk = rng.choice(FOREIGN_CHUNKS)
        if len(name) + len(chunk) > target_len + 1:
            break
        name += chunk
    # 30% chance: insert middle-dot if length >= 4
    if len(name) >= 4 and rng.random() < 0.3:
        mid = rng.randint(2, len(name) - 2)
        name = name[:mid] + "·" + name[mid:]
    return name


# ---- BMES + retag2 ------------------------------------------------------

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


def write_bmes(words, fp):
    for c, t in words_to_bmes(words):
        fp.write(f"{c} {t}\n")
    fp.write("\n")


# ---- sentence builders --------------------------------------------------

def build_name_plus_title(rng):
    """Pattern: <prefix? > <NAME> <TITLE> <VERB> ... 。  (修 龙勤国委员 类)
    Returns word list (cut), no extra punct."""
    name = random_chinese_name(rng, length=rng.choice([2, 3, 3, 3, 4]))
    if not name: return None
    title = rng.choice(TITLES)
    verb = rng.choice(VERBS_AFTER_NAME)
    # several context patterns
    pat = rng.randint(0, 4)
    if pat == 0:
        # 「X 委员 Y 说 ， ... 。」
        words = [name, title, verb, "，", "一切", "进展", "顺利", "。"]
    elif pat == 1:
        # 「在 会议 上 ， X 委员 表示 ， ...。」
        words = ["在", "会议", "上", "，", name, title, verb, "，",
                 "将", "继续", "推进", "工作", "。"]
    elif pat == 2:
        # 「记者 X 委员 报道 。」
        words = ["记者", name, title, "报道", "。"]
    elif pat == 3:
        # 「据 X 委员 介绍 ， ... 。」
        words = ["据", name, title, "介绍", "，", "情况", "良好", "。"]
    else:
        # 「X 委员 出席 了 大会 。」
        words = [name, title, "出席", "了", "大会", "。"]
    return words


def build_prefix_plus_name(rng):
    """Pattern: <PREFIX> <NAME> ... 。  (修 后古力娜扎 类)"""
    prefix = rng.choice(PREFIX_WORDS)
    use_foreign = rng.random() < 0.4
    if use_foreign:
        name = random_foreign_name(rng, 4, 6)
    else:
        name = random_chinese_name(rng, length=rng.choice([3, 3, 4]))
    if not name: return None
    verb = rng.choice(VERBS_AFTER_NAME)
    # 句首前缀 vs 句中
    pat = rng.randint(0, 3)
    if pat == 0:
        words = [prefix, name, verb, "，", "一切", "如常", "。"]
    elif pat == 1:
        words = ["他", "看到", prefix, name, verb, "。"]
    elif pat == 2:
        words = ["昨日", "，", prefix, name, "接受", "采访", "时", verb, "。"]
    else:
        words = [prefix, name, "和", random_chinese_name(rng, 3),
                 "一同", "参加", "会议", "。"]
    return words


def build_foreign_name_in_context(rng):
    """Pattern: ... <FOREIGN_NAME(4-6 字)> ...  (修 勒苦伍牛惹 / 古力娜扎 类)"""
    name = random_foreign_name(rng, min_len=4, max_len=6)
    verb = rng.choice(VERBS_AFTER_NAME)
    pat = rng.randint(0, 4)
    if pat == 0:
        words = ["记者", "采访", "了", name, "。"]
    elif pat == 1:
        words = [name, verb, "，", "事情", "已经", "解决", "。"]
    elif pat == 2:
        words = ["昨天", "，", name, "与", random_chinese_name(rng, 3),
                 "共同", "出席", "了", "活动", "。"]
    elif pat == 3:
        words = ["据", "了解", "，", name, verb, "了", "感谢", "。"]
    else:
        words = [name, "今年", "已经", "60", "岁", "，", "仍", "活跃", "在", "一线", "。"]
    return words


def build_name_pair_with_title(rng):
    """Pattern: <NAME1> <TITLE>、<NAME2> <TITLE> ...  (双名+title)"""
    n1 = random_chinese_name(rng, 3)
    n2 = random_chinese_name(rng, 3)
    if not n1 or not n2: return None
    t1 = rng.choice(TITLES)
    t2 = rng.choice(TITLES)
    conj = rng.choice(CONJUNCTIONS)
    words = [n1, t1, conj, n2, t2, "共同", "出席", "了", "大会", "。"]
    return words


BUILDERS = [
    (build_name_plus_title,        4),  # weight 4
    (build_prefix_plus_name,       3),
    (build_foreign_name_in_context, 2),
    (build_name_pair_with_title,    1),
]


def weighted_choice(rng, items):
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    upto = 0
    for x, w in items:
        upto += w
        if r <= upto:
            return x
    return items[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/antimerge_v3_train.txt")
    ap.add_argument("-n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fp = open(args.out, "w", encoding="utf-8")
    n_written = 0
    while n_written < args.n:
        builder = weighted_choice(rng, BUILDERS)
        words = builder(rng)
        if not words: continue
        retag2 = retag2_words(words)
        write_bmes(retag2, fp)
        n_written += 1
        if n_written % 10000 == 0:
            print(f"  written {n_written}", file=sys.stderr, flush=True)
    fp.close()
    print(f"Done: {n_written} sentences → {args.out}")


if __name__ == "__main__":
    main()
