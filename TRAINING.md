# 教程：用人民日报语料训练自己的中文分词模型

除了发布的 SOTA 模型（[Ismantic/wapic-cws](https://huggingface.co/Ismantic/wapic-cws)），
本仓库也提供一条**从零训练**的完整链路。本教程用公开的**人民日报 1998 语料**
（北大/富士通标注，PFR 格式）训练一个 CRF 分词模型：1–5 月做训练集、6 月做测试集。

全流程只用仓库自带的脚本，产出的分词口径与发布模型一致（retag2 / PreSegment：
人名整体、标点独立、中英数按字符类型切分）。

## 0. 先编译 wapic

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 1. 获取原始语料

原始 PFR 语料（`词/词性`）来自公开仓库，下载 1998 年 1–6 月的六个文件放进 `data/`：

```bash
git clone --depth 1 https://github.com/chenhui-bupt/PeopleDaily1998.git /tmp/pd1998
unzip -o /tmp/pd1998/199801.zip -d /tmp/pd1998
cp /tmp/pd1998/199801/1998{01,02,03,04,05,06}.txt data/
```

每行是一个段落，形如：

```
19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ...
```

> `data/` 已在 `.gitignore` 中（除 `pattern.txt`），这些数据只留在本地、不入库。

## 2. PFR → `{source, cut}` jsonl

`scripts/convert.py` 丢掉段落 id 和词性、拆开 `[...]nt` 命名实体括号、**合并连续
`/nr` 人名**（`江/nr 泽民/nr` → `江泽民`），再把 `cut` 按 PreSegment 口径归一化
（拉丁/数字 run 合并、中英数边界切、标点独立、空白丢弃，汉字分词沿用语料）：

```bash
python3 scripts/convert.py
# train: 102,739 records -> data/PeopleDaily_1-5.jsonl
# test:   21,143 records -> data/PeopleDaily_6.jsonl
```

每行一条：`{"source": "迈向充满希望的新世纪…", "cut": "迈向 充满 希望 的 新 世纪 …"}`。
（想保留语料原样的 `江 泽民` 拆分，加 `--split-names`。）

## 3. jsonl → BMES 训练格式

wapic 的原生格式是 BMES 列式（`<字> <B|M|E|S>`，每行一字、空行分句）。
`scripts/prepare.py` 把上一步的 jsonl 转成 BMES，1–5 月作训练集、6 月作测试集：

```bash
python3 scripts/prepare.py
# [train] 102,739 sentences -> data/PeopleDaily_1-5.txt
# [test ]  21,143 sentences -> data/PeopleDaily_6.txt
```

```
迈 B
向 E
充 B
满 E
...
```

## 4. 训练

用 L-BFGS（OWL-QN，带 L1 稀疏）在训练集上拟合 CRF。`data/pattern.txt` 是特征模板
（unigram + bigram）：

```bash
./build/wapic fit -p data/pattern.txt -a l-bfgs -i 100 -2 0.0001 -t 8 \
    --save-binary data/PeopleDaily_1-5.txt data/PeopleDaily_model.wac
```

常用参数：`-a sgd-l1|l-bfgs` 优化器，`-i` 最大迭代，`-1`/`-2` L1/L2 惩罚，
`-t` 线程数，`--save-binary` 存二进制模型，`--save-prune` 存时剪掉零权重。

## 5. 评估

测试集是带标签的 BMES gold；先去掉标签列得到纯字符输入，跑模型，再算 span 级 F1：

```bash
awk '{print $1}' data/PeopleDaily_6.txt > /tmp/pd6_nolabel.txt
./build/wapic test -m data/PeopleDaily_model.wac /tmp/pd6_nolabel.txt /tmp/pd6_pred.txt

python3 - data/PeopleDaily_6.txt /tmp/pd6_pred.txt <<'PY'
import sys
def cols(path, is_pred):                # 读列式：gold=[字,标签]，pred=[标签,分数]
    S, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip()
        if not line:
            if cur: S.append(cur); cur = []
            continue
        if is_pred and line.startswith("score="): continue
        cur.append(line.split())
    if cur: S.append(cur)
    return S
def spans(tags):                        # BMES 标签 -> 词的 (起,止) span 集合
    res, start = set(), 0
    for i in range(1, len(tags) + 1):
        if i == len(tags) or tags[i] in ("B", "S"):
            res.add((start, i)); start = i
    return res
gold, pred = cols(sys.argv[1], False), cols(sys.argv[2], True)
tp = fp = fn = 0
for g, p in zip(gold, pred):
    gt = [r[1] for r in g]; pt = [r[0] for r in p]
    if len(gt) != len(pt): continue     # 同一串字符，标签对齐
    a, b = spans(gt), spans(pt)
    tp += len(a & b); fp += len(b - a); fn += len(a - b)
P, R = tp / (tp + fp), tp / (tp + fn)
print(f"F1={2*P*R/(P+R)*100:.2f}  P={P*100:.2f}  R={R*100:.2f}")
PY
```

> 这段 span 级 P/R/F1 的逻辑与 `scripts/evaluate.sh` 一致。

在本教程的 1–5 月训练 / 6 月测试划分下，实测 **F1 = 97.40**（P 97.42 / R 97.38，
`-i 100 -t 8` 约 3 分钟训完，102,739 句训练 / 21,143 句测试）。这已是很不错的结果；
发布模型用了约 1400 万句、两阶段 warm-start，才进一步到 97.70/97.48。

## 想更进一步

- **换更大数据**：把更多语料转成同样的 `{source, cut}` jsonl，再走 `prepare.py`。
- **两阶段 warm-start**：先训一个 base，再 `--init-from base.wac` 在精修集上继续，
  最后 `convert --save-prune` 剪枝。发布模型的完整配方见数据集仓库
  [`RELEASE_TRAINING_DATA.md`](https://huggingface.co/datasets/Ismantic/wapic-cws-data)。
- **保持口径一致**：训练数据务必用 PreSegment 口径（本教程的 `convert.py` 已保证），
  否则和推理对不齐。字符类型规则集中在 `scripts/retag2.py`。
