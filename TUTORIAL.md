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

## 1. 解压原始语料

仓库自带人民日报 1998 标注语料（PFR 格式 `词/词性`）：`data/PeopleDaily1998.zip`。
解压 1–6 月的六个文件到 `data/`：

```bash
unzip -j data/PeopleDaily1998.zip '199801/1998*.txt' -d data/
```

每行是一个段落，形如：

```
19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ...
```

> 语料来源：PFR 语料库（北京大学计算语言学研究所 + 富士通研究开发中心，经人民日报社
> 许可制作），仅供研究复现使用，版权归原作者所有。解压出的 `1998*.txt` 及后续生成的
> jsonl / BMES 文件都在 `.gitignore` 中（仅 `pattern.txt` 与语料 zip 入库），不占用仓库。

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

`scripts/test.py` 会从 gold 生成纯字符输入、跑模型、算 span 级 P/R/F1：

```bash
python3 scripts/test.py data/PeopleDaily_model.wac --gold data/PeopleDaily_6.txt
#   PeopleDaily_6.txt  F1=97.40  P=97.42  R=97.38
```

在本教程的 1–5 月训练 / 6 月测试划分下，实测 **F1 = 97.40**（P 97.42 / R 97.38，
`-i 100 -t 8` 约 3 分钟训完，102,739 句训练 / 21,143 句测试）。这已是很不错的结果；
发布模型用了约 1400 万句、两阶段 warm-start，才进一步到 97.70/97.48。

## 想更进一步

- **换更大数据**：把更多语料转成同样的 `{source, cut}` jsonl，再走 `prepare.py`。
- **两阶段 warm-start**：先训一个 base，再 `--init-from base.wac` 在精修集上继续，
  最后 `convert --save-prune` 剪枝。发布模型的完整配方见数据集仓库
  [`RELEASE_TRAINING_DATA.md`](https://huggingface.co/datasets/Ismantic/wapic-cws-data)。
- **保持口径一致**：训练数据务必用 PreSegment 口径（本教程的 `convert.py` 已保证），
  否则和推理对不齐。字符类型规则见 `convert.py` 的 `classify()`（与 `src/preprocess.cc` 对齐）。
