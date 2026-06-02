# 复现指南

完整复现 release 模型 + 评估的 step-by-step。允许 LTP NER 在 GPU 上少量浮动（不影响下游训练结果的方向性）。

---

## 0. 外部依赖

| 资源 | 路径 |
|---|---|
| PD-1998 原文 | `data/raw/pd_raw/` |
| LTP/all12m cut | `/home/tfbao/Data/data/All.12M.cut.jsonl` |
| OpenNews 全集 | `/home/tfbao/Data/data/OpenNews.sentences.txt` (25 GB / 226M 句) |
| LTP/base1 模型 | HuggingFace (本地缓存) |
| Python venv | `~/.venv310/` with `ltp==4.2.13` |
| GPU | NVIDIA RTX 2070 8GB（CUDA 13.2）|

---

## 1. Stage 1 base 模型（已固定，不重训）

`data/all12m_compact_v2.wac`（61M）

- 来源：用 `data/all12m_train.txt`（9.5M LTP/all12m 句）训出
- 训练命令历史已不可逆查；当前 release 流程**直接使用此文件作为暖启动起点**

---

## 2. PD-1998 现代化 + NER Nh 合并测试/训练集

输出：`data/pd_mp_train.txt`、`data/pd_mp_test.txt`（+ `_nolabel.txt`）

```bash
# 1) 解析 PD-1998 原始
python scripts/parse_pd1998.py \
    --raw-dir data/raw/pd_raw/ \
    --out-train data/raw/pd_train_raw.jsonl \
    --out-test data/raw/pd_test_raw.jsonl

# 2) 现代化（NER Nh 合并）
python scripts/modernize_pd.py \
    --input data/raw/pd_train_raw.jsonl --output data/raw/pd_train_modern.jsonl
python scripts/modernize_pd.py \
    --input data/raw/pd_test_raw.jsonl  --output data/raw/pd_test_modern.jsonl

# 3) 标点拆分
python scripts/split_punct.py \
    --input data/raw/pd_train_modern.jsonl --output data/raw/pd_train_mp.jsonl
python scripts/split_punct.py \
    --input data/raw/pd_test_modern.jsonl  --output data/raw/pd_test_mp.jsonl

# 4) 转 BMES
python scripts/jsonl_to_bmes.py --input data/raw/pd_train_mp.jsonl \
    --out-bmes data/pd_mp_train.txt --no-type --min-chars 3
python scripts/jsonl_to_bmes.py --input data/raw/pd_test_mp.jsonl \
    --out-bmes data/pd_mp_test.txt --out-nolabel data/pd_mp_test_nolabel.txt \
    --no-type --min-chars 3
```

验证：
```bash
# 江泽民 应为 B-M-E
grep -A2 "江 B" data/pd_mp_train.txt | head -5
```

---

## 3. clean_aug（LTP NER 过滤的 OpenNews 人名句）

`data/clean_aug_train.txt`（55k 句，2.17M tokens）

```bash
source ~/.venv310/bin/activate
python scripts/extract_name_rich.py \
    --input data/raw/ltp_opennews_3m.jsonl \
    --output data/raw/name_aug.jsonl \
    --limit-in 300000 --limit-out 80000 --batch 64

python scripts/jsonl_to_bmes.py --input data/raw/name_aug.jsonl \
    --out-bmes data/clean_aug_train.txt --no-type --min-chars 3
```

---

## 4. antimerge 模板（抗合并，不含评估 case 词汇）

`data/antimerge_v2_train.txt`（6300 句，seed=42 可复现）

```bash
# 生成（黑名单过滤 case 子串 镇全/强峰/子健/林强/李镇/张子）
python scripts/synth_antimerge.py --output data/raw/antimerge_v2.jsonl --n 6300 --seed 42

# 转 BMES
python scripts/jsonl_to_bmes.py --input data/raw/antimerge_v2.jsonl \
    --out-bmes data/antimerge_v2_train.txt --no-type --min-chars 3
```

模板类别：
- 双 3 字名直接相邻（记者AAA BBB 联合报道、本报讯(AAA BBB)）
- 双 3 字名 + 虚词（AAA 与/和/及/、 BBB ...）
- 地名串（省+市+区）
- 颜色复合（蓝红色、灰蓝色）
- 数量+币种（三元硬币、面值N元的T）
- 英文/全角缩写+中文（AI 试剂、ＡＢＯ 血型）

**历史说明**：`data/antimerge_train.txt`（旧版）是首次生成 antimerge 时的产物，含 13 句和评估 case 部分重叠的随机姓名（非故意，统计巧合）。已被 `data/antimerge_v2_train.txt`（黑名单过滤）替代。两版在结构和数量上等价。

---

## 5. OpenNews 全量 NER（生成主要 Stage 2 数据池）

`data/raw/opennews_full_nh.jsonl`（10M 含 Nh 句）

```bash
source ~/.venv310/bin/activate
nohup python scripts/ner_opennews_full.py \
    --input /home/tfbao/Data/data/OpenNews.sentences.txt \
    --input-format txt \
    --output data/raw/opennews_full_nh.jsonl \
    --batch 32 --max-chars 100 \
    --filter-nh-only \
    --limit-out 10000000 \
    > logs/ner_full.log 2>&1 &
```

约 14 小时，3.4 GB。

派生子集（python seeded，可复现）：

```bash
# 500k 通用样本（用 sample_sentences.py 的等效逻辑，按行采样 + seed=42）
shuf --random-source=<(yes 42) -n 500000 \
    data/raw/opennews_full_nh.jsonl > data/raw/opennews_nh_500k.jsonl
# 注：shuf --random-source 行为略依赖 coreutils 版本，
# 数据有少量浮动；可改用 sample_sentences.py 增强严格性

# 句首 3 字 Nh 子集
python scripts/filter_start_nh.py \
    --input data/raw/opennews_full_nh.jsonl \
    --output data/raw/opennews_start3.jsonl \
    --name-len 3 --limit 300000

# 转 BMES
python scripts/jsonl_to_bmes.py --input data/raw/opennews_nh_500k.jsonl \
    --out-bmes data/opennews_nh_500k_train.txt --no-type --min-chars 3 --max-chars 200
python scripts/jsonl_to_bmes.py --input data/raw/opennews_start3.jsonl \
    --out-bmes data/opennews_start3_train.txt --no-type --min-chars 3 --max-chars 200
```

---

## 6. Anchor 子样（Stage 1 数据防遗忘）

`data/anchor_{100k,300k,1m}.txt`

```bash
# sentence-level reservoir sample, Python random seed=42, 确定性
python scripts/sample_sentences.py --input data/all12m_train.txt --output data/anchor_100k.txt --n 100000  --seed 42
python scripts/sample_sentences.py --input data/all12m_train.txt --output data/anchor_300k.txt --n 300000  --seed 42
python scripts/sample_sentences.py --input data/all12m_train.txt --output data/anchor_1m.txt   --n 1000000 --seed 42
```

---

## 7. Stage 2 训练（候选 release 流程，以最终 winner 为准）

TBD —— H4/H5 跑完后填入 winner 配方。模板：

```bash
INIT=data/all12m_compact_v2.wac
# pattern.txt: bigram U00-U04 + U10-U14 + B
./build/wapic fit -p data/pattern.txt --init-from $INIT \
    -a l-bfgs -i <iter> -1 <l1> -2 0.0001 -t 4 --histsz 5 -e 1e-9 --save-binary \
    <combined_data.txt> data/exp_<name>.wac
```

最终通过 convert 后剪枝：
```bash
./build/wapic convert -m data/exp_<winner>.wac \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/wapic-<date>.wac
```

---

## 8. 评估

```bash
# 三指标一次出
bash scripts/eval_both.sh data/wapic-<date>.wac

# 15-case 详情
python scripts/test_name_cases.py --model data/wapic-<date>.wac
```

---

## 9. 关键脚本一览

| 脚本 | 用途 |
|---|---|
| scripts/parse_pd1998.py | PD-1998 原文 → jsonl |
| scripts/modernize_pd.py | PD-1998 现代化 + NER Nh 合并 |
| scripts/split_punct.py | 强制把标点当独立 token |
| scripts/jsonl_to_bmes.py | jsonl 的 cut 字段 → BMES |
| scripts/extract_name_rich.py | LTP NER 过滤含 Nh 句 |
| scripts/ner_opennews_full.py | OpenNews 全量 NER（GPU）|
| scripts/filter_start_nh.py | 筛"句首 N 字 Nh" |
| scripts/sample_sentences.py | 句子级 reservoir 采样（seed） |
| scripts/eval_both.sh | F1_pdmp + F1_12m + case 三指标 |
| scripts/test_name_cases.py | 15 个人名 case 测试 |
| scripts/chain_ner_full.sh | 串接 GPU NER 任务 |

---

## 10. 已知不确定性来源

- **LTP NER on GPU**：fp16 + batch + cudnn 在不同硬件可能有微小数值差异，但 Nh 检出率/位置在主流场景一致
- **shuf --random-source**：不同 coreutils 版本可能产生不同顺序；如需严格，改用 `sample_sentences.py`
- **earlyoom 杀进程**：训练时尽量保证 free mem > 4 GB；wapic 训练本身确定（md5 一致）
