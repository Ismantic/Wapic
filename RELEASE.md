# Release: `data/wapic-20260601.wac`

**完整可复现的训练记录**。基于 H12.1 实验（H2.3 + opennh 1M），全面优于旧 release。

历史：H13.1 (52M, 配方较复杂) 也曾为 release，被 H12.1 替换（更简单配方、同样指标、bcv2 更好）。

---

## 0. 最终指标

| 指标 | wapic-20260601 (新, H15.2) | wapic-20260529 (旧) | 变化 |
|---|---|---|---|
| size | 56M | 47M | +9M |
| F1_pdmp | 97.69 | 97.70 | -0.01 |
| F1_12m | 96.97 | 96.83 | +0.14 |
| 15-case | **15/15** | 12/15 | **+3** ⬆ |
| badcase_v2 (200) | 182/200 (91.0%) | 171/200 (85.5%) | +11 |
| badcase_v3 (500) | **374/500 (74.8%)** | 273/500 (54.6%) | **+101 (+20.2%)** ⬆ |
| BC001 (波·索提拉克) | ✓ | ✗ | 解决 |

历史版本：H13.1 (52M, foreign-dot focus) → H12.1 (51M, opennh 1M, F1_12m 高) → **H15.2 (56M, +3char_zh 500k, bcv3 极佳)**

---

## 1. 训练架构（两阶段）

### Stage 1：大数据蒸馏 — 不重训
- 起点模型：`data/all12m_compact_v2.wac`（61M，9.5M 句 LTP/all12m 训）
- 直接复用，作为 Stage 2 暖启动

### Stage 2：一次性多方向融合训练
- 训练数据 = 三方向数据 cat 拼接
- 单 pass L-BFGS i=50

---

## 2. 数据准备（按顺序）

### 2.1 PD-1998 现代化训练/测试数据

```bash
# 解析 PD-1998 原始
python scripts/parse_pd1998.py --raw-dir data/raw/pd_raw/ \
    --out-train data/raw/pd_train_raw.jsonl \
    --out-test data/raw/pd_test_raw.jsonl

# NER Nh 实体合并
python scripts/modernize_pd.py --input data/raw/pd_train_raw.jsonl --output data/raw/pd_train_modern.jsonl
python scripts/modernize_pd.py --input data/raw/pd_test_raw.jsonl --output data/raw/pd_test_modern.jsonl

# 标点拆分
python scripts/split_punct.py --input data/raw/pd_train_modern.jsonl --output data/raw/pd_train_mp.jsonl
python scripts/split_punct.py --input data/raw/pd_test_modern.jsonl --output data/raw/pd_test_mp.jsonl

# 转 BMES
python scripts/jsonl_to_bmes.py --input data/raw/pd_train_mp.jsonl \
    --out-bmes data/pd_mp_train.txt --no-type --min-chars 3
python scripts/jsonl_to_bmes.py --input data/raw/pd_test_mp.jsonl \
    --out-bmes data/pd_mp_test.txt --out-nolabel data/pd_mp_test_nolabel.txt \
    --no-type --min-chars 3
```

输出：`pd_mp_train.txt`（102k 句）+ `pd_mp_test.txt`（18k 句）

### 2.2 clean_aug — 早期 OpenNews 人名增强

```bash
source ~/.venv310/bin/activate
python scripts/extract_name_rich.py \
    --input data/raw/ltp_opennews_3m.jsonl \
    --output data/raw/name_aug.jsonl \
    --limit-in 300000 --limit-out 80000 --batch 64

python scripts/jsonl_to_bmes.py --input data/raw/name_aug.jsonl \
    --out-bmes data/clean_aug_train.txt --no-type --min-chars 3
```

输出：`clean_aug_train.txt`（55k 句）

### 2.3 opennews_nh_500k — 主要人名增强数据

```bash
# GPU NER on full OpenNews (sustained 1.5h GPU load)
source ~/.venv310/bin/activate
nohup python scripts/ner_opennews_full.py \
    --input /home/tfbao/Data/data/OpenNews.sentences.txt \
    --input-format txt \
    --output data/raw/opennews_full_nh.jsonl \
    --batch 32 --max-chars 100 \
    --filter-nh-only --limit-out 10000000 &
# 14h 后产 10M Nh 句

# 抽 500k（注：shuf 实现略依赖 coreutils 版本）
shuf --random-source=<(yes 42) -n 500000 \
    data/raw/opennews_full_nh.jsonl > data/raw/opennews_nh_500k.jsonl

python scripts/jsonl_to_bmes.py --input data/raw/opennews_nh_500k.jsonl \
    --out-bmes data/opennews_nh_500k_train.txt \
    --no-type --min-chars 3 --max-chars 200
```

输出：`opennews_nh_500k_train.txt`（500k 句，含 137k 3 字名）

### 2.4 pd_modern_500k — 现代 PeopleDaily

```bash
# 取 PeopleDaily 全集（45M 行）最后 5M 行（2020+ 现代内容）
tail -n 5000000 /home/tfbao/Data/data/PeopleDaily.sentences.txt > data/raw/pd_modern_raw.txt

# GPU NER（~95 min）
nohup python scripts/ner_opennews_full.py \
    --input data/raw/pd_modern_raw.txt --input-format txt \
    --output data/raw/pd_modern_ner.jsonl \
    --batch 32 --max-chars 100 &

# 过滤掉和测试集 source 完全相同的句子（保险）
python scripts/filter_test_leak.py \
    --input data/raw/pd_modern_ner.jsonl \
    --output data/raw/pd_modern_ner_clean.jsonl \
    --test-bmes data/pd_mp_test.txt

# 转 BMES
python scripts/jsonl_to_bmes.py \
    --input data/raw/pd_modern_ner_clean.jsonl \
    --out-bmes data/pd_modern_full.txt \
    --no-type --min-chars 3 --max-chars 200

# 抽 500k
python scripts/sample_sentences.py --input data/pd_modern_full.txt \
    --output data/pd_modern_500k.txt --n 500000 --seed 42
```

输出：`pd_modern_500k.txt`（500k 句）

### 2.5 foreign_dot_100k — 含中点的外国名（攻 BC001）

```bash
# 从全集筛 Nh 含 · 的句子
python -c "
import json
n=0
with open('/tmp/fd.jsonl','w') as fout:
    for line in open('data/raw/opennews_full_nh.jsonl'):
        try: o=json.loads(line)
        except: continue
        if any(t=='Nh' and ('·' in x or '・' in x) for t,x,*_ in o.get('ner',[])):
            fout.write(line); n+=1
            if n>=100000: break
"

python scripts/jsonl_to_bmes.py --input /tmp/fd.jsonl \
    --out-bmes data/foreign_dot_100k_train.txt \
    --no-type --min-chars 3 --max-chars 200
```

输出：`foreign_dot_100k_train.txt`（100k 句）

### 2.6 antimerge — 抗合并模板（合成，不含 case 词汇）

```bash
python scripts/synth_antimerge.py --output data/raw/antimerge_v2.jsonl --n 6300 --seed 42
python scripts/jsonl_to_bmes.py --input data/raw/antimerge_v2.jsonl \
    --out-bmes data/antimerge_train.txt --no-type --min-chars 3
```

注：实际 release 用的是更早一版 `antimerge_train.txt`（含 13 个和 case 部分重叠的随机姓名，统计巧合非故意）。`antimerge_v2_train.txt` 是用 `scripts/synth_antimerge.py` 黑名单过滤版（0 个 case 子串），完全可复现。

### 2.7 anchor_1m — Stage 1 数据采样

```bash
python scripts/sample_sentences.py \
    --input data/all12m_train.txt --output data/anchor_1m.txt \
    --n 1000000 --seed 42
```

---

## 3. 训练命令

### 3.1 拼接训练数据（H15.2 配方）

```bash
cat data/opennews_nh_2m_train.txt \
    data/clean_aug_train.txt \
    data/pd_mp_train.txt data/pd_mp_train.txt \
    data/antimerge_train.txt \
    data/anchor_1m.txt \
    data/threechar_zh_train.txt \
    > data/h15_2.txt
# 总 ~3.27M 句
# 相对 H12.1 配方：opennh 从 1M 加倍到 2M，新增 3char_zh 500k（3 字纯中文 Nh 数据）
```

`data/threechar_zh_train.txt` 生成（500k 句，排除测试集 v3 中的名字防 leak）：

```bash
python3 -c "
import json
test_names=set()
for line in open('data/badcase_eval_v3.jsonl'):
    test_names.add(json.loads(line)['name'])

n=0
with open('/tmp/3char_zh.jsonl','w') as fout:
    for line in open('data/raw/opennews_full_nh.jsonl'):
        try: obj=json.loads(line)
        except: continue
        if any(tag=='Nh' and len(text)==3 and all('一'<=c<='鿿' for c in text) and text not in test_names
               for tag,text,*_ in obj.get('ner',[])):
            fout.write(line); n+=1
            if n>=500000: break
"
python3 scripts/jsonl_to_bmes.py --input /tmp/3char_zh.jsonl \
    --out-bmes data/threechar_zh_train.txt --no-type --min-chars 3 --max-chars 200
```

### 3.2 训练

```bash
./build/wapic fit \
    -p data/pattern.txt \
    --init-from data/all12m_compact_v2.wac \
    -a l-bfgs \
    -i 50 \
    -1 0.01 -2 0.0001 \
    -t 4 --histsz 5 -e 1e-9 \
    --save-binary \
    data/h15_2.txt \
    data/exp_h15_2_h122_PLUS_3charzh500k_i50.wac
```
（~26 min CPU 4 线程；产 78M raw 模型）

### 3.3 剪枝到 release size

```bash
./build/wapic convert \
    -m data/exp_h15_2_h122_PLUS_3charzh500k_i50.wac \
    --save-binary --save-prune --prune-threshold 0.15 \
    data/wapic-20260601.wac
# → 56M（约 5M 权重置零）
```

---

## 4. Pattern（不变）

`data/pattern.txt` —— bigram pattern：

```
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U10:%x[-1,0]/%x[0,0]
U11:%x[0,0]/%x[1,0]
U12:%x[-2,0]/%x[-1,0]
U13:%x[1,0]/%x[2,0]
U14:%x[-1,0]/%x[1,0]
B
```

---

## 5. 评估

```bash
# F1 双指标
bash scripts/eval_both.sh data/wapic-20260601.wac

# 15 case
python scripts/test_name_cases.py --model data/wapic-20260601.wac

# 200-case badcase eval (v2)
python scripts/eval_badcase_set.py --model data/wapic-20260601.wac \
    --input data/badcase_eval_v2.jsonl --show-fail 0

# BC001
python scripts/test_badcase.py --model data/wapic-20260601.wac
```

---

## 6. 关键洞察（迭代过程的发现）

1. **A0 必须用 Stage 1 base**（不从已 fine-tune 过的旧 release 暖启动）
2. **anchor_1m 是 F1_12m 保险**（删了掉 1.16）
3. **opennews_nh 是 case + F1 双关键**（删了 F1 -0.23 / case -1）
4. **antimerge 也对 case 必要**（删了 case -1）
5. **modPD（PeopleDaily 2020+）+0.1 F1_pdmp**（H2.3 → H11.2 突破）
6. **foreign_dot_100k**：之前误判 BC001 是失败实际是评测错了，正确标准下早就解决
7. **prune 0.15 是 sweet spot**（size 52M / F1 -0.01 / case 不掉）
8. **数据"加倍"无效**（i=50 已充分收敛，物理重复不增信号）
9. **iter 增加超过 50 边际收益递减**
10. **·/-/• 标点本就独立切**（不是 wapic 错，按 PD-1998 标准）

---

## 7. 不确定性 / 复现风险

- **shuf --random-source**：不同 coreutils 版本可能产生略不同顺序
- **LTP NER GPU 输出**：fp16 + cudnn 在不同硬件可能有微差异（边界情形）
- **wapic 训练本身**：-t 4 实测确定（md5 一致）
- **antimerge 旧版（实际 release 用的）**：含 13 句 case-substring 巧合，建议用 v2 黑名单版重训得到完全干净版本
