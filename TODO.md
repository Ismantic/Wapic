# Wapic TODO / 待办

## 1. Unicode 标点统一（・ vs ·）

### 现象
- 输入 `菲利波・加斯佩里`（用 U+30FB Katakana middle dot）→ 模型不切，输出整个连一起
- 输入 `菲利波·加斯佩里`（用 U+00B7 Middle dot）→ 模型正确切成 3 token

### 根因
| 字符 | Unicode | aug 出现 | mp 出现 | 12M LTP | 模型行为 |
|---|---|---|---|---|---|
| `·` | U+00B7 | 7250 (split) | 5892 (split) | 66k (split) | ✓ 切 |
| `・` | U+30FB | 0 | 0 | 238 (LTP 当人名连) | ✗ 不切 |

PD-1998 没 ・，12M LTP 把 ・ 当外国人名一部分连标，所以模型继承了"连"的行为。

### 修复方案（最干净）

字符规范化，处处把 ・ → ·：

1. **训练数据预处理**：`scripts/jsonl_to_bmes.py` 加 `text.replace('・', '·')`
2. **推理输入预处理**：`scripts/add_type.py` 同上
3. **REPL 输入归一**（C++ 改动）：可选，但能保证交互体验

也可以扩展正规化表：全角/半角空格、其他相似字符。

### 状态
- ☐ 未实施
- 优先级：低（实际用例罕见）

---

## 2. 三阶段训练数据 jsonl 重建

### 现状（2026-05-29）
- ☑ stage 1（12M LTP）：软链到 `data/raw/ltp_*.jsonl`
- ☑ stage 3（PD modern+punct）：`data/raw/pd_train_{raw,modern,mp}.jsonl` 已生成
- ☑ stage 3 test：`data/raw/pd_test_{raw,modern,mp}.jsonl` 已生成
- ☑ stage 2（name-rich）：`data/raw/name_aug.jsonl` 已生成（OpenNews 268k → 55,723 句）

### 待办
- 等 name_aug.jsonl 跑完，确认 ~50k 句保留
- 把 jsonl 转 BMES：`jsonl_to_bmes.py` 出 `pd_aug_train_v2.txt` / `pd_mp_train_v2.txt`
- （可选）应用规范化 ・→· 后再转

---

## 3. 12M+trigram 训练（搁置）

之前尝试 trigram pattern 在 12M 上训练，本机 + u8700 都 OOM（峰值 27-30 GB）：
- station：earlyoom + Shmem 9.9 GB 限死
- u8700：32 GB 物理不够，需要 swap 或 6M 子集

如要恢复：
- 6M 子集 + trigram，预估 16-18 GB 峰值
- 或加 zram 30 GB swap

李镇全 case 4/5 一致已经接近，可以暂时搁置。

---

## 4. release 模型当前状态

- `data/wapic-20260529.wac`（47 MB）
- F1 = 98.07（pd_mp test）
- 三阶段训练：12M LTP → 267k name-rich → 102k PD modern+punct
- 已加入仓库 + push 到 GitHub

### 改进空间
- type 特征验证为**无效**（F1 同 97.36 对 97.36）
- trigram 还没在 12M 跑通
- 后续可考虑 6M+trigram 或者更精细的 fine-tune 策略
