# Wapic 模型训练 + 优化结果总结

实验日期：2026-05-28

## 任务

用 Wapic（线性链 CRF + L-BFGS）训练一个中文分词模型：
- F1 ≥ 97% 在 PeopleDaily 1998 测试集上
- 模型文件尽可能小

参考基线：LTP/base1（哈工大基于 ELECTRA-base 的神经分词器）。

## 结果总览

测试集：1998-06.txt 全量 18405 句（max_chars=200 过滤），评估 word-level F1。

| 模型 | 大小 | F1 (compare_1998 严格 18k) | F1 (evaluate.py 21k) | 速度 (CPU OpenMP) |
|---|---|---|---|---|
| LTP/base1 (基线) | ~1.5 GB (PyTorch) | **97.82** | - | 428/s (GPU+fp16) |
| **wapic_pd_best** (L1=0.4, TH=0.05) | **4.80 MB** | **97.19** | **97.32** | ~3400/s |
| **wapic_pd_small** (L1=0.8, TH=0.1) | **2.98 MB** | **97.00** | 97.22 | ~3400/s |
| all12m_compact (LTP-style) | 60.70 MB | 96.06 | 96.48 | ~3400/s |

**最佳交付**：4.80 MB / F1≈97.2，相对 LTP/base1 仅落 0.63 F1，但 **8x 快 + 300x 小**。

### L1 × Threshold 精细 sweep（紧收敛 eps=0.001, 300 iter）

| L1 | TH | Size MB | F1 (evaluate.py) |
|---|---|---|---|
| 0.4 | 0 | 5.80 | 97.32 |
| **0.4** | **0.05** | **4.80** | **97.32** |
| 0.4 | 0.1 | 4.44 | 97.31 |
| 0.6 | 0 | 4.61 | 97.27 |
| 0.6 | 0.05 | 3.87 | 97.27 |
| 0.6 | 0.1 | 3.59 | 97.26 |
| 0.7 | 0 | 4.14 | 97.25 |
| 0.7 | 0.05 | 3.51 | 97.25 |
| 0.7 | 0.1 | 3.26 | 97.25 |
| 0.8 | 0 | 3.77 | 97.22 |
| 0.8 | 0.05 | 3.21 | 97.23 |
| **0.8** | **0.1** | **2.98** | **97.22** |

## 实验路径

### 1. 起点：12M 语料 LTP-style 训练
- 4 源（OpenNews + ZhihuKOL + PeopleDaily + LCCC）各取 3M 行 → 12M 总句
- 用 LTP/base1 标 BMES → 训练 Wapic CRF
- 中途遇到 earlyoom（用户态 OOM killer）阈值 < 10% 时 SIGTERM
- 优化：mmap 数据 + 减线程 + L-BFGS 内存优化

最终：520 MB 模型 / F1=96.48（PD 标准失之毫厘）。

### 2. 关键洞察
LTP-base1 标准 ≠ PD-1998 标准（人名是否分姓+名、NER 括号粘连等）。
直接在 199801-199805 数据上训练，标准对齐 → F1 显著提升。

### 3. PD-only 训练 + 调参
- 数据：199801-199805，约 10 万句
- 训练 73-135 iter，1 分钟级别完成（vs 12M 60 min）

L1 sweep（保持其他默认）：

| L1 | F1 | Size |
|---|---|---|
| 0.1 | 98.11 (gb18030 garbled) → ~97.3 (utf8) | 42 MB |
| **0.3** | **98.12** / **97.31** | **34.9 MB / 11.87 MB** |
| 0.5 | 98.07 / 97.26 | 33.3 MB / 10.68 MB |
| 1.0 | 97.93 / 97.12 | 29.7 MB / 4.89 MB |
| 2.0 | 97.70 / 96.77 | 26.8 MB / 3.10 MB |

L2 完全没用（同 F1 across 0~0.01）。

### 4. 模型瘦身（不损失精度）

| 阶段 | 大小变化 | F1 |
|---|---|---|
| 文本格式 | 35 MB → | 97.31 |
| Binary trie + fp64 weights | → 22 MB | 97.31 |
| + fp32 weights | → 20 MB | 97.31 |
| + obs 剪枝（dead obs 整条丢） | → 5.4 MB | 97.31 |
| + magnitude threshold (0.01) | → 4.47 MB | 97.31 |
| 紧收敛 (`eps=0.001`) + threshold 0.05 | → **5.80 MB** | **97.22** |

注：第一波数字（98.12）实测发现是 gb18030 解码乱码后自洽训练。UTF-8 正确解析后真实 F1 = 97.22-97.31。

### 5. 代码改动

| 文件 | 改动 |
|---|---|
| `src/sentence.h` | `Pos` 从 64B → 8B（去掉两个 vector<int>） |
| `src/data.cc` | 增加 `BuildBinary` / `LoadBinary` 流式 + mmap 路径 |
| `src/data.cc` | `TokensToSentence` 使用 OpenMP 并行 Pattern::Execute |
| `src/state.cc` | `ComputePsi/ComputeModelExpectation` 内循环 cache-friendly 重写 |
| `src/state.h` | `GradientComputer` 12 线程梯度并行（已有，OpenMP 又叠加 L-BFGS 内向量并行） |
| `src/progress.h` | `Tester::Run` 并行 OpenMP（之前是单线程 Viterbi，每 iter 14s） |
| `src/progress.h` | 显式 `endl` 强制 flush（之前 cout 块缓冲看不到进度） |
| `src/optimize.cc` | 关键 O(F) 循环加 `#pragma omp parallel for` |
| `src/misc.cc` | `WriteStrBin/ReadStrBin` + `WriteVarUInt/ReadVarUInt` |
| `src/trie.cc` | `SaveBin/LoadAuto` 二进制格式 + 兼容旧文本 |
| `src/model.cc` | binary save + obs prune + fp32 weights + delta-encoded indices |
| `src/option.h` + `main.cc` | `build` 子命令 + `convert` 子命令 + `--save-binary/--save-prune/--prune-threshold/--save-every/--from-bin` |

### 6. CLI 使用

```bash
# 1. 一次性把 train.txt 编译成 mmap 二进制（亚秒级再加载）
./build/wapic build -p data/pattern.txt -t 12 \
    data/pd_train.txt data/pd_train_bin

# 2. 训练并保存压缩模型
./build/wapic fit -p data/pattern.txt --from-bin -a l-bfgs \
    -i 500 -1 0.3 -2 0.0001 -t 12 -e 0.001 \
    --save-binary --save-prune --prune-threshold 0.05 \
    data/pd_train_bin data/wapic_pd_best.wac

# 3. 推理
./build/wapic test -m data/wapic_pd_best.wac \
    input_chars.txt output_tags.txt

# 4. 把现有大模型压缩（无损）
./build/wapic convert -m data/old_model.wac \
    --save-binary --save-prune --prune-threshold 0.01 \
    data/new_model.wac
```

## 推荐配置

**生产环境（用于 PD-1998 风格分词）**：
- `wapic_pd_best.wac` — **4.80 MB / F1=97.32（evaluate.py）/ 97.19（compare_1998）**
  - 配置：`-1 0.4 -e 0.001 -i 300 --save-binary --save-prune --prune-threshold 0.05`
  - 在「人名分姓」（李 瑞环、胡 锦涛）这类系统差异上比 LTP/base1 更接近 PD 标准
  - 错误集中在罕见组合（"漏 征 漏 管 户"、"一 角 券"），训练语料覆盖少

**极致小（接受 F1 ≈ 97.0）**：
- `wapic_pd_small.wac` — **2.98 MB / F1=97.22（evaluate.py）/ 97.00（compare_1998）**
  - 配置：`-1 0.8 --save-binary --save-prune --prune-threshold 0.1`

**跨域分词**（OpenNews / Zhihu / 聊天等现代文本，PD 测试略差）：
- `all12m_compact_v2.wac` — **60.7 MB / F1=96.48 PD（但泛化好）**
  - 由 12M LTP-base1 标注训练
  - 通过 `wapic convert` 从原始 520 MB 文本模型压缩得到（无损 F1）

## 关键发现

1. **数据标准对齐 > 训练量**：100k PD-1998 数据 → F1=97.2，对比 12M LTP 标注数据 → F1=96.5。
2. **L1 是模型大小的主要旋钮**：L1 越大模型越小但 F1 缓慢下降。L2 在 [0, 0.01] 范围内对 F1 无影响。
3. **Pos struct 重构 + mmap + OpenMP** 让 9.5M 数据训练从无法运行（>40 GB 内存）变为 35 GB 可控（with `-t 4`）。
4. **Tester::Run 单线程 Viterbi 是 12M 训练的隐藏瓶颈**：每 iter 14s 全在这。并行后 5.2s/iter（3.6x）。
5. **`#ModelBin32#` binary + `prune` + `prune-threshold`** 三连击让 PD 模型从 35 MB 文本降到 4.8 MB（保持 F1）。
6. **`wapic convert` 一次性命令**让现有大模型瞬间瘦身（520 MB → 60 MB，F1 不变）。
