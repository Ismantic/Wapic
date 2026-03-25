# Wapix

基于 [Wapiti](https://wapiti.limsi.fr/) 的 C++ 重构实现，支持 SGD-L1 和 L-BFGS (OWL-QN) 两种优化算法的条件随机场 (CRF) 序列标注工具。附带基于 1998 年人民日报语料训练的中文分词模型。

## 特性

- **两种优化器**
  - SGD-L1 — 随机梯度下降 + 累积 L1 惩罚，适合大规模稀疏特征
  - L-BFGS — 限制内存 BFGS + OWL-QN 扩展，支持 L1/L2 正则化

- **CRF 核心算法**
  - Forward-Backward 算法计算边际概率和梯度
  - Viterbi 解码寻找最优标签序列
  - 基于模板的特征提取（unigram / bigram / both）

- **训练特性**
  - L1 正则化的累积惩罚策略（懒更新，高效处理百万级稀疏特征）
  - L-BFGS 的两阶段递归搜索方向计算
  - Wolfe 条件线搜索 / Armijo 规则
  - 正交投影保证 L1 权重符号一致性
  - Token/Sentence 级错误率监控和收敛检测

- **高效数据结构**
  - 自定义 Binary Trie 管理特征和标签词表
  - 稀疏模型存储（仅保存非零权重）

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 使用

### 训练

```bash
./build/src/wapix train \
    -p patterns.txt \
    -a l-bfgs \
    -1 0.5 -2 0.0001 \
    -i 100 \
    train.txt model.crf
```

### 推理

```bash
./build/src/wapix label \
    -m model.crf \
    test.txt result.txt
```

### 交互式分词 (REPL)

```bash
./build/src/wapix repl -m data/model_v2.crf
```

```
Loading model... done.
IsmaWapiti REPL. Type Chinese text, press Enter. Ctrl+D to quit.
>>> 中华人民共和国是一个伟大的国家
中华人民共和国 是 一个 伟大 的 国家
>>> 北京大学和清华大学是中国最好的两所高校
北京大学 和 清华大学 是 中国 最 好 的 两 所 高校
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-a, --algo` | 优化器：`sgd-l1`, `l-bfgs` | `l-bfgs` |
| `-p, --pattern` | 特征模板文件 | - |
| `-m, --model` | 模型文件 | - |
| `-i, --maxiter` | 最大迭代次数 | 100 |
| `-1, --rho1` | L1 正则化系数 | 0.5 |
| `-2, --rho2` | L2 正则化系数 | 0.0001 |
| `-w, --stopwin` | 收敛检测窗口大小 | 5 |
| `-e, --stopeps` | 收敛阈值 | 0.02 |
| `--eta0` | SGD 初始学习率 | 0.8 |
| `--alpha` | SGD 学习率衰减 | 0.85 |
| `--histsz` | L-BFGS 历史大小 | 5 |
| `--maxls` | L-BFGS 最大线搜索次数 | 40 |

### 数据格式

训练数据为空格分隔的列式文本，每行一个 token，空行分隔句子：

```
观测列1 观测列2 ... 标签
观测列1 观测列2 ... 标签

观测列1 观测列2 ... 标签
...
```

### 特征模板

模板文件定义特征提取规则，格式为 `%[u|b|*][offset,column]`：

- `u` — unigram 特征（观测相关）
- `b` — bigram 特征（标签转移相关）
- `*` — 同时作为 unigram 和 bigram 特征

```
# Unigram
U00:%x[-1,0]
U01:%x[0,0]
U02:%x[1,0]

# Bigram
B
```

## 中文分词模型

仓库附带一个预训练的 CRF 中文分词模型 (`data/model_v2.crf`)。

### 训练数据

1998 年人民日报标注语料（`data/1998-*.txt`），JSON 格式，包含原文和分词标注。

- 训练集：1998 年 1-5 月，约 10.3 万句
- 测试集：1998 年 6 月，约 2.1 万句

### 标注方案

采用 **BMES** 四标签字级标注：

| 标签 | 含义 | 示例 |
|------|------|------|
| B | 词首字 | **中**华 |
| M | 词中字 | 中**华**人民共**和** |
| E | 词尾字 | 中华人民共和**国** |
| S | 单字词 | **的**、**是** |

使用 `scripts/prepare_data.py` 将 JSON 语料转换为 CRF 训练格式（每字一行，空行分句）。

### 特征模板

模板文件 `data/pattern.txt`，共 11 个特征：

| 模板 | 含义 |
|------|------|
| U00-U04 | 当前字 ±2 窗口内的单字特征 |
| U05 | 前字 + 当前字 |
| U06 | 当前字 + 后字 |
| U07 | 前第2字 + 前字 |
| U08 | 后字 + 后第2字 |
| U09 | 前字 + 后字（跳字特征） |
| B | 标签转移特征 |

### 模型效果

使用 L-BFGS 优化器训练，L1=0.5, L2=0.0001，79 轮收敛。

在 1998 年 6 月测试集上的评估结果：

| 指标 | 数值 |
|------|------|
| Token 准确率 | 97.69% |
| 词级 Precision | 97.33% |
| 词级 Recall | 97.14% |
| **词级 F1** | **97.24%** |
| 模型大小 | 80 MB |

### 复现训练

```bash
# 1. 预处理语料
python3 scripts/prepare_data.py

# 2. 训练
./build/src/wapix train \
    -p data/pattern.txt \
    -a l-bfgs \
    -1 0.5 -2 0.0001 \
    -i 100 \
    data/train.txt data/model_v2.crf

# 3. 评估
./build/src/wapix label \
    -m data/model_v2.crf \
    data/test_nolabel.txt data/test_result.txt
python3 scripts/evaluate.py
```

## 项目结构

```
src/
  option.h/cc         - 命令行参数解析
  model.h/cc          - 模型容器（权重、偏移量、序列化）
  data.h/cc           - 数据加载、特征索引构建
  pattern.h/cc        - 特征模板解析与匹配
  state.h/cc          - Forward-Backward、梯度计算
  score.h/cc          - Viterbi 解码
  optimize.h/cc       - SGD-L1 和 L-BFGS 优化器
  progress.h/cc       - 训练进度监控与收敛检测
  trie.h/cc           - Binary Trie
  sentence.h          - 句子数据结构
  misc.h/cc           - 工具函数
data/
  1998-*.txt          - 人民日报标注语料
  pattern.txt         - 特征模板
  model_v2.crf        - 预训练分词模型
scripts/
  prepare_data.py     - 语料预处理（JSON → BMES）
  evaluate.py         - 分词评估（P/R/F1）
  analyze_model.py    - 模型大小分析
  repl.py             - Python 版 REPL（已由 C++ REPL 替代）
docs/
  Wapiti.md           - CRF 算法原理详解
```

## 算法原理

详见 [Wapiti.md](Wapiti.md)，包含：

- CRF 概率模型与势函数
- Forward-Backward 算法推导
- 梯度计算（模型期望 - 经验期望）
- SGD + 累积 L1 惩罚
- L-BFGS + OWL-QN 扩展

## License

MIT
