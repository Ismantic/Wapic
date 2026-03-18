# IsmaWapiti

条件随机场 (CRF) 序列标注的 C++ 实现，支持 SGD-L1 和 L-BFGS (OWL-QN) 两种优化算法。

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
./build/src/isma_wapiti_test train \
    -p patterns.txt \
    -a l-bfgs \
    -1 0.5 -2 0.0001 \
    -i 100 \
    train.txt model.crf
```

### 推理

```bash
./build/src/isma_wapiti_test label \
    -m model.crf \
    test.txt result.txt
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
