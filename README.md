# Wapic

[Wapiti](https://wapiti.limsi.fr/) 的 C++ 重构实现，支持 SGD-L1 和 L-BFGS (OWL-QN) 两种优化算法的条件随机场 (CRF) 序列标注工具。

附带基于人民日报 1998 + LTP/base1 12M 语料三阶段训练的中文分词模型。

## 预训练模型

| 文件 | 大小 | F1 (PD-1998) | 备注 |
|---|---|---|---|
| **`data/wapic-20260529.wac`** | **47 MB** | **~98** | 推荐 — 三阶段训练，现代中文标准（人名整体、标点分开） |

```bash
./build/wapic -m data/wapic-20260529.wac
```

```
>>> 中华人民共和国是一个伟大的国家
中华人民共和国 是 一个 伟大 的 国家
>>> 李镇全今天接受了记者的采访
李镇全 今天 接受 了 记者 的 采访
```

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 推理（批处理）

```bash
./build/wapic test -m data/wapic-20260529.wac input_chars.txt output_tags.txt
```

`input_chars.txt` 每行一个字符，空行分句。`output_tags.txt` 输出 BMES 标签。

## 训练

详细训练流程见 [`HANDOFF.md`](HANDOFF.md)（三阶段：12M LTP base → name-rich aug → PD modern+punct）。

简单 PD-only 训练：

```bash
cd scripts
make prepare   # 语料 JSON → BMES 格式
make fit       # 训练 CRF 模型
make test      # 标注测试集
make review    # 评估 P/R/F1
make           # 以上全部
```

训练参数在 `scripts/Makefile` 中调整。完整结果对比见 [`RESULTS.md`](RESULTS.md)。

## 性能

| 项 | Wapic | LTP/base1（基线） |
|---|---|---|
| 模型大小 | 47 MB | 1.5 GB |
| F1 (PD-1998) | ~98 | 97.82 |
| 推理速度 (CPU) | ~3400 sent/s | 428 sent/s (GPU+fp16) |

## License

MIT
