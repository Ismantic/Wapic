# Wapic

[Wapiti](https://wapiti.limsi.fr/) 的 C++ 重构实现，支持 SGD-L1 和 L-BFGS (OWL-QN) 两种优化算法的条件随机场 (CRF) 序列标注工具。

附带基于 1998 年人民日报语料训练的中文分词模型。

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## 训练与评估

```bash
cd scripts
make prepare   # 语料 JSON → BMES 格式
make fit       # 训练 CRF 模型
make test      # 标注测试集
make review    # 评估 P/R/F1
make           # 以上全部
```

训练参数在 `scripts/Makefile` 中调整。

## 交互式体验

```bash
./build/wapic -m data/cut.wac
```

```
>>> 中华人民共和国是一个伟大的国家
中华人民共和国 是 一个 伟大 的 国家
```

## License

MIT
