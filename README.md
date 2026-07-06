# Wapic

Wapic 是一个 C++17 线性链 CRF 序列标注工具，主要用于中文分词。它支持
SGD-L1 与 L-BFGS/OWL-QN 训练、BMES 解码、批量推理和可选的 Python 绑定。

模型与数据不存放在 Git 仓库中：

- 模型：[Ismantic/wapic-cws](https://huggingface.co/Ismantic/wapic-cws)
- 数据：[Ismantic/wapic-cws-data](https://huggingface.co/datasets/Ismantic/wapic-cws-data)

## 构建

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

训练对编译优化较敏感，建议始终使用 Release 构建。

## 下载模型

安装 `huggingface_hub`，然后下载主模型：

```bash
uv pip install huggingface_hub
python3 scripts/download.py model
```

模型默认保存到 `data/model/wapic-cws.wac`。

## 推理

交互式分词：

```bash
./build/wapic -m data/model/wapic-cws.wac
```

批量 BMES 标注：

```bash
./build/wapic test -m data/model/wapic-cws.wac \
  input_chars.txt output_tags.txt
```

输入文件每行一个字符，空行分句。

## Python 绑定

```bash
uv pip install pybind11
cmake -B build_py -DWAPIC_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build_py
PYTHONPATH=build_py/python python3 -c \
  'import wapic; print(wapic.Segmenter("data/model/wapic-cws.wac").cut("中华人民共和国"))'
```

## 评估

```bash
python3 scripts/download.py data
python3 scripts/test.py data/model/wapic-cws.wac
```

发布模型在 PDMP/12M retag2 测试集上的 F1 为97.70/97.48。训练数据与
发布复现说明见 [REPRODUCE.md](REPRODUCE.md)。

## 训练自己的模型

除了使用发布模型，你也可以按 [TRAINING.md](TRAINING.md) 用公开的人民日报 1998
语料从零训练一个分词模型（1–5 月训练 / 6 月测试，单机约 3 分钟得到 F1≈97.4）。
涉及脚本：`scripts/convert.py`（PFR→jsonl）、`scripts/prepare.py`（jsonl→BMES）。

## License

MIT
