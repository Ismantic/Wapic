# Wapic

[![PyPI](https://img.shields.io/pypi/v/wapic)](https://pypi.org/project/wapic/)
[![Python](https://img.shields.io/pypi/pyversions/wapic)](https://pypi.org/project/wapic/)
[![CI](https://github.com/Ismantic/Wapic/actions/workflows/ci.yml/badge.svg)](https://github.com/Ismantic/Wapic/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Ismantic/Wapic)](LICENSE)

Wapic 是一个 C++17 线性链 CRF 中文分词工具，支持 SGD-L1 与
L-BFGS/OWL-QN 训练、BMES 解码、批量推理和 Python API。

## 快速开始

预编译 wheel 支持 CPython 3.9–3.14，以及 Linux、Windows、Intel macOS 和
Apple Silicon macOS。安装时会自动安装约 30 MB 的默认模型，无需编译或另外下载：

```bash
pip install wapic
```

```python
import wapic

segmenter = wapic.Segmenter()
print(segmenter.segment("中华人民共和国成立了"))
# ['中华人民共和国', '成立', '了']
```

## Python API

```python
import wapic

segmenter = wapic.Segmenter()

# 单句分词
words = segmenter.segment("中国AI模型2.0")

# 批量多核推理
batch = segmenter.segment_batch(["第一句", "第二句"])

# 字符与 BMES 标签
chars, tags = segmenter.tag("中华人民共和国")

# 显式加载自己训练的模型
custom = wapic.Segmenter("/path/to/custom.wac")
```

完整接口见 [API.md](API.md)。默认模型来自
[Ismantic/Wapic-CWS](https://huggingface.co/Ismantic/Wapic-CWS)，并通过独立的
[`wapic-cws-model`](https://pypi.org/project/wapic-cws-model/) 包发布。

## CLI 与源码构建

训练和命令行推理需要从源码构建。请使用 Release 模式，训练和推理性能依赖编译器优化：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

命令行程序使用前需下载模型：

```bash
python3 -m pip install huggingface_hub
python3 scripts/download.py model
```

模型保存到 `data/model/wapic-cws.wac`。交互式分词：

```bash
./build/wapic -m data/model/wapic-cws.wac
```

批量 BMES 标注：

```bash
./build/wapic test -m data/model/wapic-cws.wac \
  input_chars.txt output_tags.txt
```

输入文件每行一个字符，空行分句。

开发 Python 扩展时也可从仓库安装或直接构建：

```bash
pip install .
python3 -m pip install pybind11
cmake -B build_py -DWAPIC_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build_py
PYTHONPATH=build_py/python python3 -c 'import wapic; print(wapic.Segmenter("data/model/wapic-cws.wac").segment("中华人民共和国"))'
```

## 评估

```bash
python3 scripts/download.py data          # 添加 --full 下载完整训练集
python3 scripts/test.py data/model/wapic-cws.wac
```

发布模型在 Peopeley 和 自建测试集上的 F1 分别为 98.01 和 97.95。完整训练数据与两阶段
warm-start 配方见数据集仓库 [Ismantic/Wapic-CWS-Data](https://huggingface.co/datasets/Ismantic/Wapic-CWS-Data)。

## 训练

除了使用发布模型，也可以按 [TUTORIAL.md](TUTORIAL.md) 用公开的人民日报 1998
语料从零训练一个分词模型（1–5 月训练 / 6 月测试，单机约 3 分钟得到 F1≈97.4）。
涉及脚本：`scripts/convert.py`（PFR→jsonl）、`scripts/prepare.py`（jsonl→BMES）。
仓库内的 PFR 语料不适用 MIT 许可证，详情见 [data/README.md](data/README.md)。

## 文档

CRF、前向后向算法、梯度计算以及 L-BFGS/OWL-QN 的原理讲解见《底层实现：文本处理》的
[中文分词：高级篇](https://ismantic.github.io/text/wapic.html)。

## License

源码与发布模型采用 MIT 许可证。`data/PeopleDaily1998.zip` 的版权和使用条件独立于
源码许可证。
