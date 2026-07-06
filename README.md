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

pip 安装（会现场编译 C++ 扩展，需 CMake ≥ 3.14 与 C++17 编译器）：

```bash
pip install git+https://github.com/Ismantic/Wapic.git
# 或在仓库根目录：pip install .
```

用法（模型见上面「下载模型」）：

```python
import wapic
seg = wapic.Segmenter("data/model/wapic-cws.wac")
print(seg.segment("中华人民共和国是一个伟大的国家"))
# ['中华人民共和国', '是', '一个', '伟大', '的', '国家']
```

开发时也可源码构建后用 `PYTHONPATH` 直接引用：

```bash
uv pip install pybind11
cmake -B build_py -DWAPIC_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build_py
PYTHONPATH=build_py/python python3 -c 'import wapic; print(wapic.Segmenter("data/model/wapic-cws.wac").segment("中华人民共和国"))'
```

## 评估

```bash
python3 scripts/download.py data          # 加 --full 下载完整训练集
python3 scripts/test.py data/model/wapic-cws.wac
```

发布模型在 PDMP/12M retag2 测试集上的 F1 为97.70/97.48。完整训练数据与两阶段
warm-start 配方见数据集仓库 [Ismantic/wapic-cws-data](https://huggingface.co/datasets/Ismantic/wapic-cws-data)。

## 训练自己的模型

除了使用发布模型，你也可以按 [TUTORIAL.md](TUTORIAL.md) 用公开的人民日报 1998
语料从零训练一个分词模型（1–5 月训练 / 6 月测试，单机约 3 分钟得到 F1≈97.4）。
涉及脚本：`scripts/convert.py`（PFR→jsonl）、`scripts/prepare.py`（jsonl→BMES）。

## License

MIT
