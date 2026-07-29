# Wapic

English | [中文](README.md)

[![PyPI](https://img.shields.io/pypi/v/wapic)](https://pypi.org/project/wapic/)
[![Python](https://img.shields.io/pypi/pyversions/wapic)](https://pypi.org/project/wapic/)
[![CI](https://github.com/Ismantic/Wapic/actions/workflows/ci.yml/badge.svg)](https://github.com/Ismantic/Wapic/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Ismantic/Wapic)](LICENSE)

Wapic is a C++17 reimplementation of
[Wapiti](https://github.com/Jekub/Wapiti) for Chinese word segmentation. It
provides a linear-chain CRF, SGD-L1 and L-BFGS/OWL-QN training, BMES decoding,
batch inference, and a Python API.

## Quick Start

Prebuilt wheels support CPython 3.9-3.14 on Linux, Windows, Intel macOS, and
Apple Silicon macOS. Installation automatically includes the approximately
30 MB default model, with no compilation or separate download required:

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

# Segment one sentence
words = segmenter.segment("中国AI模型2.0")

# Batch inference using multiple CPU cores
batch = segmenter.segment_batch(["第一句", "第二句"])

# Characters and BMES tags
chars, tags = segmenter.tag("中华人民共和国")

# Explicitly load a custom model
custom = wapic.Segmenter("/path/to/custom.wac")
```

See [API.md](API.md) for the complete API. The default model comes from
[Ismantic/Wapic-CWS](https://huggingface.co/Ismantic/Wapic-CWS).

## Building

Training and command-line inference require a source build. Use Release mode,
as training and inference performance depend on compiler optimization:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Download the model before using the command-line program:

```bash
python3 -m pip install huggingface_hub
python3 scripts/download.py model
```

The model is saved to `data/model/wapic-cws.wac`. For interactive segmentation:

```bash
./build/wapic -m data/model/wapic-cws.wac
```

For batch BMES tagging:

```bash
./build/wapic test -m data/model/wapic-cws.wac \
  input_chars.txt output_tags.txt
```

The input file should contain one character per line, with blank lines separating
sentences.

When developing the Python extension, install it from the repository or build it
directly:

```bash
pip install .
python3 -m pip install pybind11
cmake -B build_py -DWAPIC_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build_py
PYTHONPATH=build_py/python python3 -c 'import wapic; print(wapic.Segmenter("data/model/wapic-cws.wac").segment("中华人民共和国"))'
```

## Evaluation

```bash
python3 scripts/download.py data          # Add --full to download the full training set
python3 scripts/test.py data/model/wapic-cws.wac
```

The released model reaches F1 scores of 98.01 on PeopleDaily and 97.95 on the
custom test set. The complete training data and two-stage warm-start recipe are
available in the
[Ismantic/Wapic-CWS-Data](https://huggingface.co/datasets/Ismantic/Wapic-CWS-Data)
dataset repository.

## Training

In addition to using the released model, you can follow
[TUTORIAL.md](TUTORIAL.md) to train a segmentation model from scratch on the
public People's Daily 1998 corpus. It uses January-May for training and June for
testing, and reaches approximately F1 97.4 in about three minutes on one machine.
The relevant scripts are `scripts/convert.py` for PFR-to-JSONL conversion and
`scripts/prepare.py` for JSONL-to-BMES conversion. The PFR corpus included in the
repository is not covered by the MIT License; see
[data/README.md](data/README.md) for details.

## Documentation

For explanations of CRFs, the forward-backward algorithm, gradient computation,
and L-BFGS/OWL-QN, see the
[Advanced Chinese Word Segmentation](https://ismantic.github.io/text/wapic.html)
chapter in the Chinese-language book "Low-Level Implementation: Text
Processing."

## Notes

Wapic is derived from the
[Wapiti](https://github.com/Jekub/Wapiti) linear-chain CRF toolkit developed by
Thomas Lavergne and collaborators. It modernizes the core design in C++17 and
adds Chinese word segmentation, batch inference, and Python bindings. Wapic is
maintained independently and is not an official successor to Wapiti.

When using this project for research, please also refer to the original Wapiti
paper: Thomas Lavergne, Olivier Cappé, and François Yvon, "Practical Very Large
Scale CRFs," ACL 2010.

## License

The source code and released model are licensed under the MIT License. The
copyright and terms of use for `data/PeopleDaily1998.zip` are separate from the
source-code license. The original Wapiti copyright belongs to CNRS (2009-2013)
and is governed by its
[BSD License](https://github.com/Jekub/Wapiti/blob/master/COPYING).
