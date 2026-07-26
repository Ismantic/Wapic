# Wapic Python API

本文档描述 `wapic` Python 模块的接口。模块由 `src/pip.cc` 通过
PyBind11 暴露，提供基于 Wapic CRF 模型的中文分词、BMES 标注与批量推理。

## 安装

从 GitHub 安装：

```bash
pip install git+https://github.com/Ismantic/Wapic.git
```

也可以在仓库根目录安装：

```bash
pip install .
```

安装过程会现场编译 C++ 扩展，需要 CMake 3.14 或更高版本、C++17 编译器和
Python 3.9 或更高版本。

发布模型可用仓库脚本下载：

```bash
uv pip install huggingface_hub
python3 scripts/download.py model
```

模型默认保存到 `data/model/wapic-cws.wac`。

---

## `wapic.Segmenter`

加载 Wapic CRF 模型并进行分词。一个 `Segmenter` 实例可以重复处理多段文本。

### 构造

```python
segmenter = wapic.Segmenter(model_path: str)
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `model_path` | `str` | Wapic `.wac` 模型文件路径。构造时立即加载模型。 |

模型不存在、无法读取或格式无效时，构造函数会抛出异常。

```python
import wapic

segmenter = wapic.Segmenter("data/model/wapic-cws.wac")
```

### 方法

#### `segment(text: str) -> list[str]`

对文本进行默认分词。

输入会先按空白、汉字、拉丁字母、数字和标点等类别预切分，仅连续汉字段送入
CRF；其他非空白片段直接作为 token 返回。空白用于分隔，不包含在结果中。

```python
segmenter.segment("中华人民共和国是一个伟大的国家")
# → ['中华人民共和国', '是', '一个', '伟大', '的', '国家']

segmenter.segment("中国AI模型2.0")
# → 中文片段由 CRF 分词；'AI'、'2'、'.'、'0' 按预切分结果返回
```

这是普通推理时推荐使用的接口。

#### `segment_batch(texts: list[str]) -> list[list[str]]`

并行处理多段文本，输出顺序与输入顺序一致。每个元素的结果与分别调用
`segment()` 相同。

该方法使用 OpenMP 多线程，并在计算期间释放 Python GIL，适合较大的批量推理
任务。小批量调用不一定比逐条调用更快。

```python
texts = ["第一句话", "第二句话", "中国AI模型"]
results = segmenter.segment_batch(texts)
# → [
#     ['第一', '句', '话'],
#     ['第二', '句', '话'],
#     [...],
# ]
```

#### `segment_raw(text: str) -> list[str]`

不做预切分，将整段文本按 Unicode 字符逐个送入 CRF，再根据 BMES 标签组合成词。
结果会保留输入中的所有字符，包括空格。

```python
words = segmenter.segment_raw("abc 123 中国")
assert "".join(words) == "abc 123 中国"
```

该接口主要用于需要与训练时逐字符输入路径保持一致的场景。一般中文分词应优先
使用 `segment()`。

#### `tag(text: str) -> tuple[list[str], list[str]]`

返回逐字符序列及对应的 BMES 标签：

- `B`：词首（Begin）
- `M`：词中（Middle）
- `E`：词尾（End）
- `S`：单字成词（Single）

两个列表长度相同；空字符串返回 `([], [])`。输入中的空白字符会保留在
`chars` 中，并标记为 `S`。

```python
chars, tags = segmenter.tag("中国")
print(chars)  # ['中', '国']
print(tags)   # 例如 ['B', 'E']
```

与 `segment_raw()` 一样，`tag()` 不使用默认的文本预切分路径。

#### `word_starts(text: str) -> list[int]`

返回默认预切分与 CRF 分词结果中每个词的字符起始位置，并在末尾追加
`len(chars)` 作为哨兵。索引按 Unicode 字符计算，不是 UTF-8 字节偏移。

空白不视为词，但会计入字符位置。

```python
segmenter.word_starts("abc 123 中国")
# → [0, 4, 8, 10]
```

其中 `0`、`4`、`8` 分别是 `abc`、`123`、`中国` 的起始位置，`10` 是末尾
哨兵。该接口适合构造 Whole Word Masking（WWM）边界。

### 只读属性

#### `label_count -> int`

返回模型的标签数量。中文分词模型通常使用 `B`、`M`、`E`、`S` 四个标签，但
具体数值取决于加载的模型。

```python
segmenter.label_count
# → 4
```

#### `feature_count -> int`

返回模型中存储的特征数量。

```python
segmenter.feature_count
# → 模型的特征总数
```

---

## 完整示例

```python
import wapic

seg = wapic.Segmenter("data/model/wapic-cws.wac")

# 默认分词
text = "中华人民共和国是一个伟大的国家"
words = seg.segment(text)
print(words)
# ['中华人民共和国', '是', '一个', '伟大', '的', '国家']

# BMES 标注
chars, tags = seg.tag("中国")
print(list(zip(chars, tags)))

# 批量并行分词
batch = seg.segment_batch([
    "第一句话",
    "第二句话",
    "中国AI模型2.0",
])
for words in batch:
    print(words)

# WWM 词首位置
starts = seg.word_starts("abc 123 中国")
print(starts)
# [0, 4, 8, 10]

print(seg.label_count)
print(seg.feature_count)
```

---

## 注意事项

1. **模型必须在构造时成功加载**：`Segmenter` 没有单独的 `load()` 方法；更换
   模型需要创建新的实例。
2. **默认分词与原始路径不同**：`segment()` 和 `segment_batch()` 会先做文本
   预切分；`segment_raw()` 与 `tag()` 则将非空白片段按字符直接交给 CRF。
3. **空白处理**：`segment()` 会丢弃空白；`segment_raw()` 和 `tag()` 保留
   空白；`word_starts()` 跳过空白词但会把空白计入索引。
4. **字符索引不是字节索引**：`word_starts()` 的位置基于 UTF-8 解码后的 Unicode
   字符序列。
5. **长文本分块**：逐字符 CRF 路径会以最多 1024 个字符为一块进行推理，块边界
   会被视为句子边界。普通短文本不受影响。
6. **线程行为**：单条推理方法在同一实例内串行保护；批量推理应使用
   `segment_batch()`，它为各线程创建独立的打分器并释放 GIL。
