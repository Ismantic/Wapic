# Release Reproduction

Git 仓库只维护 Wapic 源码和轻量工具。发布模型、冻结训练集和评估集以
Hugging Face 为唯一数据源，避免仓库中的历史实验文件与最终 retag2 数据混淆。

## Published Artifacts

- `Ismantic/wapic-cws`
  - `model/wapic-cws.wac`: 主发布模型
  - `model/wapic-cws-base.wac`: Stage-1 warm-start 模型
  - `model/pattern.txt`: 发布模型使用的特征模板
- `Ismantic/wapic-cws-data`
  - `dataset/wapic-cws-data-1.*`: Stage-1 冻结训练集
  - `dataset/wapic-cws-data-2.*`: Stage-2 冻结训练集
  - `dataset/wapic-cws-data-test-2.*`: PD-1998 modern+punct 评估集
  - `dataset/wapic-cws-data-test-1.*`: 12M 泛化评估集

数据集仓库中的 `RELEASE_TRAINING_DATA.md` 记录完整训练参数、数据来源和
warm-start 顺序。

## Build and Download

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
uv pip install huggingface_hub
python3 scripts/download.py model
python3 scripts/download.py data
```

`download.py data` 默认只下载评估所需文件。需要完整训练包时运行：

```bash
python3 scripts/download.py data --full
```

## Verify the Release

```bash
bash scripts/evaluate.sh data/model/wapic-cws.wac
```

预期结果：

```text
F1_pdmp=97.70 F1_12m=97.48
```

指标波动或重新训练结果必须同时报告PDMP和12M结果。
