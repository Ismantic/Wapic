# Release Reproduction

Git 仓库只维护 Wapic 源码和轻量工具。发布模型、冻结训练集和评估集以
Hugging Face 为唯一数据源，避免仓库中的历史实验文件与最终 retag2 数据混淆。

## Published Artifacts

- `Ismantic/wapic-cws`
  - `wapic-20260605.wac`: 主发布模型
  - `wapic-20260606-base.wac`: Stage-1 warm-start 模型
  - `pattern.txt`: 发布模型使用的特征模板
- `Ismantic/wapic-cws-data`
  - `train/stage1_train.*`: Stage-1 冻结训练集
  - `train/stage2_train.*`: Stage-2 冻结训练集
  - `test/pdmp_test.*`: PD-1998 modern+punct 评估集
  - `test/12m_test.*`: 12M 泛化评估集

数据集仓库中的 `RELEASE_TRAINING_DATA.md` 记录完整训练参数、数据来源和
warm-start 顺序。

## Build and Download

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
uv pip install huggingface_hub
python3 scripts/download_model.py
python3 scripts/download_dataset.py
```

`download_dataset.py` 默认只下载评估所需文件。需要完整训练包时运行：

```bash
python3 scripts/download_dataset.py --full
```

## Verify the Release

```bash
bash scripts/evaluate.sh data/model/wapic-20260605.wac
```

预期结果：

```text
F1_pdmp=97.71 F1_12m=97.49
```

指标波动或重新训练结果必须同时报告PDMP和12M结果。
