# Repository Guidelines

## Project Structure

Wapic is a C++17 CRF sequence-labeling tool for Chinese word segmentation.

- `src/`: core CRF library, CLI, optimizer, decoder, and pybind11 binding.
- `python/wapic/`: Python package entry point.
- `scripts/`: model/data download and release evaluation tools.
- `data/pattern.txt`: default feature template used for training.

Models and datasets live on Hugging Face, not in Git. Downloaded files under
`data/model/` and `data/dataset/` are ignored.

## Build and Development Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
python3 scripts/download_model.py
./build/wapic -m data/model/wapic-cws.wac
```

Use Release builds because training and inference depend heavily on compiler
optimization. For the optional Python module, install pybind11 and follow the
command in `README.md`.

## Coding Style

Use four-space indentation in C++ and Python. Follow existing C++ naming:
`PascalCase` for types and public methods, `snake_case` for local helpers, and
trailing underscores for private fields. Keep headers paired with `.cc`
implementations under `src/`. Python files and functions use `snake_case`.
No formatter is configured, so match adjacent code and keep changes focused.

## Testing

After C++ changes, build from a clean CMake directory and exercise the affected
CLI mode. For segmentation behavior, run:

```bash
python3 scripts/download_dataset.py
bash scripts/evaluate.sh data/model/wapic-cws.wac
```

Report both PDMP and 12M F1. The expected release baseline is `97.71` and
`97.49`.

## Commits and Pull Requests

Use imperative subjects such as `Fix`, `Add`, `Remove`, or `Release`. Keep
commits focused and never include downloaded models, datasets, build trees, or
experiment logs. Pull requests should explain behavior changes, list validation
commands, report metric deltas, and link relevant issues.
