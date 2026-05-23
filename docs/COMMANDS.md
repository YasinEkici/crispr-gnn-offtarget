# Commands

## Setup

```bash
uv sync
```

### macOS XGBoost runtime

XGBoost requires the OpenMP runtime on macOS. If `import xgboost` fails with a missing `libomp.dylib`, install:

```bash
brew install libomp
```

## Tests

```bash
uv run pytest -q
```

## Sprint 0 smoke commands

```bash
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml --debug --max-epochs 1
uv run python scripts/evaluate.py --config configs/experiments/gcn_minimal.yaml --debug
```

## Sprint 2 split artifacts

```bash
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml
```

## Sprint 2 feature catalog

```bash
uv run python scripts/build_features.py --config configs/data/mak2022.yaml
```

## Later sprint command pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
