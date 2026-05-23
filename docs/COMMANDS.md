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

## Sprint 2 dummy and Logistic Regression baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_logistic_regression.yaml
```

This command also writes Logistic Regression diagnostics under:

```text
outputs/diagnostics/sprint2/
```

## Sprint 2 XGBoost baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
```

This command also writes:

```text
outputs/diagnostics/sprint2/xgboost_feature_column_audit.csv
outputs/diagnostics/sprint2/xgboost_feature_importance.csv
```

## Sprint 2 tabular MLP baselines

```bash
uv run python scripts/train.py --config configs/experiments/baseline_mlp.yaml
```

This command also writes:

```text
outputs/diagnostics/sprint2/tabular_mlp_training_summary.csv
outputs/diagnostics/sprint2/tabular_mlp_feature_column_audit.csv
```

## Sprint 2 sequence-only CNN/BiLSTM baselines

```bash
uv run python scripts/train.py --config configs/experiments/sequence_cnn_bilstm.yaml
```

This command also writes:

```text
outputs/diagnostics/sprint2/sequence_input_audit.csv
outputs/diagnostics/sprint2/sequence_training_summary.csv
```

## Sprint 2 CNN + F3/F4 late-fusion baselines

```bash
uv run python scripts/train.py --config configs/experiments/sequence_cnn_late_fusion.yaml
```

This command also writes:

```text
outputs/diagnostics/sprint2/sequence_late_fusion_input_audit.csv
outputs/diagnostics/sprint2/sequence_late_fusion_training_summary.csv
```

## Later sprint command pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
