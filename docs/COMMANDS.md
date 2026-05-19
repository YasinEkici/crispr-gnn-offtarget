# Commands

## Setup

```bash
uv sync
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

## Later sprint command pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
