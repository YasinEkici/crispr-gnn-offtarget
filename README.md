# Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction - BTU CENG SPRING 26' SENIOR PROJECT

Sprint 0 repository scaffold for a two-person capstone project on epigenetic-context-aware GNNs for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. This repository does not commit large datasets, checkpoints, or run artifacts.

## Setup

Install and sync dependencies with `uv`:

```bash
uv sync
```

This project uses `pyproject.toml` and `uv.lock` as the dependency source of truth. Do not manually create or maintain `requirements.txt`.

## Basic Commands

```bash
uv run pytest -q
uv run python scripts/audit_dataset.py --config configs/data/mak2022.yaml --sample
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml --debug --max-epochs 1
```

## Data

Place the primary raw dataset at:

```text
data/raw/260520_putative_nucleosomal.parquet
```

Raw, interim, and processed datasets are ignored by git. Use `data/sample/` only for tiny test fixtures.

## Current Scope

Sprint 0 contains scaffold, configs, docs, label helpers, and smoke scripts only. Real ML models, PyTorch, and PyTorch Geometric are intentionally deferred.
