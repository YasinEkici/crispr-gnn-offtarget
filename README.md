# Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction - BTU CENG SPRING 26' SENIOR PROJECT

Repository for a two-person capstone project on epigenetic-context-aware GNNs for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. This repository does not commit large datasets, checkpoints, or run artifacts.

Current status:

- Sprint 1 dataset audit, label policy, and feature parsing policy are documented.
- Sprint 2 fair non-graph baselines are complete under the locked guide-level measured-only split.
- Sprint 3 dependency-light Graph A/B/C datasets and leakage controls are complete under the same locked split; graph-model training is not yet implemented.
- The strongest current non-graph baseline is `xgboost_unweighted / F4`; future GNN work should compare against it under the same split, label, and metric policy.

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
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml
uv run python scripts/build_features.py --config configs/data/mak2022.yaml
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
uv run python scripts/build_graph.py --config configs/data/mak2022.yaml --schema-config configs/sweeps/graph_schema_ablation.yaml
```

## Data

Place the primary raw dataset at:

```text
data/raw/260520_putative_nucleosomal.parquet
```

Raw, interim, and processed datasets are ignored by git. Use `data/sample/` only for tiny test fixtures.

## Key Sprint 2 Artifacts

- Results table: `outputs/results/baseline_results.csv`
- Baseline report: `outputs/reports/baseline_report.md`
- Split manifest: `outputs/splits/sprint2_guides.json`
- Split summary: `outputs/splits/sprint2_split_summary.csv`
- Feature catalog: `outputs/features/sprint2_feature_catalog.md`

## Key Sprint 3 Artifacts

- Graph schema report: `outputs/reports/graph_schema_report.md`
- Generated typed graph tables and manifests: `data/processed/graphs/sprint3/`
- Graph schema configuration: `configs/sweeps/graph_schema_ablation.yaml`

## Current Scope

The next major scope is Sprint 4 graph-model training from the validated typed graph artifacts. PyTorch is present for Sprint 2 non-graph sequence baselines. PyTorch Geometric remains deferred until graph-model implementation.
