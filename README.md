# Context-Aware GNN for CRISPR-Cas9 Off-Target Prediction - BTU CENG SPRING 26' SENIOR PROJECT

Repository for a two-person capstone project on epigenetic-context-aware GNNs for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. This repository does not commit large datasets, checkpoints, or run artifacts.

Current status:

- Sprint 1 dataset audit, label policy, and feature parsing policy are documented.
- Sprint 2 fair non-graph baselines are complete under the locked guide-level measured-only split.
- Sprint 3 dependency-light Graph A/B/C datasets and leakage controls are complete under the same locked split.
- Sprint 4 GCN baseline training is complete: Graph A, Graph B (bounded topology-control), and Graph C all have validated Colab GPU runs under the frozen contract. None beats `xgboost_unweighted / F4` on primary test AUPRC. Consolidated comparison: `outputs/sprint4/gcn_sprint4_comparison_results.csv`.

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

- Results table: `outputs/sprint2/baseline_results.csv`
- Baseline report: `outputs/sprint2/baseline_report.md`
- Split manifest: `outputs/splits/sprint2_guides.json`
- Split summary: `outputs/splits/sprint2_split_summary.csv`
- Feature catalog: `outputs/features/sprint2_feature_catalog.md`

## Key Sprint 3 Artifacts

- Graph schema report: `outputs/sprint3/graph_schema_report.md`
- Generated typed graph tables and manifests: `data/processed/graphs/sprint3/`
- Graph schema configuration: `configs/sweeps/graph_schema_ablation.yaml`

## Current Scope

The current Sprint 4 scope has validated Graph A and Graph C as same-contract
GCN baselines from typed Sprint 3 artifacts. Graph C changes both topology and
target semantics/context representation relative to Graph A, so it is not a
topology-only comparison. Graph B remains an optional bounded control.
PyTorch Geometric is part of the Sprint 4 graph-model implementation.
