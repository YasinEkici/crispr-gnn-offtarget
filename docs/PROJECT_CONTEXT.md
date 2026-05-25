# Project Context

This project builds an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large data files live outside git under `data/raw/`, `data/interim/`, or `data/processed/`.

Sprint 1 completed the dataset audit, label scheme validation, and feature parsing policy. Sprint 2 completed fair same-split non-GNN and sequence baselines under the locked guide-level measured-only evaluation protocol. Sprint 3 completed dependency-light Graph A/B/C artifact construction and leakage-control checks under that same frozen contract, without training a graph model. The strongest current non-graph baseline is `xgboost_unweighted / F4`; future graph models should compare against it under the same split, label, feature, and metric policy.

The near-term path is Sprint 4 graph-model materialization and evaluation from the validated typed artifacts, beginning with Graph A and Graph C while retaining Graph B as a bounded control.

Core rules:

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Keep Colab as a command runner.
- Put reusable logic in `src/crispr_gnn/`.
- Keep notebooks exploratory or runner-only.
- Use guide-level evaluation for final results.
- Treat random splits as debug-only.
- Keep PyTorch Geometric deferred until graph-model implementation; PyTorch is already allowed for non-graph sequence baselines.
