# Project Context

This project builds an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large data files live outside git under `data/raw/`, `data/interim/`, or `data/processed/`.

Sprint 1 completed the dataset audit, label scheme validation, and feature parsing policy. Sprint 2 completed fair same-split non-GNN and sequence baselines under the locked guide-level measured-only evaluation protocol. Sprint 3 completed dependency-light Graph A/B/C artifact construction and leakage-control checks under that same frozen contract, without training a graph model. Sprint 4 has validated real Colab GPU Graph A and Graph C GCN runs from the typed Sprint 3 artifacts. The strongest current non-graph baseline is still `xgboost_unweighted / F4`; the validated Graph A and Graph C GCN runs are same-contract graph baselines, but neither beats the F4 XGBoost reference on primary test AUPRC.

The near-term path is Sprint 4 closure or the optional Graph B bounded control. Graph C must be interpreted as changing both topology and target semantics/context representation relative to Graph A, not as a topology-only comparison.

Core rules:

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Keep Colab as a command runner.
- Put reusable logic in `src/crispr_gnn/`.
- Keep notebooks exploratory or runner-only.
- Use guide-level evaluation for final results.
- Treat random splits as debug-only.
- Keep PyTorch Geometric limited to the approved Sprint 4 graph-model implementation; PyTorch is also used for non-graph sequence baselines.
