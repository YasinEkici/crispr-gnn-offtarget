# Project Context

This project builds an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large data files live outside git under `data/raw/`, `data/interim/`, or `data/processed/`.

Sprint 1 completed the dataset audit, label scheme validation, and feature parsing policy. Sprint 2 completed fair same-split non-GNN and sequence baselines under the locked guide-level measured-only evaluation protocol. Sprint 3 completed dependency-light Graph A/B/C artifact construction and leakage-control checks under that same frozen contract, without training a graph model. Sprint 4 is complete: all three GCN schemas (Graph A, Graph B as bounded secondary control, Graph C) have validated real Colab GPU runs under the frozen Sprint 2/Sprint 3 contract. The strongest current non-graph baseline is still `xgboost_unweighted / F4`; none of the GCN schemas beats the F4 XGBoost reference on primary test AUPRC.

Sprint 4 final results (test AUPRC / test MCC, positive prevalence 0.9007):

- Graph A: AUPRC 0.9663 / MCC 0.3008 — minimal physical-target GCN baseline
- Graph B: AUPRC 0.9666 / MCC 0.1266 — bounded topology-only control (guide-similarity edges added; MCC is threshold-sensitive and should be interpreted with caution)
- Graph C: AUPRC 0.9616 / MCC 0.4538 — context-enriched comparison (changes both topology and target semantics; not topology-only)
- F4 XGBoost: AUPRC 0.9925 / MCC 0.3452

The near-term path is Sprint 5 systematic epigenetic feature ablation, which is the project's main scientific novelty experiment. Graph C must continue to be interpreted as changing both topology and target semantics/context representation relative to Graph A, not as a topology-only comparison.

Core rules:

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Keep Colab as a command runner.
- Put reusable logic in `src/crispr_gnn/`.
- Keep notebooks exploratory or runner-only.
- Use guide-level evaluation for final results.
- Treat random splits as debug-only.
- Keep PyTorch Geometric limited to the approved Sprint 4 graph-model implementation; PyTorch is also used for non-graph sequence baselines.
