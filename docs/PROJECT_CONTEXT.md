# Project Context

This project builds an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large data files live outside git under `data/raw/`, `data/interim/`, or `data/processed/`.

Sprint 1 completed the dataset audit, label scheme validation, and feature parsing policy. Sprint 2 completed fair same-split non-GNN and sequence baselines under the locked guide-level measured-only evaluation protocol. Sprint 3 completed dependency-light Graph A/B/C artifact construction and leakage-control checks under that same frozen contract, without training a graph model. Sprint 4 is complete: all three GCN schemas (Graph A, Graph B as bounded secondary control, Graph C) have validated real Colab GPU runs under the frozen Sprint 2/Sprint 3 contract. Sprint 5 is complete as a Graph A fixed-topology feature-family ablation, with Sprint 5B as one secondary Graph C energy-sensitivity run. The strongest current non-graph baseline is still `xgboost_unweighted / F4`; no GCN result beats the F4 XGBoost reference on primary test AUPRC.

Sprint 4 final results (test AUPRC / test MCC, positive prevalence 0.9007):

- Graph A: AUPRC 0.9663 / MCC 0.3008 — minimal physical-target GCN baseline
- Graph B: AUPRC 0.9666 / MCC 0.1266 — bounded topology-only control (guide-similarity edges added; MCC is threshold-sensitive and should be interpreted with caution)
- Graph C: AUPRC 0.9616 / MCC 0.4538 — context-enriched comparison (changes both topology and target semantics; not topology-only)
- F4 XGBoost: AUPRC 0.9925 / MCC 0.3452

Sprint 5/5B interpretation:

- Graph A `S5F2_energy`: AUPRC 0.9766 / MCC 0.4779 — strongest GCN result so far; binding-energy features are the main positive feature-ablation signal.
- Graph A `S5F3_experimental_epi`: AUPRC 0.9672 / MCC 0.3144 — raw experimental epigenetic scalars do not improve over `S5F2_energy`.
- Graph A `S5F4_computed_agg` and `S5F5_computed_pos`: AUPRC about 0.91 / MCC 0.0 — computed context feature additions collapse under the current Graph A GCN edge-feature formulation.
- Graph C Sprint 5B `GraphCContext+S5F2_energy`: AUPRC 0.9725 / MCC 0.2743 — improves Sprint 4 Graph C AUPRC, but not Graph A `S5F2_energy`; threshold-selected classification recognizes very few negatives (TN/FP/FN/TP = 14/155/0/1533).

Graph C must continue to be interpreted as changing both topology and target semantics/context representation relative to Graph A, not as a topology-only comparison. The near-term path is Sprint 6 imbalance/threshold/loss analysis, because several high-AUPRC GCN runs show weak negative-class recognition under validation-selected thresholds.

Core rules:

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Keep Colab as a command runner.
- Put reusable logic in `src/crispr_gnn/`.
- Keep notebooks exploratory or runner-only.
- Use guide-level evaluation for final results.
- Treat random splits as debug-only.
- Keep PyTorch Geometric limited to the approved Sprint 4 graph-model implementation; PyTorch is also used for non-graph sequence baselines.
