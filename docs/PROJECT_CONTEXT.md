# Project Context

This project builds an epigenetic-context-aware GNN framework for CRISPR-Cas9 off-target prediction.

The first working dataset is the Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset. Large data files live outside git under `data/raw/`, `data/interim/`, or `data/processed/`.

Sprint 1 completed the dataset audit, label scheme validation, and feature parsing policy. Sprint 2 completed fair same-split non-GNN and sequence baselines under the locked guide-level measured-only evaluation protocol. Sprint 3 completed dependency-light Graph A/B/C artifact construction and leakage-control checks under that same frozen contract, without training a graph model. Sprint 4 is complete: all three GCN schemas (Graph A, Graph B as bounded secondary control, Graph C) have validated real Colab GPU runs under the frozen Sprint 2/Sprint 3 contract. Sprint 5 is complete as a Graph A fixed-topology feature-family ablation, with Sprint 5B as one secondary Graph C energy-sensitivity run. Sprint 6 is complete (Slices 0–4; Slice 5 optional/deferred): a predeclared loss/sampling comparison on the fixed Graph A + `S5F2_energy` setting, where no objective beat weighted BCE. Sprint 7 is complete: Graph C GATv2 target-context modelling produced the strongest same-contract GNN results so far. Sprint 8 is complete: Sprint 8A selected `S8A_R2_context_edge_film` by validation AUPRC, and Sprint 8B selected `S8B_R2_sequence_plus_context` by validation AUPRC, but no Sprint 8 variant beat the XGBoost F4 primary-AUPRC bar. The strongest current non-graph baseline remains `xgboost_unweighted / F4`. Sprint 9 is complete (robustness/uncertainty, interpretation-only): guide-cluster bootstrap CIs, paired-difference, and multi-seed retraining show **no robust AUPRC improvement** by any Sprint 8 candidate over its lineage or over F4 (all 8 predeclared paired AUPRC differences include zero; per-config seed-std exceeds the candidate gains), and F4 keeps the highest mean AUPRC with the smallest spread. A threshold-dependent operating-point effect (GNNs recover more rare negatives at their thresholds) exists but is seed-fragile and does not override AUPRC. See `outputs/sprint9/robustness_report.md`.

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

Graph C must continue to be interpreted as changing both topology and target semantics/context representation relative to Graph A, not as a topology-only comparison.

Sprint 6 result (loss/sampling comparison on fixed Graph A + `S5F2_energy`): no loss or measured-only sampling beats weighted BCE on primary AUPRC (best `S6R0_wbce` 0.9769), and weighted BCE also gives the best negative-class recognition (specificity/TNR/MCC); focal/Dice/Tversky underperform and generalized Dice collapses below the prevalence floor (AUPRC 0.871, TN=0). Because candidate-edge features do not enter GCN message passing in the current architecture, residual threshold collapse is not attributed to loss alone.

Sprint 7/8 result: Graph C GATv2 target-context modelling remains the strongest GNN direction under the frozen contract. Sprint 7F R3 remains the strongest carry-forward GNN AUPRC reference (test AUPRC 0.9849), Sprint 8A selected R2 head-only FiLM (`S8A_R2_context_edge_film`) by validation AUPRC (test AUPRC 0.9828, MCC 0.5637), and Sprint 8B selected `S8B_R2_sequence_plus_context` by validation AUPRC (test AUPRC 0.9860, MCC 0.5673). Sprint 8B added a small single-seed AUPRC gain over the Sprint 8A R2 carry-forward (`+0.003263`) but still did not beat the XGBoost F4 AUPRC bar; the sequence-only S1 Conv+BiLSTM path collapsed under the frozen measured-only contract. Sprint 8 candidates are validation-selected, not superiority claims. Robustness and superiority claims were resolved in Sprint 9 (multi-seed, guide-level bootstrap CIs, paired-difference): on the primary metric AUPRC none of the eight predeclared paired differences (P1–P8) excludes zero — the Sprint 8 gains are inside single-seed / guide-composition / training-stochasticity uncertainty (≈±0.05 guide-cluster intervals; seed-std 0.004–0.012 > the ~0.003 candidate gains), so the candidates are reported as **compatible with no difference** (not "equivalent" — TOST deferred). At their frozen thresholds the GNNs do robustly recover more rare negatives than F4 (P4–P6 specificity, P5/P6 MCC and macro-F1 exclude zero), but this operating-point effect is threshold-dependent, guide-`9251`-fragile, and seed-fragile, and per the §11 claim boundaries does not override the AUPRC ranking. Sprint 9 uses percentile-primary, BCa-sensitivity-only intervals.

Core rules:

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Keep Colab as a command runner.
- Put reusable logic in `src/crispr_gnn/`.
- Keep notebooks exploratory or runner-only.
- Use guide-level evaluation for final results.
- Treat random splits as debug-only.
- Keep PyTorch Geometric limited to the approved Sprint 4 graph-model implementation; PyTorch is also used for non-graph sequence baselines.
