# Sprint 6 Loss Comparison Report

Run batch: `sprint6_loss_comparison_seed42_20260606_182812`

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Graph schema: `graph_a_minimal_physical_target`.
- Feature set: `S5F2_energy`.
- Training regime: measured-only headline; no `measured=0` putative rows.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Primary metric: AUPRC. Threshold metrics are secondary interpretation outputs.
- Test positive prevalence: `0.900705`.
- Required reference: `xgboost_unweighted / F4` test AUPRC `0.992522`.
- Sprint 5 Graph A `S5F2_energy` reference test AUPRC `0.976585`.

## Result Summary

| predeclared_run_id | loss | loss_params | sampling | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp | delta_auprc_vs_S6R0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S6R0_wbce | weighted_bce | {"pos_weight": "auto"} | null | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 | 0.000000 |
| S6R1_bce_unw | bce_unweighted | {"pos_weight": 1.0} | null | 0.949280 | 0.685090 | 0.550154 | 0.234557 | 0.082840 | 14 | 155 | 4 | 1529 | -0.027655 |
| S6R2_focal_g2_a25 | focal | {"alpha": 0.25, "gamma": 2.0} | null | 0.963497 | 0.805436 | 0.525318 | 0.205834 | 0.053254 | 9 | 160 | 1 | 1532 | -0.013439 |
| S6R3_focal_g1_a25 | focal | {"alpha": 0.25, "gamma": 1.0} | null | 0.962372 | 0.795308 | 0.508688 | 0.162851 | 0.035503 | 6 | 163 | 1 | 1532 | -0.014563 |
| S6R4_focal_g2_a50 | focal | {"alpha": 0.5, "gamma": 2.0} | null | 0.956803 | 0.773731 | 0.508688 | 0.162851 | 0.035503 | 6 | 163 | 1 | 1532 | -0.020132 |
| S6R5_dice | generalized_dice | {"class_weights": "inverse_volume", "epsilon": 1.0} | null | 0.871174 | 0.382662 | 0.473879 | 0.000000 | 0.000000 | 0 | 169 | 0 | 1533 | -0.105762 |
| S6R6_tversky_a70_b30 | tversky | {"alpha": 0.7, "beta": 0.3, "epsilon": 1.0} | null | 0.955804 | 0.710071 | 0.551863 | 0.262978 | 0.082840 | 14 | 155 | 1 | 1532 | -0.021132 |
| S6R7_balanced_sampling | bce_unweighted | {"pos_weight": 1.0} | {"deterministic_by_seed": true, "scope": "measured_only", "strategy": "balanced_subsample", "target_ratio": 1.0} | 0.976205 | 0.815167 | 0.673742 | 0.447602 | 0.248521 | 42 | 127 | 5 | 1528 | -0.000731 |

## Interpretation Boundaries

Sprint 6 varies only the loss function or measured-only training-time sampling.
AUPRC remains the primary comparison. Specificity, TNR, MCC, and macro F1 are
reported to diagnose threshold behavior and negative-class recognition, but
improvements in those secondary metrics must not be described as AUPRC gains.

If threshold collapse persists across losses, the interpretation must include
the architecture caveat: in the current `GraphAEdgeGCN`, candidate-edge features
such as `S5F2_energy` are concatenated at the edge classifier and do not enter
GCN message passing. Collapse therefore cannot be attributed to the loss alone.

## Final Interpretation

Under the locked Scheme A / guide-level / measured-only / `experiment_id=18`-excluded
protocol and the fixed Graph A + `S5F2_energy` setting, **no predeclared objective
beat the weighted-BCE baseline on the primary metric.** Best test AUPRC is
`S6R0_wbce` at `0.976935`; every alternative is lower (Δ vs S6R0: balanced
sampling `-0.000731`; focal `-0.013` to `-0.020`; Tversky `-0.021`; unweighted
BCE `-0.028`; generalized Dice `-0.106`). Weighted BCE is **also** best on
negative-class recognition (specificity `0.290`, TNR 49/169, MCC `0.484`); the
cost-sensitive losses are worse on both axes (focal TN 6–9; Tversky TN 14), and
generalized Dice collapses below the positive-prevalence AUPRC floor
(`0.871` < `0.900705`) with TN=0.

**No objective improved negative-class recognition without sacrificing AUPRC.**
Because candidate-edge features (`S5F2_energy`) enter only the edge-classifier
head and not GCN message passing in the current `GraphAEdgeGCN`, the residual
threshold collapse is attributed to architecture / feature-distribution limits,
not to the loss alone — motivating the Sprint 7 edge-aware (GAT/GATv2)
investigation. No GCN objective beats `xgboost_unweighted / F4` (`0.992522`),
which remains the authoritative bar. This is **not** a reproduction of Gao 2020,
Guan 2024, or Mak 2022, and used **no test-driven selection** (checkpoint and
threshold selected on validation only).

Robustness: guide-level (cluster) bootstrap CIs at
`diagnostics_sprint6/imbalance_bootstrap_cis.csv` are wide and overlapping (e.g.
`S6R0` AUPRC `0.976935`, 95% CI `[0.904, 0.9995]`), so the headline AUPRC deltas
above are within noise — only generalized Dice is clearly separable downward.
Full uncertainty quantification (BCa, paired-difference, multi-seed) is the
optional Sprint 8 robustness layer.

## Artifact Index

Diagnostic tables:
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_score_direction.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_threshold_metrics.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_score_deciles.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_per_guide_metrics.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_per_guide_metric_distribution.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_positive_retrieval_summary.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_negative_retrieval_summary.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_per_genome_metrics.csv`

Figures:
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_auprc_comparison.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_pr_curves.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_threshold_metrics.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_score_distributions.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_per_guide_metric_distribution.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_positive_retrieval_summary.png`
- `outputs/sprint6/loss_comparison/figures_sprint6/imbalance_negative_retrieval_summary.png`
