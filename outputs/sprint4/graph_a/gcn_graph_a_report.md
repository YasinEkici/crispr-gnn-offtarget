# Sprint 4 GCN Report

Run label: `sprint4_graph_a_gcn_seed42_20260601`

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- Thresholds are selected from validation only.
- Test diagnostics are interpretation-only and cannot drive model or schema decisions.

## Baseline Reference

- Required comparison: `xgboost_unweighted / F4`.
- Baseline test AUPRC: `0.992522`.
- Test positive prevalence for GCN result: `0.900705`.
- Graph A test AUPRC is below the F4 XGBoost reference, so Graph A is a
  validated same-contract GCN baseline but not a stronger predictive baseline.

## Result Summary

| model_name | graph_schema | feature_set | test_auprc | test_auroc | test_f1 | test_mcc |
| --- | --- | --- | --- | --- | --- | --- |
| gcn_graph_a | graph_a_minimal_physical_target | S1_pair+F1 | 0.966287 | 0.745076 | 0.951845 | 0.300781 |

## Artifact Index

Diagnostic tables:
- `outputs/sprint4/graph_a/diagnostics/gcn_graph_a_score_direction.csv`
- `outputs/sprint4/graph_a/diagnostics/gcn_graph_a_fixed_threshold_metrics.csv`
- `outputs/sprint4/graph_a/diagnostics/gcn_graph_a_score_deciles.csv`
- `outputs/sprint4/graph_a/diagnostics/gcn_graph_a_per_genome_metrics.csv`
- `outputs/sprint4/graph_a/diagnostics/gcn_graph_a_test_per_guide_metrics.csv`

Figures:
- `outputs/sprint4/graph_a/figures/gcn_graph_a_graph_schema_auprc_comparison.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_pr_curves.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_roc_curves.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_training_curves.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_score_distributions.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_confusion_matrices.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_decile_lift.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_per_genome_metrics.png`
- `outputs/sprint4/graph_a/figures/gcn_graph_a_view_sanity_example.png`

## Interpretation Boundaries

- Graph-view visualizations are bounded sanity checks, not performance claims.
- Smoke or mocked outputs are not final Sprint 4 performance evidence.
- Graph C, when added later, must not be described as topology-only.
