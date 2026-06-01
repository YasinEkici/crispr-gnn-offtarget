# Sprint 4 GCN Report

Run label: `sprint4_graph_c_gcn_seed42_20260601`

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

## Result Summary

| model_name | graph_schema | feature_set | target_node_representation | target_semantics | test_auprc | test_auroc | test_f1 | test_mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_graph_c | graph_c_context_observation | CandidatePair+TargetObservationContext | target_observation_context_encoder | observation_level_context_target | 0.961586 | 0.759886 | 0.958896 | 0.453738 |

## Artifact Index

Diagnostic tables:
- `outputs/sprint4/graph_c/diagnostics/gcn_graph_c_score_direction.csv`
- `outputs/sprint4/graph_c/diagnostics/gcn_graph_c_fixed_threshold_metrics.csv`
- `outputs/sprint4/graph_c/diagnostics/gcn_graph_c_score_deciles.csv`
- `outputs/sprint4/graph_c/diagnostics/gcn_graph_c_per_genome_metrics.csv`
- `outputs/sprint4/graph_c/diagnostics/gcn_graph_c_test_per_guide_metrics.csv`

Figures:
- `outputs/sprint4/graph_c/figures/gcn_graph_c_graph_schema_auprc_comparison.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_pr_curves.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_roc_curves.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_training_curves.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_score_distributions.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_confusion_matrices.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_decile_lift.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_per_genome_metrics.png`
- `outputs/sprint4/graph_c/figures/gcn_graph_c_view_sanity_example.png`

## Interpretation Boundaries

- Graph-view visualizations are bounded sanity checks, not performance claims.
- Smoke or mocked outputs are not final Sprint 4 performance evidence.
- Graph C must not be described as topology-only; it changes both topology and target semantics/context representation.
