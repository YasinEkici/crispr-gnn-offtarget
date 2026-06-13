# Sprint 5B Graph C Energy Sensitivity Report

Run label: `sprint5b_graph_c_context_observation_gcn_graph_c_sprint5b_energy_seed42_20260603T124750`

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

| model_name | graph_schema | feature_set | target_node_representation | target_semantics | test_auprc | test_auroc | test_f1 | test_macro_f1 | test_mcc | test_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_graph_c_sprint5b_energy | graph_c_context_observation | GraphCContext+S5F2_energy | target_observation_context_encoder | observation_level_context_target | 0.972481 | 0.836219 | 0.951878 | 0.552442 | 0.274287 | 0.082840 |

## Comparison Context

| reference | setting | test_auprc | test_macro_f1 | test_mcc | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | --- |
| Sprint 4 Graph C | `CandidatePair+TargetObservationContext` | 0.961586 | n/a | 0.453738 | 43/126/5/1528 |
| Sprint 5 Graph A | `S5F2_energy` | 0.976585 | 0.695284 | 0.477933 | 48/121/6/1527 |
| Sprint 5B Graph C | `GraphCContext+S5F2_energy` | 0.972481 | 0.552442 | 0.274287 | 14/155/0/1533 |
| Sprint 2 XGBoost | `F4` | 0.992522 | n/a | 0.345198 | 38/131/21/1512 |

Sprint 5B improves Graph C's threshold-free AUPRC relative to Sprint 4 Graph C,
which supports the Sprint 5 finding that binding-energy features carry useful
signal. It does not outperform the fixed-topology Graph A `S5F2_energy` run.

## MCC and Macro F1 Interpretation

The lower MCC and macro F1 are driven by the validation-selected threshold,
not by AUPRC ranking alone. At the selected threshold, the test confusion
matrix is:

```text
TN = 14
FP = 155
FN = 0
TP = 1533
```

The model catches every positive test row but recognizes only 14 of 169
negative rows. Because the locked test set has positive prevalence `0.900705`,
binary F1 remains high while negative-class performance is weak. MCC and macro
F1 expose this imbalance because they penalize poor negative-class recognition
more strongly than positive-class F1.

This means Sprint 5B is a useful ranker sensitivity but not a better operating
point classifier. The result strengthens the case for Sprint 6 imbalance,
threshold, and loss analysis rather than opening more Sprint 5 feature tuning.

## Literature Context

Mak et al. 2022 reports that CRISPRspec binding-energy scores have SHAP
importance comparable to high-performing nucleosome-organization scores, while
the six experimental epigenetic scalar features show weak direct correlation
with off-target activity. Sprint 5 and Sprint 5B are consistent with the
binding-energy part of that literature: `S5F2_energy` is the strongest GCN
feature signal observed so far.

The broader chromatin biology literature supports that nucleosomes and
chromatin accessibility can affect Cas9 binding and cleavage. This Sprint 5B
run does not reject that biology. It shows a narrower modeling result: under
the current GCN architecture, locked guide-level split, validation-selected
threshold policy, and Graph C context-observation representation, adding the
best energy edge signal improves Graph C AUPRC but does not make Graph C
outperform Graph A `S5F2_energy` or XGBoost `F4`.

## Artifact Index

Diagnostic tables:
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_score_direction.csv`
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_fixed_threshold_metrics.csv`
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_score_deciles.csv`
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_per_genome_metrics.csv`
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_test_per_guide_metrics.csv`

Figures:
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_graph_schema_auprc_comparison.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_pr_curves.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_roc_curves.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_training_curves.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_score_distributions.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_confusion_matrices.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_decile_lift.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_per_genome_metrics.png`
- `outputs/sprint5b/graph_c/figures/gcn_graph_c_view_sanity_example.png`

## Interpretation Boundaries

- Graph-view visualizations are bounded sanity checks, not performance claims.
- Smoke or mocked outputs are not final Sprint 5B performance evidence.
- Graph C must not be described as topology-only; it changes both topology and target semantics/context representation.
- The test-set confusion profile is interpretation-only and must not be used
  to retune thresholds, hyperparameters, features, or topology inside Sprint 5B.
