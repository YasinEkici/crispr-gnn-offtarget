# Sprint 5 Graph A Feature Ablation Report

Run label: `sprint5_graph_a_feature_ablation_seed42_20260603_104923`

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

## Feature Ladder

Sprint 5 keeps `graph_a_minimal_physical_target` fixed and changes only the
candidate-pair edge feature table. The ablation is cumulative:

| feature_set | contents | columns |
| --- | --- | ---: |
| `S5F0_seq` | Guide and target sequence one-hot channels only; no explicit mismatch channel. | 230 |
| `S5F1_mismatch` | `S5F0_seq` plus engineered sequence summary and mismatch-position features. | 263 |
| `S5F2_energy` | `S5F1_mismatch` plus binding-energy scalar features: `energy_1`-`energy_5`. | 268 |
| `S5F3_experimental_epi` | `S5F2_energy` plus experimental epigenetic scalar features: `epigen_ctcf`, `epigen_dnase`, `epigen_rrbs`, `epigen_h3k4me3`, `epigen_drip`, `MNase`. | 274 |
| `S5F4_computed_agg` | `S5F3_experimental_epi` plus aggregated computed nucleosome features and explicit missingness indicators. | 365 |
| `S5F5_computed_pos` | `S5F4_computed_agg` plus position-resolved computed nucleosome vectors. | 664 |

Binding-energy features are not epigenetic features. In this report, epigenetic
information starts at `S5F3_experimental_epi`; computed nucleosome/context
features start at `S5F4_computed_agg`.

## Result Summary

| model_name | graph_schema | feature_set | target_node_representation | target_semantics | test_auprc | test_auroc | test_f1 | test_macro_f1 | test_mcc | test_specificity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F0_seq | zero_type_feature | minimal_physical_target | 0.960432 | 0.723843 | 0.949444 | 0.667772 | 0.360493 | 0.295858 |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F1_mismatch | zero_type_feature | minimal_physical_target | 0.968203 | 0.745651 | 0.947759 | 0.473879 | 0.000000 | 0.000000 |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F2_energy | zero_type_feature | minimal_physical_target | 0.976585 | 0.817765 | 0.960075 | 0.695284 | 0.477933 | 0.284024 |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F3_experimental_epi | zero_type_feature | minimal_physical_target | 0.967214 | 0.761677 | 0.952083 | 0.613482 | 0.314439 | 0.171598 |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F4_computed_agg | zero_type_feature | minimal_physical_target | 0.910866 | 0.544960 | 0.947759 | 0.473879 | 0.000000 | 0.000000 |
| gcn_graph_a_sprint5 | graph_a_minimal_physical_target | S5F5_computed_pos | zero_type_feature | minimal_physical_target | 0.909031 | 0.506268 | 0.947759 | 0.473879 | 0.000000 | 0.000000 |

## Confusion Summary

All thresholds are selected on validation data only.

| feature_set | threshold | test_tn | test_fp | test_fn | test_tp | test_specificity | test_sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S5F0_seq` | 0.151176 | 50 | 119 | 40 | 1493 | 0.295858 | 0.973907 |
| `S5F1_mismatch` | 0.095585 | 0 | 169 | 0 | 1533 | 0.000000 | 1.000000 |
| `S5F2_energy` | 0.157007 | 48 | 121 | 6 | 1527 | 0.284024 | 0.996086 |
| `S5F3_experimental_epi` | 0.139497 | 29 | 140 | 13 | 1520 | 0.171598 | 0.991520 |
| `S5F4_computed_agg` | 0.998882 | 0 | 169 | 0 | 1533 | 0.000000 | 1.000000 |
| `S5F5_computed_pos` | 0.745966 | 0 | 169 | 0 | 1533 | 0.000000 | 1.000000 |

## Cross-Sprint Context

| model | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_tn | test_fp | test_fn | test_tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost_unweighted / F4` | 0.992522 | 0.938416 | 0.642737 | 0.345198 | 38 | 131 | 21 | 1512 |
| `sprint4_graph_a_gcn` | 0.966287 | 0.745076 | 0.602136 | 0.300781 | 26 | 143 | 11 | 1522 |
| `sprint4_graph_b_gcn` | 0.966570 | 0.743586 | 0.491761 | 0.126559 | 3 | 166 | 0 | 1533 |
| `sprint4_graph_c_gcn` | 0.961586 | 0.759886 | 0.677604 | 0.453738 | 43 | 126 | 5 | 1528 |
| `sprint5_graph_a / S5F2_energy` | 0.976585 | 0.817765 | 0.695284 | 0.477933 | 48 | 121 | 6 | 1527 |
| `sprint5b_graph_c / GraphCContext+S5F2_energy` | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 14 | 155 | 0 | 1533 |

## Headline Findings

- `S5F2_energy` is the strongest Sprint 5 setting: best test AUPRC, AUROC,
  macro F1, MCC, and a strong confusion profile among the Sprint 5 feature sets.
- The binding-energy step is the main positive ablation result. Adding
  `energy_1`-`energy_5` to sequence/mismatch features improves `S5F1_mismatch`
  from MCC `0.000000` to `0.477933`, and increases test AUPRC from `0.968203`
  to `0.976585`.
- Experimental epigenetic scalar features do not improve the Graph A GCN in this
  run. `S5F3_experimental_epi` drops below `S5F2_energy` on AUPRC, macro F1,
  MCC, and true negatives.
- Computed nucleosome aggregate and position-resolved features are not supported
  by this run as useful Graph A edge features. `S5F4_computed_agg` and
  `S5F5_computed_pos` collapse to zero true negatives at the selected threshold.
- XGBoost `F4` remains the strongest AUPRC baseline, but Sprint 5
  `S5F2_energy` gives the best MCC and macro F1 among the compared GCN results
  and exceeds the XGBoost `F4` MCC under the validation-selected threshold
  policy.
- Sprint 5B shows that adding the best Sprint 5 energy feature table to Graph C
  improves Graph C AUPRC relative to Sprint 4 (`0.972481` vs. `0.961586`) but
  does not outperform Graph A `S5F2_energy`. Its lower MCC and macro F1 come
  from weak negative-class recognition at the validation-selected threshold
  (TN/FP/FN/TP `14/155/0/1533`), which motivates Sprint 6 imbalance and
  threshold/loss analysis.

## Artifact Index

Diagnostic tables:
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/gcn_sprint5_graph_a_fixed_threshold_metrics.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/gcn_sprint5_graph_a_per_genome_metrics.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/gcn_sprint5_graph_a_score_deciles.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/gcn_sprint5_graph_a_score_direction.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/gcn_sprint5_graph_a_test_per_guide_metrics.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/sprint5_graph_a_predictions.csv`
- `outputs/sprint5/graph_a_feature_ablation/diagnostics_sprint5_graph_a/sprint5_graph_a_training_history.csv`

Figures:
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_confusion_matrices.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_decile_lift.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_graph_schema_auprc_comparison.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_per_genome_metrics.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_pr_curves.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_roc_curves.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_score_distributions.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_training_curves.png`
- `outputs/sprint5/graph_a_feature_ablation/figures_sprint5_graph_a/gcn_sprint5_graph_a_view_sanity_example.png`

## Interpretation Boundaries

- Graph-view visualizations are bounded sanity checks, not performance claims.
- Smoke or mocked outputs are not final Sprint 5 performance evidence.
- Graph C must not be described as topology-only; it changes both topology and target semantics/context representation.
