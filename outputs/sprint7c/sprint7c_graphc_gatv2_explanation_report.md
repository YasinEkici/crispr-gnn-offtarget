# Sprint 7C Graph C GATv2 Explanation Report

Generated at UTC: `2026-06-07T19:41:06.474741+00:00`

## Executive Summary

Sprint 7C is an analysis-only follow-up to Sprint 7B. It did not train a model,
change a threshold, change a graph, or select a new architecture from test
data. Its purpose was to explain the Sprint 7B Graph C GATv2 result at the
row/error-transition level before any mechanism ablation or model-improvement
work.

The identity gate passed for the relevant fixed validation/test universes:
Scheme A, `sprint2_main_seed42`, guide-disjoint, measured-only rows, and the
same test positive prevalence (`0.900705`). Within that aligned universe, Graph
C GATv2 changed the Graph C GCN operating point in a clear way: it recovered 53
Graph C GCN false positives as true negatives, lost 4 previous true negatives,
and introduced 12 new false negatives. Net true-negative gain was therefore
`+49`.

The metric recheck matches the Sprint 7B claim. Graph C GATv2 did not improve
over Graph C GCN on AUPRC (`0.969078` versus `0.972481`), but it strongly
improved rare-negative threshold diagnostics: MCC `0.531774` versus `0.274287`,
specificity `0.372781` versus `0.082840`, and TN `63` versus `14`.

The defensible Sprint 7C conclusion is that the Sprint 7B Graph C GATv2 result
is a real row-level rare-negative operating-point shift under the locked
single-seed contract. It is not a primary AUPRC win, not biological causal
evidence, and not robustness evidence.

## Contract

- Analysis-only sprint: no model was trained and no threshold/model/graph
  setting was changed.
- Label/split/evaluation remain Scheme A, `sprint2_main_seed42`,
  guide-disjoint, measured-only, validation-selected checkpoint/threshold.
- AUPRC remains the primary metric; MCC, macro F1, specificity and TN/FP/FN/TP
  are threshold diagnostics.
- Attention summaries are model-interpretation artifacts only, not biological
  causal evidence.
- Graph C is topology plus target-observation semantics/context, not topology
  only.

## Identity Gate

| audit_type | analysis_model_id | comparison_model_id | split | rows | positives | negatives | positive_rate | row_index_unique | genome_missing_rate | passed | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_model_split | graph_a_gcn_s6_weighted_bce |  | val | 1734 | 1511 | 223 | 0.871396 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_a_gcn_s6_weighted_bce |  | test | 1702 | 1533 | 169 | 0.900705 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_a_gatv2_s7 |  | val | 1734 | 1511 | 223 | 0.871396 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_a_gatv2_s7 |  | test | 1702 | 1533 | 169 | 0.900705 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_c_gcn_s5b |  | val | 1734 | 1511 | 223 | 0.871396 | True | 1.000000 | True | genome missing rate 1.000; metadata join required for per-genome claims |
| single_model_split | graph_c_gcn_s5b |  | test | 1702 | 1533 | 169 | 0.900705 | True | 1.000000 | True | genome missing rate 1.000; metadata join required for per-genome claims |
| single_model_split | graph_b_gcn_s7b |  | val | 1734 | 1511 | 223 | 0.871396 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_b_gcn_s7b |  | test | 1702 | 1533 | 169 | 0.900705 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_b_gatv2_s7b |  | val | 1734 | 1511 | 223 | 0.871396 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_b_gatv2_s7b |  | test | 1702 | 1533 | 169 | 0.900705 | True | 0.000000 | True | passed basic split distribution audit |
| single_model_split | graph_c_gatv2_s7b |  | val | 1734 | 1511 | 223 | 0.871396 | True | 1.000000 | True | genome missing rate 1.000; metadata join required for per-genome claims |
| single_model_split | graph_c_gatv2_s7b |  | test | 1702 | 1533 | 169 | 0.900705 | True | 1.000000 | True | genome missing rate 1.000; metadata join required for per-genome claims |

Graph C GCN -> Graph C GATv2 row-level transition allowed:
`True`.

## Metric Recheck

| analysis_model_id | analysis_predeclared_run_id | source_label | split | rows | positives | negatives | positive_rate | auprc | auroc | threshold | threshold_available | f1 | macro_f1 | mcc | specificity | sensitivity | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_c_gcn_s5b | S7C_REF_GRAPH_C_GCN | Sprint 5B Graph C GCN S5F2 reference | test | 1702 | 1533 | 169 | 0.900705 | 0.972481 | 0.836219 | 0.029094 | True | 0.951878 | 0.552442 | 0.274287 | 0.082840 | 1.000000 | 14 | 155 | 0 | 1533 |
| graph_c_gatv2_s7b | S7C_GRAPH_C_GATV2 | Sprint 7B Graph C GATv2 S5F2 result | test | 1702 | 1533 | 169 | 0.900705 | 0.969078 | 0.849705 | 0.093162 | True | 0.962658 | 0.739526 | 0.531774 | 0.372781 | 0.992172 | 63 | 106 | 12 | 1521 |

## Threshold Transfer

| analysis_model_id | analysis_predeclared_run_id | source_label | split | threshold | threshold_available | auprc | auroc | f1 | macro_f1 | mcc | specificity | sensitivity | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_c_gatv2_s7b | S7C_GRAPH_C_GATV2 | Sprint 7B Graph C GATv2 S5F2 result | test | 0.093162 | True | 0.969078 | 0.849705 | 0.962658 | 0.739526 | 0.531774 | 0.372781 | 0.992172 | 63 | 106 | 12 | 1521 |
| graph_c_gatv2_s7b | S7C_GRAPH_C_GATV2 | Sprint 7B Graph C GATv2 S5F2 result | val | 0.093162 | True | 0.981135 | 0.904114 | 0.935915 | 0.764567 | 0.530438 | 0.627803 | 0.927862 | 140 | 83 | 109 | 1402 |
| graph_c_gcn_s5b | S7C_REF_GRAPH_C_GCN | Sprint 5B Graph C GCN S5F2 reference | test | 0.029094 | True | 0.972481 | 0.836219 | 0.951878 | 0.552442 | 0.274287 | 0.082840 | 1.000000 | 14 | 155 | 0 | 1533 |
| graph_c_gcn_s5b | S7C_REF_GRAPH_C_GCN | Sprint 5B Graph C GCN S5F2 reference | val | 0.029094 | True | 0.980476 | 0.886871 | 0.932716 | 0.488288 | 0.139980 | 0.022422 | 1.000000 | 5 | 218 | 0 | 1511 |

## Error Transitions

- Recovered negatives (`FP_to_TN`): `53`
- Lost negatives (`TN_to_FP`): `4`
- Net TN gain: `49`
- New false negatives (`TP_to_FN`): `12`

| baseline_confusion | candidate_confusion | transition | rows |
| --- | --- | --- | --- |
| FP | FP | FP_to_FP | 102 |
| FP | TN | FP_to_TN | 53 |
| TN | FP | TN_to_FP | 4 |
| TN | TN | TN_to_TN | 10 |
| TP | FN | TP_to_FN | 12 |
| TP | TP | TP_to_TP | 1521 |

## Score Distribution And Negative Rank

Score distribution and rank diagnostics are written as tables. Use them to
describe model score movement, not to select a new threshold or model.

Key Graph C score rows:

| analysis_model_id | split | label | rows | mean_score | median_score | std_score | q05_score | q25_score | q75_score | q95_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_c_gcn_s5b | val | 1 | 1511 | 0.529949 | 0.535146 | 0.265362 | 0.094896 | 0.319536 | 0.729952 | 0.996176 |
| graph_c_gcn_s5b | val | 0 | 223 | 0.161801 | 0.115612 | 0.135951 | 0.056890 | 0.080553 | 0.186530 | 0.465679 |
| graph_c_gcn_s5b | test | 1 | 1533 | 0.579559 | 0.607748 | 0.289522 | 0.138742 | 0.310201 | 0.818337 | 0.999435 |
| graph_c_gcn_s5b | test | 0 | 169 | 0.229908 | 0.170719 | 0.226879 | 0.022929 | 0.070465 | 0.300857 | 0.716473 |
| graph_c_gatv2_s7b | val | 1 | 1511 | 0.354976 | 0.287444 | 0.239699 | 0.069533 | 0.214400 | 0.419007 | 0.992351 |
| graph_c_gatv2_s7b | val | 0 | 223 | 0.106685 | 0.083183 | 0.104530 | 0.046091 | 0.059160 | 0.108972 | 0.230810 |
| graph_c_gatv2_s7b | test | 1 | 1533 | 0.482656 | 0.395517 | 0.308507 | 0.146456 | 0.201656 | 0.752761 | 0.999187 |
| graph_c_gatv2_s7b | test | 0 | 169 | 0.175027 | 0.126825 | 0.208750 | 0.028219 | 0.038288 | 0.201941 | 0.519144 |

Negative rank summary:

| analysis_model_id | split | rows | negatives | negative_median_rank | negative_mean_rank | negative_top_decile_count | positive_bottom_decile_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_c_gcn_s5b | val | 1734 | 223 | 1517.000000 | 1452.062780 | 0 | 84 |
| graph_c_gcn_s5b | test | 1702 | 169 | 1498.000000 | 1366.923077 | 6 | 91 |
| graph_c_gatv2_s7b | val | 1734 | 223 | 1529.000000 | 1478.116592 | 2 | 80 |
| graph_c_gatv2_s7b | test | 1702 | 169 | 1590.000000 | 1387.597633 | 7 | 71 |

## Per-Guide Concentration

| grna_target_id | rows | negatives | baseline_tn | candidate_tn | recovered_fp_to_tn | lost_tn_to_fp | new_fn | mean_score_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1028 | 15 | 14 | 0 | 14 | 14 | 0 | 1 | -0.058278 |
| 1120 | 15 | 14 | 1 | 14 | 13 | 0 | 1 | -0.010588 |
| 1060 | 15 | 13 | 0 | 13 | 13 | 0 | 2 | -0.316349 |
| 1044 | 13 | 12 | 1 | 12 | 11 | 0 | 1 | -0.024383 |
| 1408 | 9 | 9 | 8 | 9 | 1 | 0 | 0 | 0.007206 |
| 9251 | 176 | 80 | 0 | 1 | 1 | 0 | 0 | -0.059955 |
| 228 | 12 | 7 | 0 | 0 | 0 | 0 | 0 | -0.363637 |
| 278 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0.117236 |
| 635 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0.049441 |
| 804 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001685 |
| 1018 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | -0.012927 |
| 1026 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | -0.011054 |
| 1088 | 13 | 12 | 4 | 0 | 0 | 4 | 0 | 0.295157 |
| 1505 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0.273695 |
| 3265 | 35 | 0 | 0 | 0 | 0 | 0 | 0 | 0.148602 |
| 3300 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0.237698 |
| 3420 | 320 | 0 | 0 | 0 | 0 | 0 | 0 | -0.122328 |
| 7696 | 126 | 0 | 0 | 0 | 0 | 0 | 0 | -0.223249 |
| 8319 | 101 | 0 | 0 | 0 | 0 | 0 | 0 | -0.182419 |
| 8420 | 408 | 0 | 0 | 0 | 0 | 0 | 0 | -0.271083 |

## Attention Summary

Existing Sprint 7B attention evidence is aggregate by edge kind/layer/head.
It does not support attention-by-confusion-category claims.

| predeclared_run_id | graph_schema | architecture | split | edge_kind | rows | total_edges | mean_attention | mean_attention_std | min_attention | max_attention | interpretation_limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S7B_R2_graph_b_gatv2_s5f2 | graph_b_guide_similarity_control | gatv2 | test | candidate_forward | 8 | 77696 | 0.515917 | 0.163968 | 0.000009 | 0.999855 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R2_graph_b_gatv2_s5f2 | graph_b_guide_similarity_control | gatv2 | test | candidate_reverse | 8 | 77696 | 0.006410 | 0.017964 | 0.000000 | 0.936429 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R2_graph_b_gatv2_s5f2 | graph_b_guide_similarity_control | gatv2 | test | self_loop | 8 | 71440 | 0.425168 | 0.108049 | 0.000005 | 0.999991 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R2_graph_b_gatv2_s5f2 | graph_b_guide_similarity_control | gatv2 | test | sequence_similar_to | 8 | 15648 | 0.030884 | 0.019097 | 0.000005 | 0.177927 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R3_graph_c_gatv2_s5f2 | graph_c_context_observation | gatv2 | test | candidate_forward | 8 | 77696 | 0.637952 | 0.198627 | 0.000000 | 1.000000 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R3_graph_c_gatv2_s5f2 | graph_c_context_observation | gatv2 | test | candidate_reverse | 8 | 77696 | 0.011788 | 0.054519 | 0.000000 | 0.999998 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R3_graph_c_gatv2_s5f2 | graph_c_context_observation | gatv2 | test | context_similar_to | 8 | 1206112 | 0.021734 | 0.018066 | 0.000000 | 0.383099 | aggregate edge-kind attention only; no row-level confusion-category claim |
| S7B_R3_graph_c_gatv2_s5f2 | graph_c_context_observation | gatv2 | test | self_loop | 8 | 78712 | 0.025620 | 0.027210 | 0.000000 | 0.999972 | aggregate edge-kind attention only; no row-level confusion-category claim |

## Interpretation

Sprint 7C explains the observed Sprint 7B behavior as an operating-point
rare-negative separation improvement by Graph C GATv2, conditional on the
identity audit. It does not prove robustness, and it does not prove biological
causality. Graph C GATv2's MCC/specificity gain is meaningful as a secondary
diagnostic, while XGBoost F4 remains the AUPRC bar.

## Artifact Index

Diagnostics:

- `outputs/sprint7c/diagnostics/sprint7c_prediction_alignment_audit.csv`
- `outputs/sprint7c/diagnostics/sprint7c_metric_recomputation.csv`
- `outputs/sprint7c/diagnostics/sprint7c_threshold_transfer.csv`
- `outputs/sprint7c/diagnostics/sprint7c_score_distribution_by_label.csv`
- `outputs/sprint7c/diagnostics/sprint7c_negative_rank_summary.csv`
- `outputs/sprint7c/diagnostics/sprint7c_error_transitions.csv`
- `outputs/sprint7c/diagnostics/sprint7c_per_guide_error_gain.csv`
- `outputs/sprint7c/diagnostics/sprint7c_attention_edge_kind_summary.csv`

Figures:

- `outputs/sprint7c/figures/sprint7c_score_distribution_by_label.png`
- `outputs/sprint7c/figures/sprint7c_graphc_score_delta.png`
- `outputs/sprint7c/figures/sprint7c_negative_rank_shift.png`
- `outputs/sprint7c/figures/sprint7c_error_transition_matrix.png`
- `outputs/sprint7c/figures/sprint7c_threshold_transfer.png`
- `outputs/sprint7c/figures/sprint7c_per_guide_negative_gain.png`
- `outputs/sprint7c/figures/sprint7c_attention_edge_kind_summary.png`

## Verdict

Sprint 7C is complete as an explanation/audit sprint. It validates that the
Sprint 7B Graph C GATv2 rare-negative improvement is row-level real relative to
Graph C GCN, with 53 recovered false positives, 4 lost true negatives, and 12
new false negatives. The result justifies Sprint 7D's mechanism ablation focus
on Graph C GATv2 components, while preserving the boundary that XGBoost F4
remains the AUPRC bar and that attention/context diagnostics are not biological
causal evidence.
