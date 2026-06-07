# Sprint 7 GAT/GATv2 Attention Comparison Report

Run batch: `sprint7_gat_gatv2_seed42_20260607_153444`

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Graph schema: `graph_a_minimal_physical_target`.
- Feature set: `S5F2_energy`.
- Edge feature columns: `268`.
- Training regime: measured-only headline; no `measured=0` putative rows.
- Loss: Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Primary metric: AUPRC. Threshold metrics are secondary interpretation outputs.
- Test positive prevalence: `0.900705`.
- Required reference: `xgboost_unweighted / F4` test AUPRC `0.992522`.
- Sprint 6 Graph A `S5F2_energy` weighted-BCE GCN reference test AUPRC `0.976935`.

Sprint 7 changes only the model architecture for the trained headline runs.
Graph topology, feature family, loss, split, target-node representation,
checkpoint policy, and threshold policy remain fixed.

## Architecture Ladder

| predeclared_run_id | role | architecture | edge-aware attention | heads | parameter_count |
| --- | --- | --- | --- | ---: | ---: |
| `S7R0_gcn_reference` | Sprint 6 weighted-BCE GCN reference; no retrain | GCN | n/a | n/a | n/a |
| `S7R1_gat_edge_aware` | Edge-aware `GATConv` test of the Sprint 6 architecture hypothesis | GAT | true | 4 | 217985 |
| `S7R2_gatv2_edge_aware` | Edge-aware dynamic-attention variant | GATv2 | true | 4 | 250753 |

For the attention runs, candidate-edge `S5F2_energy` features enter message
passing through PyG `edge_attr` / `edge_dim`. Reverse candidate edges duplicate
the same candidate-edge feature row. Self-loop edge features are zero-filled
(`self_loop_edge_fill: 0.0`).

## Result Summary

| predeclared_run_id | architecture | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp | delta_auprc_vs_S7R0 | delta_auprc_vs_F4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7R0_gcn_reference` | GCN | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 | 0.000000 | -0.015587 |
| `S7R1_gat_edge_aware` | GAT | 0.950763 | 0.673950 | 0.481476 | -0.016850 | 0.183432 | 31 | 138 | 316 | 1217 | -0.026172 | -0.041759 |
| `S7R2_gatv2_edge_aware` | GATv2 | 0.965449 | 0.818282 | 0.627128 | 0.291367 | 0.218935 | 37 | 132 | 35 | 1498 | -0.011487 | -0.027073 |

## Threshold Summary

All thresholds are selected on validation data only.

| predeclared_run_id | architecture | threshold | best_epoch | epochs_ran | best_val_auprc | val_mcc | val_specificity | test_mcc | test_specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7R1_gat_edge_aware` | GAT | 0.114547 | 52 | 67 | 0.948275 | 0.378595 | 0.300448 | -0.016850 | 0.183432 |
| `S7R2_gatv2_edge_aware` | GATv2 | 0.142158 | 20 | 35 | 0.984323 | 0.590185 | 0.730942 | 0.291367 | 0.218935 |

## Attention Summary

Attention weights are model-interpretation diagnostics only. They are not
biological causal evidence.

| predeclared_run_id | architecture | edge_kind | total_edges_summarized | mean_attention |
| --- | --- | --- | ---: | ---: |
| `S7R1_gat_edge_aware` | GAT | candidate_forward | 77696 | 0.543017 |
| `S7R1_gat_edge_aware` | GAT | candidate_reverse | 77696 | 0.011253 |
| `S7R1_gat_edge_aware` | GAT | self_loop | 71440 | 0.397192 |
| `S7R2_gatv2_edge_aware` | GATv2 | candidate_forward | 77696 | 0.523584 |
| `S7R2_gatv2_edge_aware` | GATv2 | candidate_reverse | 77696 | 0.009637 |
| `S7R2_gatv2_edge_aware` | GATv2 | self_loop | 71440 | 0.420085 |

## Validation Notes

- Manifest run IDs match the predeclared Sprint 7 run matrix:
  `S7R0_gcn_reference`, `S7R1_gat_edge_aware`, `S7R2_gatv2_edge_aware`.
- Optional edge-blind controls were not executed.
- Graph provenance records `graph_a_minimal_physical_target`,
  `sprint2_main_seed42`, `scheme_a`, and `S5F2_energy: 268`.
- Test prediction universe is unchanged: 1702 rows, 1533 positives, 169
  negatives, positive prevalence `0.900705`.
- Result-table confusion matrices were recomputed from `gat_predictions.csv`
  with the validation-selected thresholds and matched the report exactly.
- Checkpoints contain edge-aware PyG parameters:
  - GAT: `lin_edge.weight` and `att_edge`.
  - GATv2: `lin_edge.weight`.
- `model.pt` checkpoint files are run artifacts and must remain untracked.

## Final Interpretation

Under the locked Scheme A / guide-level / measured-only /
`experiment_id=18`-excluded protocol and the fixed Graph A + `S5F2_energy` +
weighted-BCE setting, **neither predeclared edge-aware attention model improves
over the Sprint 6 GCN reference on the primary metric.**

`S7R1_gat_edge_aware` drops to test AUPRC `0.950763` and degrades threshold
behavior substantially: MCC `-0.016850`, specificity `0.183432`, and
TN/FP/FN/TP `31/138/316/1217`. This is a broad degradation, not an isolated
negative-class tradeoff.

`S7R2_gatv2_edge_aware` is the stronger attention variant. It reaches test
AUPRC `0.965449` and nearly matches the GCN AUROC (`0.818282` vs `0.819972`),
but it remains below the GCN reference on AUPRC (`-0.011487`), MCC
(`0.291367` vs `0.483719`), specificity (`0.218935` vs `0.289941`), and true
negative recovery (`37/169` vs `49/169`).

The Sprint 7 result therefore narrows the Sprint 6 architecture hypothesis:
adding edge-aware GAT/GATv2 message passing to fixed Graph A is not sufficient
to improve the best GNN setting. The best GNN remains the Sprint 6 weighted-BCE
Graph A GCN reference (`S7R0_gcn_reference` / `S6R0_wbce`), and
`xgboost_unweighted / F4` remains the authoritative overall AUPRC bar.

This is not evidence that attention cannot work in other graph formulations.
It is a same-contract result for fixed Graph A with featureless physical target
nodes. Any Graph B/Graph C + GATv2 experiment would change the topology and/or
target semantics and should be reported as a separate exploratory sprint, not
as a post-hoc Sprint 7 rerun.

## Artifact Index

Diagnostic tables:

- `outputs/sprint7/diagnostics/gat_threshold_metrics.csv`
- `outputs/sprint7/diagnostics/gat_comparison_deltas.csv`
- `outputs/sprint7/diagnostics/attention_contract_summary.csv`
- `outputs/sprint7/diagnostics/attention_weight_summary.csv`
- `outputs/sprint7/diagnostics/gat_predictions.csv`
- `outputs/sprint7/diagnostics/gat_training_history.csv`
- `outputs/sprint7/diagnostics/gat_per_guide_score_summary.csv`
- `outputs/sprint7/diagnostics/gat_per_genome_score_summary.csv`
- `outputs/sprint7/diagnostics/gat_score_deciles.csv`

Figures:

- `outputs/sprint7/figures/gat_model_auprc_comparison.png`
- `outputs/sprint7/figures/gat_pr_curves.png`
- `outputs/sprint7/figures/gat_roc_curves.png`
- `outputs/sprint7/figures/gat_training_curves.png`
- `outputs/sprint7/figures/gat_threshold_metrics.png`
- `outputs/sprint7/figures/gat_score_distributions.png`
- `outputs/sprint7/figures/gat_per_guide_metric_distribution.png`
- `outputs/sprint7/figures/attention_weight_summary.png`

Per-run artifacts:

- `outputs/sprint7/runs/sprint7_gat_gatv2_seed42_20260607_153444_S7R1_gat_edge_aware/`
- `outputs/sprint7/runs/sprint7_gat_gatv2_seed42_20260607_153444_S7R2_gatv2_edge_aware/`

## Interpretation Boundaries

- AUPRC is the primary headline metric.
- Specificity, MCC, macro F1, and TN/FP/FN/TP diagnose rare-negative recognition
  and threshold behavior, but they do not override AUPRC.
- Attention summaries are model diagnostics only and must not be described as
  biological causal evidence.
- No test-driven tuning is permitted after this result. Heads, dropout,
  learning rate, thresholds, losses, and graph schema must not be revised from
  the returned test diagnostics.
- Graph B/Graph C + GATv2, sequence encoders, stronger edge MLPs, and multi-seed
  robustness are separate future scopes, not Sprint 7 headline continuations.
