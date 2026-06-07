# Sprint 7B GATv2 Topology/Context Follow-Up Report

Run batch: `sprint7b_gatv2_topology_seed42_20260607_174739`

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- Training regime: measured-only headline; no `measured=0` putative rows.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Primary metric: AUPRC. Threshold metrics are secondary interpretation outputs.
- Test positive prevalence: `0.900705`.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Required reference: `xgboost_unweighted / F4` test AUPRC `0.992522`.
- Attention summaries are model-interpretation artifacts only, not biological causal evidence.

## Baseline Reference

- Authoritative bar: `xgboost_unweighted / F4`.
- XGBoost F4 test AUPRC: `0.992522`.
- XGBoost F4 test AUROC: `0.938416`.
- XGBoost F4 test MCC: `0.345198`.
- No Sprint 7B GNN run exceeds the XGBoost F4 AUPRC bar.
- Sprint 7B Graph C GATv2 exceeds the XGBoost F4 MCC under the validation-selected threshold policy (`0.531774` vs. `0.345198`), but this is a secondary threshold metric and must not be reported as an AUPRC win.

## Run Ladder

Sprint 7B is a follow-up sensitivity, not a new primary architecture search. It asks whether the Sprint 7 edge-aware GATv2 recipe behaves differently when the graph topology/target semantics change.

| run | setting | role |
| --- | --- | --- |
| `S7B_REF_XGB_F4` | XGBoost `F4` | Authoritative tabular reference; no retrain. |
| `S7B_REF_GA_GCN` | Graph A GCN + `S5F2_energy` | Sprint 6 weighted-BCE Graph A reference; no retrain. |
| `S7B_REF_GA_GATV2` | Graph A GATv2 + `S5F2_energy` | Sprint 7 Graph A GATv2 reference; no retrain. |
| `S7B_REF_GC_GCN` | Graph C GCN + `S5F2_energy` | Sprint 5B Graph C energy reference; no retrain. |
| `S7B_R1_graph_b_gcn_s5f2` | Graph B GCN + `S5F2_energy` | Matched Graph B GCN reference produced in Sprint 7B. |
| `S7B_R2_graph_b_gatv2_s5f2` | Graph B GATv2 + `S5F2_energy` | Tests guide-similarity topology with edge-aware GATv2. |
| `S7B_R3_graph_c_gatv2_s5f2` | Graph C GATv2 + `S5F2_energy` | Tests context-observation target semantics with edge-aware GATv2. |

Graph B keeps Graph A physical-target semantics and adds label-free `sequence_similar_to` guide topology. Candidate edges use `S5F2_energy`; `sequence_similar_to` edges are topology-only and receive zero edge attributes in GATv2 message passing.

Graph C uses `target_observation` nodes with context features and `context_similar_to` edges. Candidate edges use `S5F2_energy`; context edges are topology/context-neighbor links and receive zero edge attributes in GATv2 message passing. Graph C is not topology-only because it changes both topology and target-node semantics.

## Result Summary

| predeclared_run_id | graph_schema | architecture | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7B_REF_XGB_F4` | `tabular_reference` | `xgboost` | 0.992522 | 0.938416 | n/a | 0.345198 | n/a | 38 | 131 | 21 | 1512 |
| `S7B_REF_GA_GCN` | `graph_a_minimal_physical_target` | `gcn` | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 |
| `S7B_REF_GA_GATV2` | `graph_a_minimal_physical_target` | `gatv2` | 0.965449 | 0.818282 | 0.627128 | 0.291367 | 0.218935 | 37 | 132 | 35 | 1498 |
| `S7B_REF_GC_GCN` | `graph_c_context_observation` | `gcn` | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 0.082840 | 14 | 155 | 0 | 1533 |
| `S7B_R1_graph_b_gcn_s5f2` | `graph_b_guide_similarity_control` | `gcn` | 0.973583 | 0.800777 | 0.674412 | 0.437157 | 0.254438 | 43 | 126 | 8 | 1525 |
| `S7B_R2_graph_b_gatv2_s5f2` | `graph_b_guide_similarity_control` | `gatv2` | 0.979139 | 0.832397 | 0.561471 | 0.272970 | 0.094675 | 16 | 153 | 2 | 1531 |
| `S7B_R3_graph_c_gatv2_s5f2` | `graph_c_context_observation` | `gatv2` | 0.969078 | 0.849705 | 0.739526 | 0.531774 | 0.372781 | 63 | 106 | 12 | 1521 |

## Matched Comparisons

All comparisons use the locked measured-only Scheme A test universe. AUPRC is the primary metric; MCC, macro F1, specificity, and TN are threshold diagnostics under the validation-selected threshold.

| comparison | delta_test_auprc | delta_test_mcc | delta_specificity | interpretation |
| --- | ---: | ---: | ---: | --- |
| Graph B GATv2 vs. Graph B GCN | +0.005556 | -0.164187 | -0.159763 | Graph B GATv2 improves AUPRC but weakens negative-class recognition. |
| Graph C GATv2 vs. Graph C GCN | -0.003402 | +0.257487 | +0.289941 | Graph C GATv2 keeps AUPRC competitive and strongly improves the operating-point classifier. |
| Graph C GATv2 vs. Graph A GCN | -0.007857 | +0.048055 | +0.082840 | Graph C GATv2 improves MCC/specificity over the best Graph A GCN reference while staying in the same high-AUPRC regime. |
| Graph C GATv2 vs. XGBoost F4 | -0.023444 | +0.186576 | n/a | XGBoost remains the AUPRC bar; Graph C GATv2 is stronger on MCC under the selected threshold. |

## Threshold And Negative-Class Interpretation

Sprint 7B's strongest operating-point result is `S7B_R3_graph_c_gatv2_s5f2`:

```text
TN = 63
FP = 106
FN = 12
TP = 1521
MCC = 0.531774
Macro F1 = 0.739526
Specificity = 0.372781
```

This is the best GNN negative-class profile observed so far in the Sprint 4-7B same-contract sequence. It improves over the Sprint 6 Graph A weighted-BCE reference (`MCC 0.483719`, specificity `0.289941`, TN `49`) and over the Sprint 5B Graph C GCN reference (`MCC 0.274287`, specificity `0.082840`, TN `14`). The improvement is not just a small threshold artifact: Graph C GATv2 recognizes 63 of 169 test negatives, compared with 49 for Graph A GCN and 14 for Graph C GCN.

The AUPRC tradeoff is modest but real. Graph C GATv2 test AUPRC is `0.969078`, below Graph A GCN `0.976935`, Graph B GATv2 `0.979139`, and XGBoost F4 `0.992522`. However, it remains well above the test positive-prevalence floor (`0.900705`) and in the same high-AUPRC regime as the previous GNN baselines. The defensible claim is therefore not that Graph C GATv2 wins the primary metric; it is that Graph C target-observation semantics plus edge-aware GATv2 substantially improve rare-negative recognition while keeping AUPRC competitive.

Graph B tells a different story. `S7B_R2_graph_b_gatv2_s5f2` is the best Sprint 7B GNN by AUPRC (`0.979139`) and improves over the matched Graph B GCN by `+0.005556`, but its threshold metrics collapse relative to Graph B GCN (MCC `0.272970` vs. `0.437157`, specificity `0.094675` vs. `0.254438`, TN `16` vs. `43`). Guide-similarity topology alone is therefore not supported as the explanation for the Graph C GATv2 negative-class gain.

## Attention Summary

Attention summaries were written for the two GATv2 runs:

- `S7B_R2_graph_b_gatv2_s5f2`: edge kinds include `candidate_forward`, `candidate_reverse`, `sequence_similar_to`, and `self_loop`.
- `S7B_R3_graph_c_gatv2_s5f2`: edge kinds include `candidate_forward`, `candidate_reverse`, `context_similar_to`, and `self_loop`.

These summaries validate that auxiliary graph edges participate in the GATv2 attention path and that candidate-edge `S5F2_energy` features enter GATv2 through the `edge_attr` / `edge_dim` path. They should be used as model-behavior diagnostics only. They are not biological causal evidence, and they do not identify binding-energy or context features as causal drivers.

## Final Interpretation

Sprint 7B produces a useful split result:

- **Graph B GATv2** improves AUPRC but does not improve negative-class recognition. This argues against a simple "attention over guide-similarity topology fixes imbalance" interpretation.
- **Graph C GATv2** does not win AUPRC, but it gives the best same-contract GNN threshold profile so far: MCC `0.531774`, macro F1 `0.739526`, specificity `0.372781`, and TN/FP/FN/TP `63/106/12/1521`.

The strongest scientific framing is: under the frozen Scheme A / guide-level / measured-only / validation-only threshold protocol, Graph C target-observation context semantics combined with edge-aware GATv2 improve rare-negative separation while preserving competitive AUPRC. This is a meaningful Sprint 7B result and a strong candidate for Sprint 8 robustness, but it is not a guaranteed architecture win and not an AUPRC replacement for XGBoost F4.

No test-driven tuning was performed in Sprint 7B. The run matrix, loss, feature set, split, validation checkpoint policy, and validation threshold policy were predeclared. The results are still single-seed/single-split; paired-difference confidence intervals, guide-level bootstrap, and/or multi-seed consolidation remain the appropriate Sprint 8 robustness layer before treating the Graph C GATv2 gain as stable.

## Artifact Index

Top-level artifacts:

- `outputs/sprint7b/gatv2_topology_comparison.csv`
- `outputs/sprint7b/gatv2_topology_report.md`
- `outputs/sprint7b/gatv2_topology_run_manifest.json`
- `outputs/sprint7b/graph_artifact_provenance.json`
- `outputs/sprint7b/graph_b_s5f2_artifact_report.md`

Diagnostic tables:

- `outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_training_history.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_threshold_metrics.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_deltas.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_attention_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_attention_contract_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_per_guide_score_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_per_genome_score_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_score_deciles.csv`

Figures:

- `outputs/sprint7b/figures/gatv2_topology_auprc_comparison.png`
- `outputs/sprint7b/figures/gatv2_topology_pr_curves.png`
- `outputs/sprint7b/figures/gatv2_topology_roc_curves.png`
- `outputs/sprint7b/figures/gatv2_topology_training_curves.png`
- `outputs/sprint7b/figures/gatv2_topology_threshold_metrics.png`
- `outputs/sprint7b/figures/gatv2_topology_score_distributions.png`
- `outputs/sprint7b/figures/gatv2_topology_per_guide_metric_distribution.png`
- `outputs/sprint7b/figures/gatv2_topology_attention_by_edge_kind.png`

Per-run artifacts:

- `outputs/sprint7b/runs/sprint7b_gatv2_topology_seed42_20260607_174739_S7B_R1_graph_b_gcn_s5f2/`
- `outputs/sprint7b/runs/sprint7b_gatv2_topology_seed42_20260607_174739_S7B_R2_graph_b_gatv2_s5f2/`
- `outputs/sprint7b/runs/sprint7b_gatv2_topology_seed42_20260607_174739_S7B_R3_graph_c_gatv2_s5f2/`

Each per-run directory contains `resolved_config.yaml`, `runtime.json`, `training_history.csv`, `metrics.csv`, and `attention_summary.csv` where applicable. Local repository evidence intentionally excludes `model.pt` checkpoint files; those remain Drive-held run artifacts.

## Interpretation Boundaries

- AUPRC remains the primary metric. MCC, macro F1, specificity, and true negatives are secondary threshold diagnostics.
- Graph C must not be described as topology-only; it changes both topology and target semantics/context representation.
- Attention weights are interpretation-only model signals, not biological causal evidence.
- Graph B and Graph C are not direct replacements for XGBoost F4; XGBoost F4 remains the required AUPRC reference.
- Smoke or mocked outputs are not final Sprint 7B performance evidence; this report uses the returned Colab batch `sprint7b_gatv2_topology_seed42_20260607_174739`.
- Test diagnostics must not be used to retune thresholds, losses, features, topology, or architecture inside Sprint 7B.
