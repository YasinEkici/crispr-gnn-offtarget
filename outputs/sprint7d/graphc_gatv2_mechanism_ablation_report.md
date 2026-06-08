# Sprint 7D Graph C GATv2 Mechanism Ablation Report

Run batch: `sprint7d_graphc_gatv2_mechanism_seed42_20260607_203646`

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- Training regime: measured-only headline; no `measured=0` putative rows.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Primary metric: AUPRC. Threshold metrics are secondary interpretation outputs.
- Test positive prevalence: `0.900705`; negatives are the rare class.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Required reference: `xgboost_unweighted / F4` test AUPRC `0.992522`.
- Architecture policy: only Graph C GATv2 was newly trained in Sprint 7D.
- Attention summaries are model-interpretation artifacts only, not biological causal evidence.

Sprint 7D is a mechanism/isolation ablation, not a model-search sprint. It
does not introduce new graph schemas, losses, samplers, feature families,
sequence encoders, or post-hoc thresholds.

## Mechanism Question

Sprint 7B found that Graph C GATv2 did not win the primary AUPRC metric, but it
gave the strongest same-contract GNN rare-negative operating point observed so
far:

```text
Full Graph C GATv2:
test AUPRC = 0.969078
test MCC = 0.531774
test specificity = 0.372781
TN/FP/FN/TP = 63/106/12/1521
```

Sprint 7C then showed that this behavior was row-level real relative to Graph C
GCN: Graph C GATv2 recovered 53 Graph C GCN false positives as true negatives,
lost 4 previous true negatives, and introduced 12 new false negatives.

Sprint 7D asks which existing Graph C GATv2 component supports that
rare-negative behavior:

1. explicit `context_similar_to` target-observation topology,
2. direct `target_observation` context node features,
3. candidate-edge `S5F2_energy` inside GATv2 attention/message passing,
4. or an interaction among those components.

## Run Ladder

Carry-forward references were not retrained.

| run | setting | role |
| --- | --- | --- |
| `S7D_REF_XGB_F4` | XGBoost `F4` | Authoritative tabular AUPRC reference; no retrain. |
| `S7D_REF_GRAPH_A_GCN` | Graph A GCN + `S5F2_energy` | Sprint 6 weighted-BCE Graph A reference; no retrain. |
| `S7D_REF_GRAPH_A_GATV2` | Graph A GATv2 + `S5F2_energy` | Sprint 7 Graph A attention reference; no retrain. |
| `S7D_REF_GRAPH_C_GCN` | Graph C GCN + `S5F2_energy` | Sprint 5B Graph C energy reference; no retrain. |
| `S7D_REF_FULL_GRAPH_C_GATV2` | Full Graph C GATv2 + `S5F2_energy` | Sprint 7B full Graph C GATv2 reference; no retrain. |
| `S7D_R1_no_context_edges` | Graph C GATv2 without `context_similar_to` edges | Tests whether explicit context-similarity topology is necessary. |
| `S7D_R2_edge_blind_attention` | Graph C GATv2 with candidate `S5F2_energy` zeroed only inside attention/message passing | Tests whether classifier-only access to S5F2 is sufficient. |
| `S7D_R3_mask_target_context_features` | Graph C GATv2 with direct `target_observation` node features masked | Tests whether direct target-observation context features are necessary. |

Graph C is not topology-only. It combines observation-level target semantics,
direct context node features, candidate-edge `S5F2_energy`, and auxiliary
`context_similar_to` target-observation edges.

## Result Summary

| predeclared_run_id | graph_schema | architecture | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7D_REF_XGB_F4` | `tabular_reference` | `xgboost` | 0.992522 | 0.938416 | n/a | 0.345198 | n/a | 38 | 131 | 21 | 1512 |
| `S7D_REF_GRAPH_A_GCN` | `graph_a_minimal_physical_target` | `gcn` | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 |
| `S7D_REF_GRAPH_A_GATV2` | `graph_a_minimal_physical_target` | `gatv2` | 0.965449 | 0.818282 | 0.627128 | 0.291367 | 0.218935 | 37 | 132 | 35 | 1498 |
| `S7D_REF_GRAPH_C_GCN` | `graph_c_context_observation` | `gcn` | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 0.082840 | 14 | 155 | 0 | 1533 |
| `S7D_REF_FULL_GRAPH_C_GATV2` | `graph_c_context_observation` | `gatv2` | 0.969078 | 0.849705 | 0.739526 | 0.531774 | 0.372781 | 63 | 106 | 12 | 1521 |
| `S7D_R1_no_context_edges` | `graph_c_context_observation` | `gatv2` | 0.965598 | 0.850137 | 0.733910 | 0.517970 | 0.366864 | 62 | 107 | 14 | 1519 |
| `S7D_R2_edge_blind_attention` | `graph_c_context_observation` | `gatv2` | 0.945691 | 0.686298 | 0.591245 | 0.182915 | 0.248521 | 42 | 127 | 112 | 1421 |
| `S7D_R3_mask_target_context_features` | `graph_c_context_observation` | `gatv2` | 0.893657 | 0.407817 | 0.473391 | -0.013952 | 0.000000 | 0 | 169 | 3 | 1530 |

## Ablation Deltas

All deltas below compare against the full Graph C GATv2 Sprint 7B reference.

| ablation | delta_test_auprc | delta_test_mcc | delta_specificity | delta_tn | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `S7D_R1_no_context_edges` | -0.003481 | -0.013804 | -0.005917 | -1 | Removing explicit context-similarity topology causes only minor degradation. |
| `S7D_R2_edge_blind_attention` | -0.023388 | -0.348859 | -0.124260 | -21 | Removing S5F2 from attention/message passing substantially weakens ranking and operating-point behavior. |
| `S7D_R3_mask_target_context_features` | -0.075422 | -0.545726 | -0.372781 | -63 | Masking direct target-observation context features collapses true-negative recognition. |

## Validation-Selected Threshold Summary

All thresholds were selected on validation data only.

| predeclared_run_id | threshold | best_epoch | epochs_ran | best_val_auprc | val_mcc | val_specificity | test_mcc | test_specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7D_R1_no_context_edges` | 0.090871 | 49 | 64 | 0.976727 | 0.498385 | 0.493274 | 0.517970 | 0.366864 |
| `S7D_R2_edge_blind_attention` | 0.145281 | 32 | 47 | 0.976784 | 0.321260 | 0.251121 | 0.182915 | 0.248521 |
| `S7D_R3_mask_target_context_features` | 0.156156 | 1 | 16 | 0.835662 | 0.000000 | 0.000000 | -0.013952 | 0.000000 |

The R3 collapse is not only a test-threshold artifact. It is also visible in
validation: best validation AUPRC drops to `0.835662`, validation specificity
is `0.000000`, and the best checkpoint is epoch 1. Test AUROC also drops below
random (`0.407817`), indicating ranking degradation as well as threshold
failure.

## Component Audit

`outputs/sprint7d/diagnostics/graphc_gatv2_component_ablation_audit.csv`
confirms that the ablations changed the intended tensors:

| run | audit finding |
| --- | --- |
| `S7D_R1_no_context_edges` | `context_edges_used = 0` in train/val/test; candidate attention attributes, classifier S5F2 attributes, and target-observation features remain nonzero. |
| `S7D_R2_edge_blind_attention` | candidate attention attribute absolute sum is `0.0`; classifier candidate-edge S5F2 absolute sum remains nonzero in train/val/test. |
| `S7D_R3_mask_target_context_features` | target-observation feature absolute sum after masking is `0.0`; context topology and candidate S5F2 attention attributes remain active. |

This audit is important for interpretation. R2 does not remove `S5F2_energy`
from the final edge classifier; it removes the candidate energy vector only
from GATv2 attention/message passing. R3 does not remove all context-derived
information; `context_similar_to` topology remains, but direct
`target_observation` node features are zeroed.

## Mechanism Interpretation

### R1: Context-Similarity Topology Is Not The Main Driver

Removing `context_similar_to` edges preserves nearly the full Graph C GATv2
rare-negative behavior:

```text
Full Graph C GATv2:  AUPRC 0.969078, MCC 0.531774, specificity 0.372781, TN 63
No context edges:    AUPRC 0.965598, MCC 0.517970, specificity 0.366864, TN 62
```

The small decline does not prove that context edges are useless. It does show
that explicit context-similarity topology is not required to reproduce most of
the Sprint 7B rare-negative operating point under this split. Direct
`target_observation` context features and candidate `S5F2_energy` edge-aware
attention remain active in R1, so the correct conclusion is not "context is
irrelevant"; it is that the observed behavior is not primarily
context-topology-driven.

### R2: S5F2 Must Enter Message Passing, Not Only The Classifier

R2 keeps `S5F2_energy` in the final edge classifier but zeroes it inside GATv2
attention/message passing. The degradation is substantial:

```text
Full Graph C GATv2:  AUPRC 0.969078, MCC 0.531774, TN/FN 63/12
Edge-blind attention: AUPRC 0.945691, MCC 0.182915, TN/FN 42/112
```

This supports the Sprint 7 architecture hypothesis in a Graph C-specific way:
classifier-only access to `S5F2_energy` is not sufficient to reproduce the
full operating-point behavior. Routing candidate energy features through the
GATv2 `edge_attr` / `edge_dim` path contributes materially to ranking and
threshold behavior. This remains a model-mechanism finding, not biological
causal evidence about binding energy.

### R3: Direct Target-Observation Context Features Are The Strongest Signal

R3 masks the direct `target_observation` node feature matrix while preserving
`context_similar_to` topology and edge-aware `S5F2_energy`. The model collapses
on negative recognition:

```text
Full Graph C GATv2:      AUPRC 0.969078, AUROC 0.849705, MCC 0.531774, TN 63
Masked target features: AUPRC 0.893657, AUROC 0.407817, MCC -0.013952, TN 0
```

This is the strongest Sprint 7D signal. Context-derived topology alone is not
enough to preserve rare-negative recognition when direct target-observation
context features are removed. Under the locked split, the observed Graph C
GATv2 behavior appears primarily supported by direct context node features,
with edge-aware `S5F2_energy` message passing as an important complementary
mechanism.

## Attention Summary

Attention summaries were written for the three trained ablations:

- `S7D_R1_no_context_edges`: edge kinds include `candidate_forward`,
  `candidate_reverse`, and `self_loop`; `context_similar_to` is absent by
  design.
- `S7D_R2_edge_blind_attention`: edge kinds include `candidate_forward`,
  `candidate_reverse`, `context_similar_to`, and `self_loop`; candidate edge
  attributes are zero in attention by design.
- `S7D_R3_mask_target_context_features`: edge kinds include `candidate_forward`,
  `candidate_reverse`, `context_similar_to`, and `self_loop`; direct
  target-observation features are zero by design.

These summaries validate that the expected edge kinds participate in the
attention path. They should not be used to infer biological causality, and they
should not be used to tune a new threshold or model.

## Final Interpretation

Sprint 7D explains the Sprint 7B Graph C GATv2 rare-negative result as a
context-feature-driven mechanism rather than a context-topology-driven one.

The full Graph C GATv2 reference remains the strongest same-contract GNN
operating-point profile observed so far (MCC `0.531774`, specificity
`0.372781`, TN `63`). Removing explicit `context_similar_to` topology preserves
almost the same behavior (MCC `0.517970`, specificity `0.366864`, TN `62`),
so the auxiliary context-similarity edges are not the main driver under this
split. In contrast, removing candidate `S5F2_energy` from GATv2
attention/message passing substantially degrades both AUPRC and threshold
behavior despite keeping S5F2 in the final classifier. Masking direct
`target_observation` context node features collapses specificity to zero and
removes all true-negative recognition.

The defensible mechanism claim is:

> Graph C GATv2's rare-negative operating-point gain appears to depend
> primarily on direct `target_observation` context node features, supported by
> edge-aware `S5F2_energy` message passing. Explicit `context_similar_to`
> topology is not the primary driver in the locked single-seed Scheme A split.

This is not a primary-metric model win. XGBoost F4 remains the authoritative
AUPRC bar (`0.992522`), and Graph A GCN remains stronger than full Graph C
GATv2 on AUPRC (`0.976935` vs `0.969078`). Sprint 7D should therefore be
reported as mechanism evidence for the Graph C GATv2 operating point, not as
statistical superiority, biological causality, or a new model-selection result.

## Next-Step Implications

The next model-improvement question should focus on the signal Sprint 7D
actually identified:

- direct `target_observation` context features;
- how those context features are encoded;
- how candidate `S5F2_energy` is routed through edge-aware message passing.

More context topology is not the most supported next direction from Sprint 7D.
A cleaner next modeling sprint would be a predeclared context-feature subgroup
ablation and/or target-observation context encoder improvement. Multi-seed,
guide-level bootstrap, and paired-difference consolidation remain the
appropriate robustness layer before making stable superiority claims.

## Artifact Index

Top-level artifacts:

- `outputs/sprint7d/graphc_gatv2_mechanism_ablation.csv`
- `outputs/sprint7d/graphc_gatv2_mechanism_ablation_report.md`
- `outputs/sprint7d/graphc_gatv2_mechanism_ablation_run_manifest.json`
- `outputs/sprint7d/graph_artifact_provenance.json`

Diagnostic tables:

- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_predictions.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_training_history.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_threshold_metrics.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_deltas.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_component_ablation_audit.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_attention_summary.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_attention_contract_summary.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_per_guide_score_summary.csv`
- `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_score_deciles.csv`

Figures:

- `outputs/sprint7d/figures/graphc_gatv2_mechanism_auprc_comparison.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_pr_curves.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_roc_curves.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_training_curves.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_threshold_metrics.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_score_distributions.png`
- `outputs/sprint7d/figures/graphc_gatv2_mechanism_attention_by_edge_kind.png`

Per-run artifacts:

- `outputs/sprint7d/runs/sprint7d_graphc_gatv2_mechanism_seed42_20260607_203646_S7D_R1_no_context_edges/`
- `outputs/sprint7d/runs/sprint7d_graphc_gatv2_mechanism_seed42_20260607_203646_S7D_R2_edge_blind_attention/`
- `outputs/sprint7d/runs/sprint7d_graphc_gatv2_mechanism_seed42_20260607_203646_S7D_R3_mask_target_context_features/`

Each per-run directory contains `resolved_config.yaml`, `runtime.json`,
`training_history.csv`, `metrics.csv`, `attention_summary.csv`, and
`component_audit.csv`. Local repository evidence intentionally excludes
`model.pt` checkpoint files; those remain Drive-held run artifacts.

## Interpretation Boundaries

- AUPRC remains the primary metric.
- MCC, macro F1, specificity, and true negatives are secondary threshold
  diagnostics.
- Sprint 7D is single-seed/single-split mechanism evidence, not statistical
  superiority.
- Graph C must not be described as topology-only; it changes target semantics
  and direct context feature availability.
- R3 masks direct target-observation context node features; it does not remove
  all context-derived topology.
- Attention weights are interpretation-only model signals, not biological
  causal evidence.
- XGBoost F4 remains the required AUPRC reference.
- Smoke or mocked outputs are not final Sprint 7D performance evidence; this
  report uses the returned Colab batch
  `sprint7d_graphc_gatv2_mechanism_seed42_20260607_203646`.
- Test diagnostics must not be used to retune thresholds, losses, features,
  topology, hyperparameters, or architecture inside Sprint 7D.
