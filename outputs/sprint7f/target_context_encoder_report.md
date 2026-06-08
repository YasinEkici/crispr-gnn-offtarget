# Sprint 7F Family-Aware Target-Observation Context Encoder Report

Run batch: `sprint7f_target_context_encoder_seed42_20260608_051125`

## Executive Summary

Sprint 7F followed the Sprint 7D and Sprint 7E mechanism evidence. Sprint 7D
showed that Graph C GATv2 rare-negative behavior was mainly supported by direct
`target_observation` context features rather than explicit `context_similar_to`
topology. Sprint 7E then showed that the six direct experimental epigenetic
target-observation features were necessary for that behavior under the
no-context-edge Graph C GATv2 setting.

Sprint 7F therefore changed only one controlled variable: the
`target_observation` encoder used before Graph C GATv2 message passing. Graph C,
GATv2, `S5F2_energy`, weighted BCE, the fixed `sprint2_main_seed42` split,
validation-only checkpointing, validation-only thresholding, and the
no-context-edge policy were held fixed.

The strongest defensible result is that family-aware target-observation encoding
materially improves the Graph C/no-context-edge GATv2 line. The best primary
ranking result is `S7F_R3_family_aware_experimental_emphasis`, with test AUPRC
`0.984945` and AUROC `0.926551`. This substantially narrows but does not close
the gap to XGBoost F4 (`0.992522` AUPRC). The strongest rare-negative
operating-point result is `S7F_R2_family_aware_context_encoder`, with MCC
`0.603489`, specificity `0.650888`, test macro F1 `0.801716`, and TN/FP/FN/TP
`110/59/63/1470` under the validation-selected threshold.

The unified deep encoder is an important control. It has more target-encoder
parameters than the family-aware encoders but does not reproduce their gains
(`0.969201` AUPRC, `0.500460` MCC). This supports the interpretation that the
family-aware feature structure, not raw encoder capacity alone, mattered in this
single-seed locked contract.

Sprint 7F is not a final robustness claim and not biological causal evidence.
It is a strong single-seed model-mechanism result that identifies family-aware
target context encoding as the most promising GNN direction so far.

## Frozen Contract

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`.
- Split ID: `sprint2_main_seed42`; guide-disjoint split.
- Training/evaluation universe: measured-only headline rows; experiment `18`
  excluded by the frozen upstream contract.
- Graph schema: `graph_c_context_observation`.
- Graph visibility: `strict_inductive_primary`.
- Candidate edge feature set: `S5F2_energy`, 268 columns.
- Target-observation feature table: `target_observation_features`, 212 columns.
- Loss: Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Graph C topology policy: `context_similar_to` edges dropped for all newly
  trained Sprint 7F rows.
- Attention policy: candidate `S5F2_energy` remains active in GATv2 attention
  through `edge_attr` / `edge_dim` and remains active in the final edge
  classifier.
- Primary metric: AUPRC.
- Secondary diagnostics: AUROC, macro F1, MCC, specificity/TNR, sensitivity,
  TN/FP/FN/TP, score distributions, attention summaries, and target-encoder
  activation summaries.
- Test positive prevalence context: `0.900705`; negatives are the rare class.
- Authoritative AUPRC bar: `xgboost_unweighted / F4`, test AUPRC `0.992522`.

## Run Matrix

Carry-forward references were not retrained.

| Run | Role | Setting |
| --- | --- | --- |
| `S7F_REF_XGB_F4` | authoritative reference | Sprint 2 XGBoost F4 tabular baseline |
| `S7F_REF_GRAPH_A_GCN` | carry-forward GNN reference | Sprint 6 Graph A GCN + `S5F2_energy` + weighted BCE |
| `S7F_REF_GRAPH_C_GCN` | carry-forward Graph C reference | Sprint 5B Graph C GCN |
| `S7F_REF_FULL_GRAPH_C_GATV2` | carry-forward attention reference | Sprint 7B full Graph C GATv2 |
| `S7F_REF_NO_CONTEXT_EDGE_GATV2` | primary Sprint 7F base | Sprint 7D no-context-edge Graph C GATv2 |
| `S7F_R1_unified_deep_context_encoder` | new 7F run | deeper unified encoder over all 212 target-observation columns |
| `S7F_R2_family_aware_context_encoder` | new 7F run | balanced family branches for target sequence, experimental epigenetic, computed aggregates, and missingness |
| `S7F_R3_family_aware_experimental_emphasis` | new 7F run | family-aware encoder with more capacity assigned to the experimental epigenetic branch |

## Result Table

| predeclared_run_id | encoder | test_auprc | test_auroc | test_macro_f1 | test_mcc | specificity | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7F_REF_XGB_F4` | n/a | 0.992522 | 0.938416 |  | 0.345198 |  | 38 | 131 | 21 | 1512 |
| `S7F_REF_GRAPH_A_GCN` | n/a | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 |
| `S7F_REF_GRAPH_C_GCN` | n/a | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 0.082840 | 14 | 155 | 0 | 1533 |
| `S7F_REF_FULL_GRAPH_C_GATV2` | unified shallow reference | 0.969078 | 0.849705 | 0.739526 | 0.531774 | 0.372781 | 63 | 106 | 12 | 1521 |
| `S7F_REF_NO_CONTEXT_EDGE_GATV2` | unified shallow reference, no context edges | 0.965598 | 0.850137 | 0.733910 | 0.517970 | 0.366864 | 62 | 107 | 14 | 1519 |
| `S7F_R1_unified_deep_context_encoder` | `unified_deep` | 0.969201 | 0.848765 | 0.707321 | 0.500460 | 0.301775 | 51 | 118 | 5 | 1528 |
| `S7F_R2_family_aware_context_encoder` | `family_aware` | 0.982062 | 0.906557 | 0.801716 | 0.603489 | 0.650888 | 110 | 59 | 63 | 1470 |
| `S7F_R3_family_aware_experimental_emphasis` | `family_aware_experimental_emphasis` | 0.984945 | 0.926551 | 0.777185 | 0.568108 | 0.497041 | 84 | 85 | 31 | 1502 |

## Deltas Against Primary Sprint 7F Base

Primary Sprint 7F base: `S7F_REF_NO_CONTEXT_EDGE_GATV2`, AUPRC `0.965598`,
MCC `0.517970`, specificity `0.366864`, TN/FP/FN/TP `62/107/14/1519`.

| Run | Delta AUPRC | Delta MCC | Delta Macro F1 | Test Macro F1 | Specificity | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `S7F_R1_unified_deep_context_encoder` | +0.003603 | -0.017510 | -0.026589 | 0.707321 | 0.301775 | 51/118/5/1528 |
| `S7F_R2_family_aware_context_encoder` | +0.016464 | +0.085519 | +0.067806 | 0.801716 | 0.650888 | 110/59/63/1470 |
| `S7F_R3_family_aware_experimental_emphasis` | +0.019347 | +0.050138 | +0.043275 | 0.777185 | 0.497041 | 84/85/31/1502 |

Both family-aware rows improve the primary no-context-edge Graph C GATv2 base
on AUPRC, AUROC, MCC, and rare-negative recognition. The unified deep row gives
only a small AUPRC increase and loses MCC/specificity relative to the base.

## R2 Versus R3

Sprint 7F has two different winners, and the report should keep them separate.

`S7F_R3_family_aware_experimental_emphasis` is the best Sprint 7F GNN by the
primary threshold-free ranking metrics:

```text
R3 AUPRC = 0.984945
R3 AUROC = 0.926551
```

`S7F_R2_family_aware_context_encoder` is the best validation-threshold
rare-negative operating point:

```text
R2 MCC = 0.603489
R2 specificity = 0.650888
R2 macro F1 = 0.801716
R2 TN/FP/FN/TP = 110/59/63/1470
```

This operating point has a visible tradeoff. R2 recovers many more true
negatives than R3 (`110` versus `84`) but creates more false negatives (`63`
versus `31`). R3 has the stronger AUPRC/AUROC and better sensitivity profile;
R2 has the stronger negative-class threshold profile and the best test macro F1
(`0.801716` versus R3's `0.777185`).

The defensible wording is:

> `S7F_R3` is the best Sprint 7F GNN by primary AUPRC/AUROC, while `S7F_R2` is
> the best rare-negative operating-point model under the frozen
> validation-selected threshold policy.

## Capacity-Control Interpretation

The unified deep encoder has more target-encoder parameters than the family
aware encoders:

| Encoder | Target-Encoder Params | Total Model Params | Test AUPRC | Test MCC |
| --- | ---: | ---: | ---: | ---: |
| `unified_deep` | 44,032 | 294,657 | 0.969201 | 0.500460 |
| `family_aware` | 24,976 | 275,601 | 0.982062 | 0.603489 |
| `family_aware_experimental_emphasis` | 23,528 | 274,153 | 0.984945 | 0.568108 |

This is the most important Sprint 7F design control. The smaller family-aware
encoders outperform the larger unified-deep encoder. Therefore the improvement
is not well explained by adding parameters alone. It is more consistent with
preserving source-defined target-context feature families during encoding.

This remains a single-seed model-mechanism interpretation. It does not prove
that family-aware encoding will dominate across seeds or future datasets.

## Validation-Selected Threshold Behavior

All thresholds were selected on validation data only.

| Run | Threshold | Best Epoch | Best Val AUPRC | Val MCC | Val Specificity | Test MCC | Test Specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7F_R1_unified_deep_context_encoder` | 0.053598 | 76 | 0.976594 | 0.302701 | 0.156951 | 0.500460 | 0.301775 |
| `S7F_R2_family_aware_context_encoder` | 0.099794 | 34 | 0.977541 | 0.383438 | 0.286996 | 0.603489 | 0.650888 |
| `S7F_R3_family_aware_experimental_emphasis` | 0.104111 | 55 | 0.987522 | 0.636566 | 0.834081 | 0.568108 | 0.497041 |

R3 is also strongest on validation AUPRC (`0.987522`) and validation MCC
(`0.636566`). The test-threshold behavior differs from validation because the
negative class is small and the split is fixed. This is another reason not to
overstate single-seed threshold superiority.

## Encoder Activation Summary

The target-context activation summaries are interpretation artifacts, not
biological causal evidence.

| Run | Family | Input Columns | Branch Dim | Mean L2 |
| --- | --- | ---: | ---: | ---: |
| `S7F_R1_unified_deep_context_encoder` | all target-observation features | 212 | 128 | 3.506358 |
| `S7F_R2_family_aware_context_encoder` | target sequence one-hot | 115 | 32 | 3.938170 |
| `S7F_R2_family_aware_context_encoder` | experimental epigenetic | 6 | 32 | 4.212251 |
| `S7F_R2_family_aware_context_encoder` | computed nucleosome aggregates | 78 | 48 | 4.968620 |
| `S7F_R2_family_aware_context_encoder` | computed nucleosome missingness | 13 | 16 | 2.500451 |
| `S7F_R3_family_aware_experimental_emphasis` | target sequence one-hot | 115 | 24 | 3.507987 |
| `S7F_R3_family_aware_experimental_emphasis` | experimental epigenetic | 6 | 48 | 4.920774 |
| `S7F_R3_family_aware_experimental_emphasis` | computed nucleosome aggregates | 78 | 40 | 4.519160 |
| `S7F_R3_family_aware_experimental_emphasis` | computed nucleosome missingness | 13 | 16 | 2.349201 |

The experimental-emphasis model assigns more branch capacity to the six
experimental epigenetic features and shows the largest experimental-branch L2
summary (`4.920774`). This is consistent with Sprint 7E's finding that direct
experimental epigenetic features were necessary, but it should not be read as
evidence that those features are biologically causal.

## Contract Audit

`outputs/sprint7f/diagnostics/target_context_encoder_audit.csv` confirms the
intended controlled setup:

- Target-observation feature dimension is `212`.
- Feature-family partition is stable: 115 target sequence columns, 6
  experimental epigenetic columns, 78 computed nucleosome aggregate columns,
  and 13 computed nucleosome missingness columns.
- `context_edges_used = 0` for all trained Sprint 7F train/validation/test
  views.
- Candidate `S5F2_energy` attention attributes remain nonzero in train,
  validation, and test.
- Candidate `S5F2_energy` classifier attributes remain nonzero in train,
  validation, and test.

This preserves the Sprint 7F controlled variable: the target-context encoder
changed, while topology, edge-aware attention, loss, split, checkpointing, and
thresholding stayed fixed.

## Relation To XGBoost F4

XGBoost F4 remains the authoritative matched-contract AUPRC bar:

```text
XGBoost F4 AUPRC = 0.992522
Best Sprint 7F GNN AUPRC = 0.984945
```

Sprint 7F substantially narrows the GNN AUPRC gap but does not close it. The
correct primary-metric claim is therefore not that GNN has beaten XGBoost. The
claim is that family-aware Graph C GATv2 is now the strongest same-contract GNN
line observed so far by AUPRC.

R2 does exceed XGBoost F4 on secondary threshold diagnostics under the frozen
validation-selected threshold policy:

```text
XGBoost F4: MCC 0.345198, TN/FP/FN/TP 38/131/21/1512
S7F_R2:     MCC 0.603489, TN/FP/FN/TP 110/59/63/1470
```

This is a rare-negative operating-point advantage, not a replacement for the
primary AUPRC ranking.

## Scientific Interpretation

Sprint 7F supports the following mechanism-level interpretation:

> Family-aware encoding of Graph C target-observation context better exploits
> the Sprint 7E-identified context signal than a unified encoder under the
> frozen single-seed Scheme A contract.

The result follows directly from the controlled ladder:

1. Sprint 7D showed that explicit context-similarity topology was not the main
   driver of the Graph C GATv2 rare-negative behavior.
2. Sprint 7E showed that direct experimental epigenetic target-observation
   features were necessary for that behavior.
3. Sprint 7F showed that preserving target-observation feature-family structure
   during encoding improves both primary ranking metrics and rare-negative
   threshold diagnostics.

The strongest primary result is R3. The strongest operating-point result is R2.
The weaker R1 capacity control makes the family-aware interpretation more
credible than a simple "larger encoder" explanation.

## Claim Boundaries

- Sprint 7F is single-seed and fixed-split. It supports a strong model
  hypothesis, not statistical superiority.
- AUPRC remains the primary metric. MCC, specificity, TN/FP, and macro F1 are
  threshold diagnostics.
- The test positive prevalence is high (`0.900705`), so threshold metrics can
  move sharply with the validation-selected threshold and the small negative
  count.
- R2's rare-negative gain has a cost: it increases false negatives relative to
  R3 and the no-context-edge reference.
- Experimental epigenetic features may reflect biological context,
  assay/source/cell-line structure, coverage, or a combination. Sprint 7F does
  not resolve that proxy question.
- Attention summaries and encoder activation summaries are model-interpretation
  artifacts only, not biological causal evidence.
- Sprint 7F does not make a topology claim; every trained row used the
  no-context-edge base.

## Next-Step Recommendation

The report should close Sprint 7F as a successful model-improvement sprint.
The most defensible next scientific step is Sprint 8 robustness rather than
another single-seed encoder tweak.

Minimal Sprint 8 candidates:

- `S7F_REF_XGB_F4`
- `S7F_REF_GRAPH_A_GCN`
- `S7F_REF_NO_CONTEXT_EDGE_GATV2`
- `S7F_R2_family_aware_context_encoder`
- `S7F_R3_family_aware_experimental_emphasis`

The key Sprint 8 question should be:

> Do the R2/R3 gains survive multi-seed, guide-level bootstrap, and paired
> comparison uncertainty under the same frozen evaluation contract?

## Artifact Index

Primary outputs:

- `outputs/sprint7f/target_context_encoder_comparison.csv`
- `outputs/sprint7f/target_context_encoder_report.md`
- `outputs/sprint7f/target_context_encoder_run_manifest.json`
- `outputs/sprint7f/graph_artifact_provenance.json`

Diagnostic tables:

- `outputs/sprint7f/diagnostics/target_context_encoder_threshold_metrics.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_deltas.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_audit.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_activation_summary.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_activation_contract_summary.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_attention_summary.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_attention_contract_summary.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_predictions.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_per_guide_score_summary.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_score_deciles.csv`
- `outputs/sprint7f/diagnostics/target_context_encoder_training_history.csv`

Figures:

- `outputs/sprint7f/figures/target_context_encoder_auprc_comparison.png`
- `outputs/sprint7f/figures/target_context_encoder_threshold_metrics.png`
- `outputs/sprint7f/figures/target_context_encoder_pr_curves.png`
- `outputs/sprint7f/figures/target_context_encoder_roc_curves.png`
- `outputs/sprint7f/figures/target_context_encoder_score_distributions.png`
- `outputs/sprint7f/figures/target_context_encoder_training_curves.png`
- `outputs/sprint7f/figures/target_context_encoder_attention_by_edge_kind.png`
- `outputs/sprint7f/figures/target_context_encoder_branch_activation_norms.png`

## Verdict

Sprint 7F is complete as a model-improvement sprint. It produces the strongest
same-contract GNN results so far: R3 as the best primary AUPRC/AUROC GNN, and
R2 as the best rare-negative validation-threshold operating point. The result
supports family-aware target-observation context encoding as the leading Graph C
GATv2 direction, while preserving the claim boundary that XGBoost F4 remains the
primary AUPRC bar and Sprint 8 robustness is required before stable superiority
claims.
