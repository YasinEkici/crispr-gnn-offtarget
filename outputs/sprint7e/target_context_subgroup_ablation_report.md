# Sprint 7E Target-Observation Context Feature Subgroup Ablation Report

Run batch: `sprint7e_target_context_subgroup_seed42_20260607_213909`

## Executive Summary

Sprint 7E followed up the Sprint 7D Graph C GATv2 mechanism result. Sprint 7D showed that full Graph C GATv2 achieved the best rare-negative operating-point behavior so far, and that dropping context-similarity edges barely changed that behavior. Sprint 7E therefore fixed the no-context-edge Graph C GATv2 setting and asked a narrower question: which direct `target_observation` feature family carries the signal?

The answer is clear under the frozen single-seed contract: masking the six direct experimental epigenetic features collapses the model. `S7E_R2_mask_experimental_epigenetic` falls to test AUPRC `0.885321`, MCC `-0.011388`, specificity `0.000000`, and TN/FP/FN/TP `0/169/2/1531`. Masking all nonsequence context reproduces the same collapse (`S7E_R5`, AUPRC `0.890660`, MCC `-0.011388`, TN `0`). By contrast, masking target sequence, computed nucleosome aggregates, or computed missingness keeps most of the Graph C GATv2 rare-negative behavior intact.

This supports a mechanism-level interpretation, not a final superiority claim: within the frozen Sprint 7E Graph C/no-context-edge GATv2 setting, direct experimental epigenetic target-observation features are necessary to reproduce the observed rare-negative operating point. The signal may reflect biological epigenetic context, assay/source/cell-line structure, or both. Sprint 7E does not establish biological causality.

## Frozen Contract

- Label/split/evaluation: Scheme A, `sprint2_main_seed42`, guide-disjoint split, measured-only headline rows, experiment `18` excluded.
- Checkpoint/threshold policy: validation-only checkpoint on `val_auprc`; validation-only threshold via `validation_max_f1`; no test tuning.
- Feature policy: candidate-edge `S5F2_energy` fixed at 268 columns; Graph C `target_observation_features` fixed at 212 columns.
- Loss policy: Sprint 6 winner `weighted_bce` with `pos_weight: auto` (`~0.1267` for the frozen measured-only train split).
- Architecture policy: GATv2 only for newly trained Sprint 7E rows; no new losses, samplers, sequence encoders, graph schemas, or hyperparameter search.
- Attention policy: candidate S5F2 edge features remain active in GATv2 attention via `edge_attr` / `edge_dim` and in the final classifier; `context_similar_to` edges are dropped in the primary 7E ablation base.
- Primary metric: AUPRC. MCC, specificity, TN/FP, macro F1, and thresholded confusion counts are operating-point diagnostics.
- Reference bar: `xgboost_unweighted / F4` remains the authoritative matched-contract AUPRC bar (`0.992522`).

## Run Matrix

| Run | Role | Controlled Change |
| --- | --- | --- |
| `S7E_REF_XGB_F4` | carry-forward reference | Sprint 2 XGBoost F4 matched-contract reference |
| `S7E_REF_GRAPH_A_GCN` | carry-forward reference | Sprint 6 best Graph A GCN, weighted BCE |
| `S7E_REF_GRAPH_C_GCN` | carry-forward reference | Sprint 5B Graph C GCN |
| `S7E_REF_FULL_GRAPH_C_GATV2` | carry-forward reference | Sprint 7B full Graph C GATv2 |
| `S7E_REF_NO_CONTEXT_EDGE_GATV2` | primary 7E baseline | Sprint 7D Graph C GATv2 with context-similarity edges dropped |
| `S7E_R1_mask_target_sequence` | new 7E run | mask 115 target sequence one-hot columns |
| `S7E_R2_mask_experimental_epigenetic` | new 7E run | mask 6 direct experimental epigenetic columns |
| `S7E_R3_mask_computed_nucleosome_aggregates` | new 7E run | mask 78 computed nucleosome aggregate columns |
| `S7E_R4_mask_computed_nucleosome_missingness` | new 7E run | mask 13 computed nucleosome missingness columns |
| `S7E_R5_mask_all_nonsequence_context` | new 7E run | mask all 97 nonsequence context columns |

## Result Table

| predeclared_run_id | test_auprc | test_auroc | test_macro_f1 | test_mcc | specificity | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S7E_REF_XGB_F4` | 0.992522 | 0.938416 |  | 0.345198 |  | 38 | 131 | 21 | 1512 |
| `S7E_REF_GRAPH_A_GCN` | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 |
| `S7E_REF_GRAPH_C_GCN` | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 0.082840 | 14 | 155 | 0 | 1533 |
| `S7E_REF_FULL_GRAPH_C_GATV2` | 0.969078 | 0.849705 | 0.739526 | 0.531774 | 0.372781 | 63 | 106 | 12 | 1521 |
| `S7E_REF_NO_CONTEXT_EDGE_GATV2` | 0.965598 | 0.850137 | 0.733910 | 0.517970 | 0.366864 | 62 | 107 | 14 | 1519 |
| `S7E_R1_mask_target_sequence` | 0.970151 | 0.900134 | 0.715395 | 0.489597 | 0.331361 | 56 | 113 | 13 | 1520 |
| `S7E_R2_mask_experimental_epigenetic` | 0.885321 | 0.350896 | 0.473554 | -0.011388 | 0.000000 | 0 | 169 | 2 | 1531 |
| `S7E_R3_mask_computed_nucleosome_aggregates` | 0.955915 | 0.728243 | 0.716753 | 0.511808 | 0.319527 | 54 | 115 | 6 | 1527 |
| `S7E_R4_mask_computed_nucleosome_missingness` | 0.949947 | 0.746975 | 0.665051 | 0.447562 | 0.230769 | 39 | 130 | 2 | 1531 |
| `S7E_R5_mask_all_nonsequence_context` | 0.890660 | 0.389521 | 0.473554 | -0.011388 | 0.000000 | 0 | 169 | 2 | 1531 |

## Deltas Against 7E Primary Baseline

Primary 7E baseline: `S7E_REF_NO_CONTEXT_EDGE_GATV2`, test AUPRC `0.965598`, MCC `0.517970`, specificity `0.366864`, TN/FP `62/107`.

| Run | Masked Family | Delta AUPRC | Delta MCC | Test Specificity | TN/FP |
| --- | --- | ---: | ---: | ---: | ---: |
| `S7E_R1_mask_target_sequence` | target sequence | +0.004553 | -0.028373 | 0.331361 | 56/113 |
| `S7E_R2_mask_experimental_epigenetic` | experimental epigenetic | -0.080277 | -0.529358 | 0.000000 | 0/169 |
| `S7E_R3_mask_computed_nucleosome_aggregates` | computed aggregates | -0.009683 | -0.006162 | 0.319527 | 54/115 |
| `S7E_R4_mask_computed_nucleosome_missingness` | computed missingness | -0.015651 | -0.070408 | 0.230769 | 39/130 |
| `S7E_R5_mask_all_nonsequence_context` | all nonsequence context | -0.074938 | -0.529358 | 0.000000 | 0/169 |

The critical comparison is not the small AUPRC variation among retained-signal runs. It is the qualitative threshold collapse when experimental epigenetic features are removed. The collapse appears on both ranking and operating-point metrics: AUROC drops to `0.350896` for `S7E_R2`, and the validation-selected threshold predicts no test negatives.

## Validation Consistency

The collapse is not a test-only artifact from threshold selection. `S7E_R2_mask_experimental_epigenetic` also has validation AUPRC `0.890900`, validation MCC `0.000000`, validation specificity `0.000000`, and validation TN/FP/FN/TP `0/223/0/1511`. `S7E_R5_mask_all_nonsequence_context` similarly has validation AUPRC `0.865452`, validation MCC `0.000000`, validation specificity `0.000000`, and validation TN/FP/FN/TP `0/223/0/1511`.

The retained-signal runs behave differently. `S7E_R3_mask_computed_nucleosome_aggregates` reaches validation AUPRC `0.978973` and test MCC `0.511808`; `S7E_R4_mask_computed_nucleosome_missingness` reaches validation AUPRC `0.979582` and test MCC `0.447562`; `S7E_R1_mask_target_sequence` keeps test AUPRC above the no-context-edge baseline but loses some MCC.

## Mask And Attention Contract Audit

The mask audit confirms the intended controlled changes:

| Run | Masked Columns | Masked Feature Abs Sum After Mask | Context Edges Used |
| --- | ---: | ---: | ---: |
| `S7E_R1_mask_target_sequence` | 115 | 0.0 | 0 |
| `S7E_R2_mask_experimental_epigenetic` | 6 | 0.0 | 0 |
| `S7E_R3_mask_computed_nucleosome_aggregates` | 78 | 0.0 | 0 |
| `S7E_R4_mask_computed_nucleosome_missingness` | 13 | 0.0 | 0 |
| `S7E_R5_mask_all_nonsequence_context` | 97 | 0.0 | 0 |

Candidate S5F2 edge attributes remain nonzero in attention and in the classifier audit for the trained rows. This preserves the Sprint 7 design premise: Graph C GATv2 is not merely swapping `GCNConv` for `GATv2Conv`; candidate-edge energy features are consumed by the attention/message-passing path.

## Context Feature Profiling

The Graph C `target_observation_features` table contains 212 columns:

| Family | Columns | Missing Values After Preprocessing | Nonzero Fraction |
| --- | ---: | ---: | ---: |
| target sequence one-hot | 115 | 0 | 0.200000 |
| experimental epigenetic | 6 | 0 | 1.000000 |
| computed nucleosome aggregates | 78 | 0 | 0.974359 |
| computed nucleosome missingness | 13 | 0 | 1.000000 |

Sprint 7E added a small per-feature experimental audit under `outputs/sprint7e/context_feature_profiling/`. It computes split/label distributions and positive-minus-negative standardized mean differences for the six direct experimental features: `epigen_ctcf`, `epigen_dnase`, `epigen_rrbs`, `epigen_h3k4me3`, `epigen_drip`, and `MNase`.

Largest absolute SMD values:

| Split | Feature | SMD |
| --- | --- | ---: |
| train | `MNase` | 0.284557 |
| train | `epigen_h3k4me3` | 0.227911 |
| train | `epigen_drip` | 0.185506 |
| val | `MNase` | 0.862604 |
| val | `epigen_drip` | 0.369752 |
| val | `epigen_dnase` | 0.206072 |
| test | `MNase` | 0.813250 |
| test | `epigen_h3k4me3` | 0.303893 |
| test | `epigen_drip` | 0.247477 |

This audit is consistent with the ablation result: the six experimental epigenetic features contain direct positive-negative separation, with `MNase` especially prominent in validation and test. It also sharpens the claim boundary. The result can reflect biological context, source/assay/cell-line structure, or both. Without metadata joins or per-source controls, Sprint 7E should not claim that `MNase` or any epigenetic feature is causal biological evidence.

## Literature Framing

The Sprint 7/7E architecture line remains aligned with the graph-learning framing from Kipf and Welling's GCN work, Veličković et al.'s Graph Attention Networks, and Brody et al.'s GATv2 dynamic-attention critique. The relevant CRISPR-GNN literature framing remains graph-based CRISPR prediction work such as Vinodkumar et al. 2021 and Liu et al. 2025 GraphCRISPR-style efficiency prediction notes in the repository.

Sprint 7E does not claim to reproduce any of those papers. It adapts the attention idea to this repository's frozen measured-only off-target evaluation contract and tests a project-specific mechanism: whether Graph C target-observation context, especially direct experimental epigenetic context, explains the rare-negative improvement observed in Sprint 7D.

## Interpretation

1. Graph C GATv2's Sprint 7D rare-negative behavior is mostly not explained by context-similarity edges. The no-context-edge reference remains close to full Graph C GATv2.
2. The direct target-observation feature vector is the important channel.
3. Within that vector, the direct experimental epigenetic family is the dominant necessary subgroup. Removing it collapses AUPRC, AUROC, specificity, MCC, and validation/test negative recognition.
4. Computed nucleosome aggregates and missingness contribute, but they do not explain the main Sprint 7D gain on their own.
5. Target sequence one-hot is not the main explanation for the rare-negative operating-point improvement in this setting.

## Claim Boundaries

- Sprint 7E is single-seed and fixed-split. It supports a mechanism hypothesis, not a final statistical superiority claim.
- XGBoost F4 remains the AUPRC bar. Sprint 7E does not beat `0.992522` AUPRC.
- MCC/specificity improvements are threshold diagnostics and must not be presented as AUPRC gains.
- Attention summaries are model-interpretation artifacts only, not biological causal evidence.
- The per-feature experimental audit is a small confound sanity check, not a full source/cell-line/assay leakage audit.
- Multi-seed consolidation, paired uncertainty, source/cell-line subgroup checks, and metadata-enriched negative analysis remain robustness/follow-up work.

## Artifact Index

Primary outputs:

- `outputs/sprint7e/target_context_subgroup_ablation.csv`
- `outputs/sprint7e/target_context_subgroup_ablation_report.md`
- `outputs/sprint7e/target_context_subgroup_ablation_run_manifest.json`
- `outputs/sprint7e/graph_artifact_provenance.json`

Diagnostics:

- `outputs/sprint7e/diagnostics/target_context_subgroup_threshold_metrics.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_deltas.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_mask_audit.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_attention_contract_summary.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_attention_summary.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_predictions.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_per_guide_score_summary.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_score_deciles.csv`
- `outputs/sprint7e/diagnostics/target_context_subgroup_training_history.csv`

Context profiling and audit:

- `outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_family_map.csv`
- `outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_group_summary.csv`
- `outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_distribution_by_split_label.csv`
- `outputs/sprint7e/context_feature_profiling/sprint7e_experimental_epigenetic_feature_distribution_by_split_label.csv`
- `outputs/sprint7e/context_feature_profiling/sprint7e_experimental_epigenetic_feature_smd_by_split.csv`
- `outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_profile_report.md`
- `outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_profile_manifest.json`

Figures:

- `outputs/sprint7e/figures/target_context_subgroup_auprc_comparison.png`
- `outputs/sprint7e/figures/target_context_subgroup_threshold_metrics.png`
- `outputs/sprint7e/figures/target_context_subgroup_pr_curves.png`
- `outputs/sprint7e/figures/target_context_subgroup_roc_curves.png`
- `outputs/sprint7e/figures/target_context_subgroup_score_distributions.png`
- `outputs/sprint7e/figures/target_context_subgroup_training_curves.png`
- `outputs/sprint7e/figures/target_context_subgroup_attention_by_edge_kind.png`
- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_missingness.png`
- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_distribution.png`
- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_experimental_epigenetic_smd_by_split.png`

## Verdict

Sprint 7E is complete as a mechanism-oriented diagnostic sprint. It identifies direct experimental epigenetic target-observation features as the necessary subgroup behind the Graph C GATv2 rare-negative operating-point behavior under the frozen single-seed contract. The result is strong enough to guide the next modeling step, provided the report keeps the claim boundary explicit: this is necessary-feature evidence under a fixed evaluation contract, not biological causality and not final robustness.
