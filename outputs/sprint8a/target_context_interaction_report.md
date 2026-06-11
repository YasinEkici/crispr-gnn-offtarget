# Sprint 8A Target-Context & Context-Edge Interaction Report

Run batch: `sprint8a_target_context_interaction_seed42_20260611_011416`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold; no test-driven selection.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Architecture policy: Graph C GATv2 only; `context_similar_to` edges dropped; candidate S5F2 active in GATv2 attention and the classifier. The frozen GATv2 attention/message passing is unchanged; the context-edge interaction is head-only.
- Canonical base: Sprint 7F R3 family-aware experimental-emphasis encoder (`S8A_REF_S7F_R3`).
- Selection: validation AUPRC primary (validation MCC / macro F1 tie-break). Test AUPRC is the primary reported test metric; MCC/specificity/macro F1 are secondary threshold diagnostics.
- Family-gate weights, FiLM scale/shift, attention summaries, and parameter counts are interpretation-only model artifacts, not biological causal evidence.

## Result Summary

| predeclared_run_id | graph_schema | architecture | target_context_encoder_type | family_gate | context_edge_interaction | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S8A_REF_XGB_F4 | tabular_reference | xgboost | nan |  |  | 0.992522 | 0.938416 | 0.642700 | 0.345198 | nan | 38 | 131 | 21 | 1512 |
| S8A_REF_GRAPH_A_GCN | graph_a_minimal_physical_target | gcn | nan |  |  | 0.976935 | 0.819972 | 0.698939 | 0.483719 | 0.289941 | 49 | 120 | 6 | 1527 |
| S8A_REF_GRAPH_C_GCN | graph_c_context_observation | gcn | nan |  |  | 0.972481 | 0.836219 | 0.552442 | 0.274287 | 0.082840 | 14 | 155 | 0 | 1533 |
| S8A_REF_FULL_GRAPH_C_GATV2 | graph_c_context_observation | gatv2 | nan |  |  | 0.969078 | 0.849705 | 0.739526 | 0.531774 | 0.372781 | 63 | 106 | 12 | 1521 |
| S8A_REF_NO_CTX_EDGE_GATV2 | graph_c_context_observation | gatv2 | nan |  |  | 0.965598 | 0.850137 | 0.733910 | 0.517970 | 0.366864 | 62 | 107 | 14 | 1519 |
| S8A_REF_S7F_R2 | graph_c_context_observation | gatv2 | family_aware |  |  | 0.982062 | 0.906557 | 0.801716 | 0.603489 | 0.650888 | 110 | 59 | 63 | 1470 |
| S8A_REF_S7F_R3 | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis |  |  | 0.984945 | 0.926551 | 0.777185 | 0.568108 | 0.497041 | 84 | 85 | 31 | 1502 |
| S8A_R0_base_reference | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | False | none | 0.975596 | 0.913840 | 0.613906 | 0.366171 | 0.159763 | 27 | 142 | 2 | 1531 |
| S8A_R1_family_gated_v2 | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | True | none | 0.976955 | 0.867757 | 0.680650 | 0.384165 | 0.319527 | 54 | 115 | 40 | 1493 |
| S8A_R2_context_edge_film | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | False | film | 0.982757 | 0.910575 | 0.777992 | 0.563656 | 0.520710 | 88 | 81 | 39 | 1494 |
| S8A_R3_gated_plus_film | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | True | film | 0.979899 | 0.885077 | 0.630384 | 0.303431 | 0.218935 | 37 | 132 | 31 | 1502 |
| S8A_R4_regularized_exp_branch | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | False | none | 0.975687 | 0.825673 | 0.694583 | 0.389211 | 0.443787 | 75 | 94 | 90 | 1443 |

## Canonical Run Definitions

- `S8A_R0_base_reference`: Sprint 7F R3 encoder, no gate, no interaction (reproduces the base path).
- `S8A_R1_family_gated_v2`: + SENET-style learned family gate over the four target-context branches.
- `S8A_R2_context_edge_film`: + head-only FiLM interaction between the target-context embedding and S5F2 edge features.
- `S8A_R3_gated_plus_film`: family gate + FiLM (the R1+R2 combination).
- `S8A_R4_regularized_exp_branch`: + bottleneck and feature-dropout on the experimental-epigenetic branch.

## Interpretation Boundaries

- Compare trained rows primarily against `S8A_REF_S7F_R3` (the base) on validation AUPRC; report all runs.
- AUPRC remains the primary metric; rare-negative gains in MCC/specificity are threshold diagnostics, not AUPRC gains.
- Report `parameter_count` next to performance; a gain that tracks added parameters is interpreted cautiously (capacity vs structure).
- A single-seed comparison supports mechanism hypotheses, not statistical superiority; robustness is Sprint 9.
- No architecture, threshold, loss, encoder, interaction, or rerun choice may be selected from test diagnostics.

## Artifact Index

Diagnostic tables:
- `outputs/sprint8a/diagnostics/target_context_interaction_audit.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_threshold_metrics.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_deltas.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_branch_gate_summary.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_film_summary.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_parameter_counts.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_per_guide_score_summary.csv`
- `outputs/sprint8a/diagnostics/target_context_interaction_score_deciles.csv`

Figures:
- `outputs/sprint8a/figures/target_context_interaction_auprc_comparison.png`
- `outputs/sprint8a/figures/target_context_interaction_threshold_metrics.png`
- `outputs/sprint8a/figures/target_context_interaction_pr_curves.png`
- `outputs/sprint8a/figures/target_context_interaction_roc_curves.png`
- `outputs/sprint8a/figures/target_context_interaction_score_distributions.png`
- `outputs/sprint8a/figures/target_context_interaction_training_curves.png`
- `outputs/sprint8a/figures/target_context_interaction_gate_weights.png`
- `outputs/sprint8a/figures/target_context_interaction_parameter_counts.png`
