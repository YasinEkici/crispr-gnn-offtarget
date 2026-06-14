# Sprint 8A Target-Context & Context-Edge Interaction Report

Run batch: `sprint9_multiseed_S8A_R2_seed123`

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
| S8A_R2_context_edge_film | graph_c_context_observation | gatv2 | family_aware_experimental_emphasis | False | film | 0.984490 | 0.893040 | 0.683553 | 0.373467 | 0.366864 | 62 | 107 | 63 | 1470 |

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
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_audit.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_threshold_metrics.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_deltas.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_branch_gate_summary.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_film_summary.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_parameter_counts.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_per_guide_score_summary.csv`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/diagnostics/target_context_interaction_score_deciles.csv`

Figures:
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_auprc_comparison.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_threshold_metrics.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_pr_curves.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_roc_curves.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_score_distributions.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_training_curves.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_gate_weights.png`
- `outputs/sprint9/multiseed/S8A_R2_context_edge_film/seed_123/figures/target_context_interaction_parameter_counts.png`
