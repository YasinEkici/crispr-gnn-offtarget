# Sprint 7F Family-Aware Target-Observation Context Encoder Report

Run batch: `sprint9_multiseed_S7F_R2_seed7`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Architecture policy: Graph C GATv2 only for newly trained runs; no new losses, samplers, graph schemas, topology changes, or hyperparameter search.
- Primary comparison base: `context_similar_to` edges dropped; candidate S5F2 remains active in GATv2 attention and the final edge classifier.
- Controlled question: whether a richer `target_observation` encoder improves the Sprint 7D no-context-edge Graph C GATv2 setting.
- Primary metric: AUPRC. MCC, specificity, TN/FP, and macro F1 are secondary operating-point diagnostics.
- Attention and target-encoder activation summaries are interpretation-only model artifacts, not biological causal evidence.

## Result Summary

| predeclared_run_id | graph_schema | architecture | target_context_encoder_type | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S7F_R2_family_aware_context_encoder | graph_c_context_observation | gatv2 | family_aware | 0.975002 | 0.922031 | 0.784050 | 0.593750 | 0.479290 | 81 | 88 | 19 | 1514 |

## Canonical Encoder Definitions

- `S7F_R1_unified_deep_context_encoder`: deeper unified MLP over all 212 target-observation columns.
- `S7F_R2_family_aware_context_encoder`: separate branches for target sequence, experimental epigenetic, computed nucleosome aggregates, and missingness indicators, followed by fusion.
- `S7F_R3_family_aware_experimental_emphasis`: same family-aware design with predeclared extra branch capacity assigned to the experimental epigenetic family.

## Interpretation Boundaries

- Compare 7F trained rows primarily against `S7F_REF_NO_CONTEXT_EDGE_GATV2`.
- AUPRC remains the primary metric; rare-negative gains in MCC/specificity must be reported as threshold diagnostics.
- A single-seed encoder comparison can support mechanism hypotheses, not statistical superiority claims.
- No architecture, threshold, loss, sampler, encoder, or rerun choice may be selected from test diagnostics.

## Artifact Index

Diagnostic tables:
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_threshold_metrics.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_deltas.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_audit.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_activation_contract_summary.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_attention_contract_summary.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_per_guide_score_summary.csv`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/diagnostics/target_context_encoder_score_deciles.csv`

Figures:
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_auprc_comparison.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_threshold_metrics.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_pr_curves.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_roc_curves.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_score_distributions.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_training_curves.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_attention_by_edge_kind.png`
- `outputs/sprint9/multiseed/S7F_R2_family_aware_context_encoder/seed_7/figures/target_context_encoder_branch_activation_norms.png`
