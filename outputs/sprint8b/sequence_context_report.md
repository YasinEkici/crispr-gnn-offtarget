# Sprint 8B Sequence-Context Encoder Report

Run batch: `sprint8b_sequence_context_seed42_20260612_220002`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold; no test-driven selection.
- Run matrix: `S8B_R0_reference` is the Sprint 8A R2 carry-forward row; only `S8B_R1_sequence_only` and `S8B_R2_sequence_plus_context` are trained.
- Sequence policy: S1 guide/target pair reconstructed from Graph C one-hot node features; no raw-data join and no energy/epigenetic/context leakage into the sequence branch.
- Architecture policy: CRISPR-Net/CRISPR-IP-adapted Conv+BiLSTM, trained from scratch; no externally pretrained CRISPR/genomic weights; no reproduction claim.
- Selection: validation AUPRC primary. Test AUPRC is the primary reported test metric; MCC/specificity/macro F1 are secondary threshold diagnostics.
- Parameter reporting: nominal and active parameter counts are reported separately because Sprint 8A interaction mode can instantiate inactive classifier parameters.

## Result Summary

| predeclared_run_id | sequence_context_mode | context_edge_interaction | test_auprc | test_auroc | test_macro_f1 | test_mcc | test_specificity | test_tn | test_fp | test_fn | test_tp | parameter_count | active_parameter_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S8B_R0_reference | reference_no_sequence_encoder | film | 0.982757 | 0.910575 | 0.777185 | 0.563656 | 0.520710 | 88 | 81 | 39 | 1494 | 381866 |  |
| S8B_R1_sequence_only | sequence_only | none | 0.856666 | 0.304164 | 0.472902 | -0.019749 | 0.000000 | 0 | 169 | 6 | 1527 | 30593 | 30593 |
| S8B_R2_sequence_plus_context | late_fusion | film | 0.986020 | 0.903480 | 0.760058 | 0.567309 | 0.863905 | 146 | 23 | 180 | 1353 | 412202 | 312105 |

## Interpretation Boundaries

- `S8B_R1_sequence_only` tests pure sequence-context value under the 8A harness.
- `S8B_R2_sequence_plus_context` tests whether S1 sequence context adds over the Sprint 8A R2 target-context + FiLM candidate.
- A single seed supports directional mechanism evidence only; robustness is Sprint 9.
- No architecture, threshold, loss, encoder, or rerun choice may be selected from test diagnostics.

## Artifact Index

Diagnostic tables:
- `outputs/sprint8b/diagnostics/sequence_context_threshold_metrics.csv`
- `outputs/sprint8b/diagnostics/sequence_context_deltas.csv`
- `outputs/sprint8b/diagnostics/sequence_context_sequence_input_audit.csv`
- `outputs/sprint8b/diagnostics/sequence_context_parameter_counts.csv`
- `outputs/sprint8b/diagnostics/sequence_context_attention_summary.csv`
- `outputs/sprint8b/diagnostics/sequence_context_target_context_encoder_summary.csv`
- `outputs/sprint8b/diagnostics/sequence_context_film_summary.csv`
- `outputs/sprint8b/diagnostics/sequence_context_per_guide_score_summary.csv`
- `outputs/sprint8b/diagnostics/sequence_context_score_deciles.csv`

Figures:
- `outputs/sprint8b/figures/sequence_context_auprc_comparison.png`
- `outputs/sprint8b/figures/sequence_context_threshold_metrics.png`
- `outputs/sprint8b/figures/sequence_context_pr_curves.png`
- `outputs/sprint8b/figures/sequence_context_roc_curves.png`
- `outputs/sprint8b/figures/sequence_context_score_distributions.png`
- `outputs/sprint8b/figures/sequence_context_training_curves.png`
- `outputs/sprint8b/figures/sequence_context_parameter_counts.png`
