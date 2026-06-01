# Sprint 4 GCN Graph A/C Comparison Report

## Contract

- Label scheme: `scheme_a`.
- Split ID: `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- Main evaluation remains measured-only with `experiment_id=18` excluded by the frozen Sprint 2 contract.
- Checkpoint and threshold selection use validation only; test diagnostics are interpretation-only.
- Required comparison baseline: `xgboost_unweighted / F4`.
- Test positive prevalence: `0.900705`.

## Result Summary

| Model | Graph schema | Feature set | Test AUPRC | Test AUROC | Test F1 | Test MCC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `xgboost_unweighted` | F4 tabular baseline | F4 | 0.992522 | 0.938416 | n/a | 0.345198 |
| `gcn_graph_a` | `graph_a_minimal_physical_target` | `S1_pair+F1` | 0.966287 | 0.745076 | 0.951845 | 0.300781 |
| `gcn_graph_c` | `graph_c_context_observation` | `CandidatePair+TargetObservationContext` | 0.961586 | 0.759886 | 0.958896 | 0.453738 |

## Interpretation

Graph A and Graph C are both validated same-contract GCN runs, but neither beats the frozen `xgboost_unweighted / F4` reference on primary test AUPRC. Graph A test AUPRC is `0.966287` and Graph C test AUPRC is `0.961586`, compared with F4 XGBoost test AUPRC `0.992522`.

Graph C must not be interpreted as a topology-only experiment. Relative to Graph A, it changes both topology through `context_similar_to` relations and target semantics through feature-bearing `target_observation` nodes instead of featureless shared `physical_target_site` nodes.

Graph C improves MCC relative to Graph A in this run (`0.453738` vs `0.300781`), but this is a secondary metric and does not override the primary AUPRC comparison. Any further changes based on these test diagnostics would violate the no-test-tuning contract.

## Artifact Index

- Consolidated results: `outputs/sprint4/gcn_sprint4_comparison_results.csv`
- AUPRC comparison figure: `outputs/sprint4/figures/gcn_sprint4_schema_auprc_comparison.png`
- PR curve comparison figure: `outputs/sprint4/figures/gcn_sprint4_pr_curves.png`
- Graph A report: `outputs/sprint4/graph_a/gcn_graph_a_report.md`
- Graph C report: `outputs/sprint4/graph_c/gcn_graph_c_report.md`
- Graph C provenance: `outputs/sprint4/graph_c/sprint4_graph_c_gcn_seed42_20260601/graph_artifact_provenance.json`

## Validation Notes

- Graph C provenance records `sgRNA=150`, `target_observation=11446`, `candidate_pair=11446`, and `context_similar_to=91754`.
- Required Graph A and Graph C diagnostic tables and figures are present in their schema-specific output directories.
- `model.pt` checkpoint files are run artifacts and must remain untracked.
