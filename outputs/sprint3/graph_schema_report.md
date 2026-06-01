# Sprint 3 Graph Schema Report

Sprint 3 constructs graph datasets and leakage audits only. No GNN model performance is reported here.

## Preserved Evaluation Contract

- Locked split: `sprint2_main_seed42`.
- Label: Scheme A, `cleavage_freq > 1e-5`.
- Main graph universe is measured-only and excludes `experiment_id=18`.
- Primary downstream comparison remains `xgboost_unweighted / F4` (test AUPRC `0.992522`).
- Primary visibility policy: strict-inductive; held-out similarity is computed only against training references.

## Candidate Edge Summary

| split | rows | positives | negatives | positive_rate |
| --- | --- | --- | --- | --- |
| train | 8010 | 7109 | 901 | 0.887516 |
| val | 1734 | 1511 | 223 | 0.871396 |
| test | 1702 | 1533 | 169 | 0.900705 |

## Graph Artifacts

| graph | node tables | relation tables | feature placement | role |
| --- | --- | --- | --- | --- |
| `graph_a_minimal_physical_target` | sgRNA=150, physical_target_site=9880 | candidate_pair=11446 | candidate_pair_edge | primary_schema |
| `graph_b_guide_similarity_control` | sgRNA=150, physical_target_site=9880 | candidate_pair=11446, sequence_similar_to=1208 | graph_a_minimal_physical_target | bounded_secondary_control |
| `graph_c_context_observation` | sgRNA=150, target_observation=11446 | candidate_pair=11446, context_similar_to=91754 | target_observation_node | primary_schema |

## Schema Decisions

- Graph A represents genome-aware physical target loci; row-varying sequence/context is carried by candidate-edge feature tables.
- Graph B inherits Graph A and adds guide-sequence similarity as a bounded control relation.
- Graph C represents target observations keyed by source row `id`; context similarity is based on train-fitted F3/F4 context only.
- Graph C is not merely Graph A with additional edges: the target semantics intentionally differ.
- Large feature tables are generated under ignored `data/processed/graphs/sprint3/`; this report is the tracked Sprint 3 artifact.

## Leakage Controls

- Candidate labels are recomputed and validated against Scheme A.
- Candidate edge membership exactly preserves the locked split assignment.
- Candidate relations are stored with split labels; a later training loader must expose only `split=train` supervised edges during fitting.
- Graph A physical target nodes contain metadata identity only, not variable F3/F4 context.
- Graph B similarity uses guide sequence only.
- Graph C context preprocessing is fitted on training observations only; validation/test observations connect only to training references.
- Auxiliary relation rows are stored as visibility fragments: a validation/test view is the union of its rows with training-view rows.
- No test metric or label is used for graph construction or schema selection.

## Sprint 4 Handoff

Sprint 4 may load these typed artifacts into PyTorch Geometric `HeteroData` after an explicit dependency decision. Train Graph A first, evaluate the primary context-enriched Graph C second, and use Graph B only as a bounded control run. Every downstream comparison must preserve the locked protocol and compare to `xgboost_unweighted / F4`.
