# Sprint 7B Graph B S5F2 Artifact Report

Sprint 7B keeps Graph B topology fixed and attaches the Sprint 5 `S5F2_energy` candidate-edge feature table for a matched GCN-vs-GATv2 comparison.

## Frozen Contract

- Graph schema: `graph_b_guide_similarity_control`.
- Label scheme: `scheme_a`.
- Split: `sprint2_main_seed42`.
- Universe: measured-only rows, with `experiment_id=18` excluded by the locked split assignment.
- Candidate edge feature set: `S5F2_energy`.
- Auxiliary relation: `sequence_similar_to` remains label-free topology.

## Feature Tables

| feature_set | columns |
| --- | ---: |
| `S5F2_energy` | 268 |

## Sprint 7B Attention Policy

- Candidate edges use `S5F2_energy` in both the edge classifier and GATv2 `edge_attr`/`edge_dim` message-passing path.
- Reverse candidate edges duplicate the candidate edge features.
- `sequence_similar_to` edges are topology-only and use zero edge_attr vectors in GATv2.
- Attention summaries are interpretation-only model artifacts, not biological causal evidence.
