# Sprint 5B Graph C Energy-Sensitivity Artifact Report

Sprint 5B keeps Graph C topology and target-observation semantics fixed, then adds the Sprint 5 `S5F2_energy` candidate-edge feature table as a secondary sensitivity.

## Interpretation Boundary

- This is not the primary Sprint 5 feature ablation.
- Graph C already encodes context through target-observation nodes and context-similarity edges.
- The result may compare Graph C context representation with the best Graph A energy-focused candidate-edge setting, but must not be described as a clean feature-family isolation.

## Frozen Contract

- Graph schema: `graph_c_context_observation`.
- Label scheme: `scheme_a`.
- Split: `sprint2_main_seed42`.
- Universe: measured-only rows, with `experiment_id=18` excluded by the locked split assignment.
- Candidate edge feature set: `S5F2_energy`.

## Feature Tables

| feature_set | columns |
| --- | ---: |
| `candidate_pair_features` | 32 |
| `target_observation_features` | 212 |
| `S5F2_energy` | 268 |

## Notes

- `target_observation_features` and `context_similar_to` remain full Graph C context representations from the Sprint 3 schema.
- `S5F2_energy` contains guide/target sequence one-hot, sequence/mismatch features, and `energy_1`-`energy_5`; binding-energy features are not epigenetic features.

## Result Interpretation

- Sprint 5B `GraphCContext+S5F2_energy` achieved test AUPRC `0.972481`,
  improving the Sprint 4 Graph C AUPRC (`0.961586`) but not exceeding Sprint 5
  Graph A `S5F2_energy` (`0.976585`).
- MCC `0.274287` and macro F1 `0.552442` are lower because the
  validation-selected threshold yields TN/FP/FN/TP `14/155/0/1533`, recognizing
  few negatives while missing no positives.
- The artifact should therefore be interpreted as a secondary ranking
  sensitivity and as motivation for Sprint 6 imbalance/threshold/loss work, not
  as evidence to tune Graph C features or topology inside Sprint 5.
