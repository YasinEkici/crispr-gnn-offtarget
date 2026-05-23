# Sprint 2 Feature Catalog

Feature matrices are generated from the locked Sprint 2 split rows.
Raw identifiers, genome labels, cell-line labels, and coordinate fields are not predictive features.
F4 computed nucleosome aggregates use missing values plus explicit missingness indicators; imputation is fit on train rows during model training.

## Feature Sets

### F1

Sequence and mismatch engineered numeric features.

Families: sequence_summary, mismatch_position

### F2

F1 plus binding-energy scalar features.

Families: sequence_summary, mismatch_position, binding_energy

### F3

F2 plus experimental epigenetic scalar features.

Families: sequence_summary, mismatch_position, binding_energy, experimental_epigenetic

### F4

F3 plus aggregated computed nucleosome features and missingness indicators.

Families: sequence_summary, mismatch_position, binding_energy, experimental_epigenetic, computed_nucleosome_aggregates, computed_nucleosome_missingness

## Summary

| feature_set | columns | rows | rows_with_missing | columns_with_missing | total_missing_values |
| --- | --- | --- | --- | --- | --- |
| F1 | 33 | 11446 | 0 | 0 | 0 |
| F2 | 38 | 11446 | 0 | 0 | 0 |
| F3 | 44 | 11446 | 0 | 0 | 0 |
| F4 | 135 | 11446 | 789 | 78 | 61542 |
