# Feature Parsing

Sprint 0 skeleton. Sprint 1 will implement and document the parser for computed nucleosome features.

## Computed feature format

The 13 computed nucleosome features are expected to be string-formatted arrays with 23 positions.

## Planned strategies

- Position-resolved: keep all 23 values per feature.
- Aggregated: compute summary statistics such as mean, max, sum, and standard deviation.
- PAM-focused: keep a defined seed or PAM-proximal subset.

## Guardrails

- Validate array length.
- Preserve missing-value behavior.
- Add tests before relying on parsed features in training.
