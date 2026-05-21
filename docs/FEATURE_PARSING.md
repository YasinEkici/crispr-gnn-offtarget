# Feature Parsing

This document defines the Sprint 1 parser policy for Mak et al. 2022 computed nucleosome features.

Generated report artifact:

- `outputs/reports/feature_missingness.md`

Parser implementation:

- `src/crispr_gnn/data/parsers.py`

## Feature Groups

Experimental epigenetic scalar features:

- `epigen_ctcf`
- `epigen_dnase`
- `epigen_rrbs`
- `epigen_h3k4me3`
- `epigen_drip`
- `MNase`

Computed nucleosome features:

- `GCContent`
- `WSScore`
- `YRScore`
- `NucleotideBDM`
- `StrongWeakBDM`
- `NuPoP_Occup_147_human`
- `NuPoP_Viterbi_147_human`
- `NuPoP_Affinity_147_human`
- `nuCpos_Occup_147_yeast`
- `nuCpos_Viterbi_147_yeast`
- `nuCpos_Affinity_147_yeast`
- `VanDerHeijden`
- `LeNupH3Q85C`

Binding-energy scalar features:

- `energy_1`
- `energy_2`
- `energy_3`
- `energy_4`
- `energy_5`

## Computed Feature Format

The 13 computed nucleosome features are stored as string-formatted arrays. Valid values contain exactly 23 numeric positions.

Sprint 1 parser behavior:

- Parse valid 23-value numeric arrays.
- Treat missing values separately from malformed arrays.
- Reject malformed lengths.
- Reject non-numeric array entries.
- Do not silently pad, clip, coerce, or impute values.

## Audit Results

Feature group summary:

| group | features | rows missing any | missing pct | status |
| --- | ---: | ---: | ---: | --- |
| Experimental epigenetic | 6 | 0 | 0.0000% | PASS |
| Computed nucleosome | 13 | 15,153 | 4.8858% | PASS |
| Binding energy | 5 | 0 | 0.0000% | PASS |

For each computed nucleosome feature:

| value type | count |
| --- | ---: |
| Valid parsed rows | 294,989 |
| Missing rows | 15,153 |
| Malformed length rows | 0 |
| Non-numeric rows | 0 |

Computed feature missingness is shared across the 13 computed features and is mostly concentrated in `experiment_id=18`, which accounts for 13,780 rows missing at least one computed feature.

## Missingness Policy

Sprint 1 does not impute missing computed features.

Later feature builders must choose and document one of these policies before using computed nucleosome features:

- Exclude rows missing computed features for feature sets that require them.
- Add explicit missingness indicators and impute using train-only statistics.
- Use a feature-set ablation where computed features are omitted.
- Report a separate sensitivity subset for rows with missing computed/context features.

Any imputation must be fit on training data only.

## Dimensionality Strategies For Later Sprints

Position-resolved:

- Keep all 23 values for each of the 13 computed features.
- Expected computed feature dimensionality: `13 * 23 = 299`.
- With the 6 experimental scalar features, epigenetic/context dimensionality becomes 305 before adding binding-energy or sequence features.

Aggregated:

- Compute summary statistics such as mean, standard deviation, minimum, maximum, and selected quantiles.
- Lower-dimensional and easier for tabular baselines.
- Must be computed after parsing and after train/test split where normalization is involved.

PAM-focused or region-focused:

- Keep a defined subset of positions around biologically relevant sequence regions.
- The exact position convention must be documented before training.
- Do not change the selected region across models in the same comparison.

## Tests

Focused parser tests live in:

- `tests/test_feature_parsers.py`

Required test behavior:

- Valid 23-value arrays parse successfully.
- Missing values are handled separately from malformed arrays.
- Wrong-length arrays are rejected.
- Non-numeric arrays are rejected.
