# Label Schemes

This document defines the Sprint 1 label policy for the Mak et al. 2022 crisprSQL-derived dataset.

Generated report artifact:

- `outputs/reports/label_threshold_sensitivity.md`

## Available Label Column

The audited raw dataset contains `cleavage_freq`.

The dataset does not contain a transformed `CA` column or other CA-like transformed label column. Any Mak-style CA target must be reproduced later from raw values and study-level transformation logic.

## Scheme A - Primary Binary Label

Definition:

```text
label = 1 if cleavage_freq > 1e-5 else 0
```

Role:

- Primary binary classification target for Sprints 2-7.
- Paper-aligned negative boundary: Mak et al. set activity values below the assay accuracy threshold of `1e-5` to the clipped low CA value in their transformed target.
- Primary metric remains AUPRC under guide-level evaluation.

Full-dataset audit result:

| positives | negatives | positive_rate | imbalance |
| ---: | ---: | ---: | ---: |
| 21,365 | 288,777 | 0.068888 | 13.52:1 |

Measured-only audit result:

| measured rows | positives | negatives | positive_rate |
| ---: | ---: | ---: | ---: |
| 25,632 | 21,365 | 4,267 | 0.833528 |

## Scheme B - Later-Only Paper Comparison

Definition:

```text
Mak-style transformed CA target, then compare against CA > -4 or use regression metrics.
```

Status:

- Deferred.
- Not central to the project.
- Not directly computable from a stored dataset column because transformed `CA` is absent.
- Use only for a future paper-comparison/reproduction track.

Reproduction requirements:

- Recompute the paper's per-study Box-Cox transformation.
- Standardize transformed values to the paper's target distribution.
- Clip to `[-4, 4]`.
- Set values below assay accuracy `1e-5` to the low clipped value.
- Match the paper's dataset, target, split, metrics, and architecture before claiming reproduction.

This project must not claim Mak et al. reproduction from Scheme A binary AUPRC results.

## Scheme C - Robustness Sensitivity

Definition:

```text
label = 1 if cleavage_freq > 1e-3 else 0
```

Role:

- Later high-confidence robustness ablation.
- Not the default training target.
- Useful for testing sensitivity to stricter activity thresholds.

Full-dataset audit result:

| positives | negatives | positive_rate | imbalance |
| ---: | ---: | ---: | ---: |
| 8,280 | 301,862 | 0.026697 | 36.46:1 |

Measured-only audit result:

| measured rows | positives | negatives | positive_rate |
| ---: | ---: | ---: | ---: |
| 25,632 | 8,280 | 17,352 | 0.323034 |

## High Threshold Audit Sensitivity

Definition:

```text
label = 1 if cleavage_freq > 0.1 else 0
```

Role:

- Audit-only sensitivity check.
- Not a planned primary scheme.

Full-dataset audit result:

| positives | negatives | positive_rate | imbalance |
| ---: | ---: | ---: | ---: |
| 1,184 | 308,958 | 0.003818 | 260.94:1 |

## Scheme D - Reserved Regression Track

Definition:

```text
continuous target using cleavage_freq or a reproduced transformed CA value
```

Status:

- Reserved for future regression work if needed.
- Not part of Sprint 1 implementation.
- Requires a separate policy for transformation, missing labels, metrics, and paper-comparison validity.

## Outlier And Subset Policy

| case | audit count | policy |
| --- | ---: | --- |
| NaN `cleavage_freq` | 78 | Label-ineligible until a documented policy is approved; do not silently impute. |
| Negative `cleavage_freq` | 685 | Below-threshold for binary sensitivity counts; flag as raw-label quality issue. |
| `cleavage_freq > 1` | 298 | Positive for binary thresholds; do not clip for binary classification. |
| `measured=0` rows | 284,510 | Training-only optional noisy negatives; never test ground truth. |
| `experiment_id=18` | 14,108 missing `cell_line` rows | Keep out of main evaluation or report as separate no-cell-line sensitivity subset. |
| Non-`hg19` genomes | 65,540 | Do not drop by default; report per-genome breakdown and avoid human-only overclaims. |

## Split Implications

- Test rows must be `measured=1` only.
- Validation should prefer `measured=1`.
- Training may include `measured=0` rows only as optional putative negatives with a label-noise caveat.
- Report measured composition for every split.
- Random edge split is debug-only.
- Final evaluation requires guide-level split and AUPRC as the primary metric.
