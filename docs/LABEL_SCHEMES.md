# Label Schemes

Sprint 0 skeleton. Sprint 1 will validate thresholds against the dataset.

## Scheme A

Primary candidate: `cleavage_freq > 1e-5`.

## Scheme B

Paper comparison track: Box-Cox transformed `CA > -4`.

## Scheme C

High-confidence ablation: `cleavage_freq > 1e-3`, with mid-range handling documented before use.

## Scheme D

Regression track on `cleavage_freq` or transformed `CA`.

## Outlier handling to decide

- NaN `cleavage_freq`.
- Negative `cleavage_freq`.
- Values above 1.
