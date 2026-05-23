# Splits

Small split manifests may be committed here when they are safe and reproducible.

Do not commit large data tables or generated graph objects.

## Sprint 2 Split Caveat

Sprint 2 uses guide-level splitting so the same guide RNA cannot appear in both train and validation/test. This prevents guide leakage, but the Mak 2022 main-clean dataset has uneven guide sizes: some guides have only a few rows while a few guides have many rows.

As a result, one large guide can account for a substantial share of a split. The locked Sprint 2 split records this as `largest_guide_share` in `outputs/splits/sprint2_split_summary.csv`.

Current locked split:

- train largest guide share: 27.6%
- validation largest guide share: 28.7%
- test largest guide share: 24.0%

This is a dataset-shape limitation, not evidence of leakage. Model reports should mention it when interpreting AUPRC and secondary metrics.
