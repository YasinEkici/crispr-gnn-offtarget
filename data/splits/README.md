# Splits

Small split manifests may be committed here when they are safe and reproducible.

Do not commit large data tables or generated graph objects.

## Expected Guide-Size Caveat

Sprint 2 follows the pre-existing evaluation decision to use guide-level splitting so the same guide RNA cannot appear in both train and validation/test. The Sprint 1 audit already documented that guide sizes are highly uneven: some guides have only a few rows while a few guides have many rows.

As an expected consequence, one large guide can account for a substantial share of a split. The locked Sprint 2 split records this planned diagnostic as `largest_guide_share` in `outputs/splits/sprint2_split_summary.csv`.

Current locked split:

- train largest guide share: 27.6%
- validation largest guide share: 28.7%
- test largest guide share: 24.0%

This is the known tradeoff of leakage-safe guide-level evaluation on this dataset, not evidence of a split bug. Model reports should mention it when interpreting AUPRC and secondary metrics.
