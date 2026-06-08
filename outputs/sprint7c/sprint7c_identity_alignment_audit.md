# Sprint 7C Identity / Alignment Audit

Graph C GCN to Graph C GATv2 test transition gate: `PASSED`

## Scope

This audit checks prediction-row comparability. Passing this audit means the
loaded prediction rows align on `row_index`, `grna_target_id`, and `label`.
It does not prove that `row_index` is a stable raw source-row ID for metadata
profiling; that remains a separate metadata-join gate.

## Summary

- Audit rows: `42`
- Pair-alignment failures: `0`
- Graph C transition analysis allowed: `True`
- Per-genome claims from Graph C prediction CSVs remain blocked if `genome` is
  missing; use a verified metadata join before making those claims.

## Artifact

- `outputs/sprint7c/diagnostics/sprint7c_prediction_alignment_audit.csv`
