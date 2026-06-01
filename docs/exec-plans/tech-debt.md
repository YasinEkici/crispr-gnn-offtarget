# Tech Debt

Track deferred cleanup here once implementation begins.

---

## ~~Predictions missing genome and grna_target_id metadata~~ — FIXED 2026-05-31

`_prediction_records()` now extracts `grna_target_id` from `audit_sgrna_ids` and
`genome` from `audit_genome` (parsed from physical target node IDs) stored during
materialization. Per-genome and per-guide diagnostic tables are now generated
automatically when these columns are present.

---

## Graph schema prefix in diagnostic and report filenames

**Discovered:** Sprint 4 Graph A run (2026-05-30)
**Fix needed before:** Graph C Slice 5 implementation

Current code generates fixed filenames that do not include the graph schema:

```
outputs/sprint4/graph_a/diagnostics/gcn_graph_a_score_direction.csv
outputs/sprint4/graph_a/diagnostics/gcn_graph_a_fixed_threshold_metrics.csv
outputs/sprint4/graph_a/diagnostics/gcn_graph_a_score_deciles.csv
outputs/sprint4/graph_a/gcn_graph_a_report.md
```

When Graph C runs, these will overwrite the Graph A files.

Files that already include `graph_a` in their name (no change needed):
```
outputs/sprint4/graph_a/diagnostics/gcn_graph_a_predictions.csv
outputs/sprint4/graph_a/diagnostics/gcn_graph_a_training_history.csv
```

**Required fix:** Pass `graph_schema` into `write_gcn_diagnostics` and
`write_gcn_report` in `src/crispr_gnn/evaluation/diagnostics.py` and
`scripts/train.py` so filenames become:
```
gcn_graph_a_score_direction.csv
gcn_graph_a_fixed_threshold_metrics.csv
gcn_graph_a_score_deciles.csv
gcn_graph_a_report.md
gcn_graph_c_score_direction.csv
...
```

**Fixed 2026-05-31:** `write_gcn_diagnostics` and `write_gcn_plots` now accept
`schema_label` parameter that places files in `sprint4/{schema_label}/` subdir
with `gcn_{schema_label}_` prefix. `write_gcn_report` accepts `root` parameter
for repo-relative paths. `scripts/train.py` computes `schema_label` from
`graph_schema` and passes it throughout.

**Workaround for Graph A first run:** Files renamed manually before commit.
