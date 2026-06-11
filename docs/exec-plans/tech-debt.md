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

---

## Sprint 8A interaction-mode parameter count includes inactive classifier

**Discovered:** Sprint 8A Slice 6 forensic validation (2026-06-11)
**Fix needed before:** Sprint 9 robustness reporting, or any parameter-budget
claim comparing Sprint 8A interaction runs.

In `GraphCEdgeGATv2`, the base `edge_classifier` is instantiated for all Graph C
GATv2 models. When `context_edge_interaction != "none"`, forward dispatch uses
`_classify_with_interaction()` and `interaction_edge_classifier` instead of the
base `edge_classifier`. The current `parameter_count` therefore includes inactive
base-classifier parameters for R2/R3.

Observed consequence in Sprint 8A:

- R0 reported params: `274153`.
- R2 reported params: `381866`.
- R3 reported params: `386126`.
- The nominal R2/R3 increase overstates active capacity by roughly the unused
  base edge-classifier size (~100k parameters). Predictions are unaffected, but
  capacity interpretation and Dwivedi-style parameter-budget discussion are
  misleading unless active and nominal counts are separated.

Required fix:

- Add an `active_parameter_count` diagnostic for models with conditional forward
  paths, or avoid instantiating inactive classifier modules when interaction mode
  is enabled if state-dict compatibility is not required.
- Keep historical Sprint 8A `parameter_count` as nominal unless the result files
  are explicitly regenerated under a new, documented reporting version.

---

## Sprint 5B output rename / working-tree integrity issue

**Discovered:** Sprint 8A Slice 6 validation workspace check (2026-06-11)
**Fix needed before:** next commit that includes returned outputs.

`git status` showed tracked Sprint 5B files under `outputs/sprint5b/graph_c/`
as deleted while a new untracked `outputs/sprint5b/graph_c_context_observation/`
directory exists. This appears to be an output-sync or rename side effect, not a
Sprint 8A modeling result.

Required fix:

- Before committing returned Sprint 8A artifacts, decide whether Sprint 5B output
  paths should remain as tracked historical `outputs/sprint5b/graph_c/` or be
  intentionally renamed.
- If the rename is not intentional, restore the tracked `outputs/sprint5b/graph_c/`
  files and remove or ignore the stray untracked copy.
- Do not mix this cleanup with model-result interpretation.
