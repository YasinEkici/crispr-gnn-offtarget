# Execution Plan: Sprint 7C Graph C GATv2 Explanation Audit

> Status: COMPLETE / FROZEN (2026-06-07). Sprint 7C analysis script and contract
> tests are in place, local source artifacts were verified, and
> `outputs/sprint7c/` was regenerated locally. Sprint 7C is an analysis and
> evidence-audit sprint, not a new model-training sprint. It explains the
> observed Sprint 7B Graph C GATv2 result as far as existing artifacts and
> verified metadata joins allow.

## 1. Goal

Sprint 7C asks why the Sprint 7B Graph C GATv2 run produced the strongest
same-contract GNN operating-point negative-class profile so far:

```text
S7B_R3_graph_c_gatv2_s5f2
test AUPRC      = 0.969078
test AUROC      = 0.849705
test macro F1   = 0.739526
test MCC        = 0.531774
test specificity= 0.372781
TN/FP/FN/TP     = 63/106/12/1521
```

Sprint 7C will first prove that row-level comparisons are valid, then analyze
score separation, threshold behavior, error transitions, per-guide
concentration, and aggregate attention diagnostics. It will distinguish strict
CSV-only analyses from analyses that require source-row metadata joins or
checkpoint-backed attention extraction.

Scientific question:

> Under the frozen Scheme A, guide-disjoint, measured-only protocol, did Graph C
> GATv2 improve rare-negative recognition because it changed score separation
> and validation-threshold transfer relative to Graph C GCN / Graph A references,
> and can that behavior be documented without overclaiming causality?

Sprint 7C will:

- Preserve the Sprint 2/3/4/5/6/7/7B evaluation contract.
- Perform an identity/alignment audit before row-level comparisons.
- Recompute AUPRC/AUROC and validation-selected threshold metrics from
  prediction CSVs wherever possible.
- Compare Graph C GCN vs Graph C GATv2 at the row level only after alignment
  passes.
- Separate CSV-only diagnostics, metadata-enriched diagnostics, and
  checkpoint-backed diagnostics in the report.
- Produce report-ready tables and figures under `outputs/sprint7c/`.
- Frame attention weights as model-interpretation signals only.

Sprint 7C will not:

- Train any model.
- Re-run Sprint 7B model fitting.
- Change model architecture, losses, samplers, thresholds, feature sets, graph
  artifacts, split artifacts, or labels.
- Select a new model, graph schema, threshold, or hyperparameter from test
  diagnostics.
- Claim biological causality from attention weights, context features, or
  recovered-negative profiles.
- Claim Graph C is topology-only. Graph C changes both topology and target-node
  semantics/context representation.
- Treat Sprint 7C as robustness evidence. Multi-seed, paired differences, and
  guide-level bootstrap CIs remain Sprint 8 scope.

## 2. Frozen Evaluation Contract

Sprint 7C inherits the same headline contract:

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`.
- Split ID: `sprint2_main_seed42`.
- Guide-disjoint split.
- `experiment_id=18` excluded.
- Measured-only headline train/validation/test universe.
- Test rows measured-only.
- Feature preprocessing from prior artifacts only; no val/test-fitted
  preprocessing.
- Checkpoint selection already occurred by validation AUPRC.
- Threshold selection already occurred by validation max-F1.
- No test tuning.
- AUPRC remains the primary metric.
- MCC, macro F1, specificity/TNR, score distributions, deciles, and confusion
  matrices are secondary diagnostics.
- Test positive prevalence context: `0.900705`; negatives are the rare class.
- Authoritative reference remains `xgboost_unweighted / F4`, test AUPRC
  `0.992522`, AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

Sprint 7C may report that Graph C GATv2 improves MCC/specificity over current
GNN references, but it must not report that as a primary AUPRC win.

## 3. Prior Result Context

Relevant same-contract rows entering Sprint 7C:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | Specificity | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `S7B_REF_XGB_F4` | XGBoost F4 | `0.992522` | `0.938416` | `0.345198` | n/a | `38/131/21/1512` |
| `S7B_REF_GA_GCN` | Graph A GCN + `S5F2_energy` | `0.976935` | `0.819972` | `0.483719` | `0.289941` | `49/120/6/1527` |
| `S7B_REF_GA_GATV2` | Graph A GATv2 + `S5F2_energy` | `0.965449` | `0.818282` | `0.291367` | `0.218935` | `37/132/35/1498` |
| `S7B_REF_GC_GCN` | Graph C GCN + `GraphCContext+S5F2_energy` | `0.972481` | `0.836219` | `0.274287` | `0.082840` | `14/155/0/1533` |
| `S7B_R1_graph_b_gcn_s5f2` | Graph B GCN + `S5F2_energy` | `0.973583` | `0.800777` | `0.437157` | `0.254438` | `43/126/8/1525` |
| `S7B_R2_graph_b_gatv2_s5f2` | Graph B GATv2 + `S5F2_energy` | `0.979139` | `0.832397` | `0.272970` | `0.094675` | `16/153/2/1531` |
| `S7B_R3_graph_c_gatv2_s5f2` | Graph C GATv2 + `GraphCContext+S5F2_energy` | `0.969078` | `0.849705` | `0.531774` | `0.372781` | `63/106/12/1521` |

Sprint 7C's main explanatory contrast is Graph C GCN vs Graph C GATv2:

```text
Graph C GCN:    TN/FP/FN/TP = 14/155/0/1533, MCC = 0.274287
Graph C GATv2:  TN/FP/FN/TP = 63/106/12/1521, MCC = 0.531774
```

This is an operating-point improvement under validation-selected thresholding,
not a primary-metric win. Graph C GATv2's AUPRC is below Graph C GCN and Graph A
GCN but remains above the positive-prevalence floor.

## 4. Literature And Interpretation Basis

Repository-local notes:

- Kipf & Welling, 2017, "Semi-Supervised Classification with Graph Convolutional
  Networks": supports the GCN reference lineage.
- Vinodkumar, Ozcinar & Anbarjafari, 2021, "Prediction of sgRNA Off-Target
  Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network":
  motivates graph/link-prediction framing for CRISPR off-target modeling. This
  project does not reproduce that paper's dataset, split, architecture, or
  metrics.
- Mak et al., 2022, CRISPR-Cas9 off-target epigenetic/nucleosome dataset:
  motivates context and binding-energy feature inspection, but Sprint 7C's
  Scheme A classification analysis is not a Mak CA-regression reproduction.
- Jiang, Li, Xiong & Liu, 2025, "Graph-CRISPR": motivates graph/attention-style
  gene-editing models and interpretation artifacts, but it is an on-target
  efficiency setting rather than this off-target Scheme A task.

Canonical external attention references:

- Velickovic et al., 2018, "Graph Attention Networks": learned
  attention-weighted neighbor aggregation.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention Networks?":
  GATv2 dynamic attention compared with static original GAT attention.

Interpretation rule:

> Attention summaries, score deltas, and recovered-negative profiles are
> model-behavior diagnostics. They are not biological causal evidence unless
> supported by separate experimental or mechanistic evidence.

## 5. Inputs

Primary Sprint 7B artifacts:

- `outputs/sprint7b/gatv2_topology_comparison.csv`
- `outputs/sprint7b/gatv2_topology_report.md`
- `outputs/sprint7b/gatv2_topology_run_manifest.json`
- `outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_threshold_metrics.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_score_deciles.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_per_guide_score_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_per_genome_score_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_attention_summary.csv`
- `outputs/sprint7b/diagnostics/gatv2_topology_attention_contract_summary.csv`

Carry-forward prediction artifacts needed for cross-model row comparisons:

- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_predictions.csv`
- `outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_fixed_threshold_metrics.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/sprint6_loss_comparison_predictions.csv`
- `outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_threshold_metrics.csv`
- `outputs/sprint7/diagnostics/gat_predictions.csv`
- `outputs/sprint7/diagnostics/gat_threshold_metrics.csv`

Optional metadata/graph sources, used only after source-row identity is proven:

- `data/raw/260520_putative_nucleosomal.parquet`
- `data/processed/graphs/sprint3/graph_c_context_observation/`
- `data/processed/graphs/sprint5b/graph_c_context_observation/`
- `data/processed/graphs/sprint7b/graph_c_context_observation/`
- Split artifacts under `outputs/splits/`.

Checkpoint-backed attention extraction is not part of the Sprint 7C core. If it
is later approved, it requires Drive-held `model.pt`, the exact resolved config,
the graph artifact, and a predeclared edge-to-supervised-row mapping.

## 6. Analysis Tiers

### Tier 1: CSV-Only Core Analyses

These analyses must be possible from existing prediction/metric/attention CSVs:

- Identity and alignment audit.
- Metric recomputation from prediction scores.
- Validation threshold transfer checks.
- Score distributions by label and model.
- Positive/negative score deciles and negative-rank summaries.
- Graph C GCN vs Graph C GATv2 score deltas.
- Row-level error transitions, only after alignment passes.
- Per-guide score and error concentration.
- Aggregate attention summaries by edge kind, layer, head, and split.

### Tier 2: Metadata-Enriched Analyses

These analyses are optional and require a verified stable source-row join:

- Cleavage-frequency profiling of recovered negatives and new false negatives.
- Binding-energy feature profiling for recovered negatives.
- Mismatch-count or mismatch-position profiling.
- Cell-line/genome/source-row metadata profiling.
- Target-observation context-feature summaries.
- Context-neighborhood feature summaries.

If source-row identity cannot be proven, Tier 2 must be skipped or reported as
blocked. The report must not infer biological/context-feature composition from
prediction CSV columns alone.

### Tier 3: Checkpoint-Backed Attention Extraction

These analyses are out of the Sprint 7C core and require explicit later approval:

- Attention by confusion category.
- Attention over recovered negatives specifically.
- Attention by score decile.
- Local context-neighborhood attention traces.
- Per-edge attention tied to supervised candidate rows.

Existing aggregate attention CSVs do not contain row-level or edge-level
confusion-category mappings, so they cannot support these claims by themselves.

## 7. Phase 0: Identity And Alignment Audit

Phase 0 is a hard gate. If it fails, Sprint 7C still produces an audit report,
but no row-level error-transition or metadata-enriched claims.

Checks:

1. Verify every loaded artifact path exists and is non-empty.
2. Verify run IDs, predeclared IDs, split names, graph schemas, feature sets, and
   label scheme fields where available.
3. Verify test row count is `1702` and label counts are `1533` positive /
   `169` negative for comparable headline rows.
4. Verify test positive prevalence is `0.900705` within rounding tolerance.
5. Verify validation/test split rows are separate and no unsupported split is
   mixed in.
6. Verify `row_index`, `grna_target_id`, and `label` alignment across comparable
   models for the same split.
7. Audit the meaning of `row_index`: source-row ID vs split-local row position.
8. Verify `genome` consistency where present; record known Graph C `genome` NaN
   limitations rather than silently filling.
9. Recompute AUPRC/AUROC from scores and compare to result tables.
10. Recompute threshold metrics using stored validation-selected thresholds and
    compare to result tables.
11. Emit pass/fail flags for every intended comparison pair.

Required output:

- `outputs/sprint7c/sprint7c_identity_alignment_audit.md`
- `outputs/sprint7c/diagnostics/sprint7c_prediction_alignment_audit.csv`

Acceptance gate:

> Error-transition, per-guide error gain, and metadata-enriched claims may only
> be generated for comparison pairs whose identity/alignment audit passes.

## 8. Phase 1: Existing-Output Explanation

Phase 1 is the Sprint 7C core.

Analyses:

- Score distribution by model, split, and true label.
- Negative score distribution focused on FP-to-TN recovery.
- Graph C GCN vs Graph C GATv2 score-delta histogram.
- Decile/rank analysis for negatives and positives.
- Validation threshold transfer table:
  - validation threshold,
  - validation MCC/specificity/F1,
  - test MCC/specificity/F1,
  - test TN/FP/FN/TP.
- Error transition table for Graph C GCN to Graph C GATv2 if alignment passes:
  - FP to TN recovered negatives,
  - TN to FP lost negatives,
  - TP to FN new false negatives,
  - FN to TP recovered positives, if any.
- Per-guide concentration:
  - number of negatives per guide,
  - recovered TN count,
  - new FP count,
  - new FN count,
  - guide-level score shift.
- Graph B vs Graph C contrast:
  - Graph B GATv2 improves AUPRC but weakens specificity/MCC.
  - Graph C GATv2 improves specificity/MCC but trades off AUPRC modestly.
- Aggregate attention sanity:
  - edge kinds present,
  - candidate/context/self-loop attention summaries,
  - no row-level causal interpretation.

Required outputs:

- `outputs/sprint7c/sprint7c_graphc_gatv2_explanation_report.md`
- `outputs/sprint7c/diagnostics/sprint7c_metric_recomputation.csv`
- `outputs/sprint7c/diagnostics/sprint7c_threshold_transfer.csv`
- `outputs/sprint7c/diagnostics/sprint7c_score_distribution_by_label.csv`
- `outputs/sprint7c/diagnostics/sprint7c_negative_rank_summary.csv`
- `outputs/sprint7c/diagnostics/sprint7c_error_transitions.csv`
- `outputs/sprint7c/diagnostics/sprint7c_per_guide_error_gain.csv`
- `outputs/sprint7c/diagnostics/sprint7c_attention_edge_kind_summary.csv`
- `outputs/sprint7c/figures/sprint7c_score_distribution_by_label.png`
- `outputs/sprint7c/figures/sprint7c_graphc_score_delta.png`
- `outputs/sprint7c/figures/sprint7c_negative_rank_shift.png`
- `outputs/sprint7c/figures/sprint7c_threshold_transfer.png`
- `outputs/sprint7c/figures/sprint7c_error_transition_matrix.png`
- `outputs/sprint7c/figures/sprint7c_per_guide_negative_gain.png`
- `outputs/sprint7c/figures/sprint7c_attention_edge_kind_summary.png`

If an artifact cannot be generated because the Phase 0 gate fails, write a
clear blocked entry in the report instead of producing a misleading fallback.

## 9. Phase 2: Conditional Metadata-Enriched Profiling

Phase 2 is optional within Sprint 7C and only runs after a stable source-row
join is proven.

Required preconditions:

- `row_index` is proven to be a stable source-row identifier, or an equivalent
  stable source-row ID is reconstructed and documented.
- Joined metadata preserves row counts and labels.
- Joined metadata does not use labels to alter graph topology, model selection,
  thresholds, or feature selection.

Possible outputs:

- `outputs/sprint7c/diagnostics/sprint7c_metadata_join_audit.csv`
- `outputs/sprint7c/diagnostics/sprint7c_recovered_negative_profile.csv`
- `outputs/sprint7c/diagnostics/sprint7c_new_false_negative_profile.csv`
- `outputs/sprint7c/diagnostics/sprint7c_context_feature_profile.csv`
- `outputs/sprint7c/figures/sprint7c_recovered_negative_energy_profile.png`
- `outputs/sprint7c/figures/sprint7c_new_false_negative_profile.png`
- `outputs/sprint7c/figures/sprint7c_context_feature_profile.png`

Interpretation limits:

- These profiles may suggest which feature distributions differ among recovered
  negatives, persistent false positives, and new false negatives.
- They must not be presented as proof that a context feature biologically caused
  the model improvement.
- If `genome` is missing in Graph C prediction CSVs, per-genome claims require a
  verified metadata join. Otherwise per-genome analysis is explicitly blocked.

## 10. Phase 3: Deferred Checkpoint-Backed Attention

Phase 3 is not required for Sprint 7C acceptance.

If later approved as Sprint 7C-extra or Sprint 7D, it must define:

- Exact checkpoint path and resolved config.
- Exact graph artifact path.
- Inference-only loader contract.
- `return_attention_weights` extraction path.
- Mapping rule from attention edges to supervised candidate rows.
- Handling of auxiliary edges such as `context_similar_to` that are not
  supervised candidate edges.
- Aggregation rules by split, label, confusion category, edge kind, layer, and
  head.
- Reproducibility checks against existing scores and metrics.

Until this is implemented and audited, Sprint 7C may only report existing
aggregate attention-by-edge-kind summaries.

## 11. Planned Implementation Files

Expected implementation changes when this plan is executed:

- Add `scripts/analyze_sprint7c_graphc_gatv2_explanation.py`.
- Add tests for identity alignment, metric recomputation, and transition gating:
  - `tests/test_sprint7c_graphc_explanation.py`
- Write outputs under `outputs/sprint7c/`.

No changes are expected in:

- `src/crispr_gnn/models/`
- `src/crispr_gnn/training/`
- `src/crispr_gnn/graph/`
- `configs/experiments/`
- `configs/sweeps/`
- `colab/`
- split artifacts
- graph artifacts

If implementation discovers that a tiny utility function must be shared, it may
be added under a reporting/diagnostics module only if it does not affect
training or model behavior.

## 12. Acceptance Criteria

Sprint 7C is accepted only if:

1. No model is trained.
2. No graph artifact, label, split, threshold, feature set, loss, sampler, or
   model architecture is changed.
3. Identity/alignment audit is produced and gates row-level claims.
4. AUPRC/AUROC and threshold metrics are recomputed from prediction CSVs where
   possible and checked against Sprint 7B / carry-forward tables.
5. CSV-only, metadata-enriched, and checkpoint-backed analyses are clearly
   separated in both code and report.
6. Metadata-enriched analysis runs only with a verified stable source-row join;
   otherwise it is reported as blocked.
7. Attention-by-confusion-category is omitted unless an explicitly approved
   checkpoint-backed extraction path is added.
8. Graph C is described as topology plus target-observation semantics/context,
   not topology-only.
9. AUPRC remains primary; MCC/specificity/macro F1 are secondary
   threshold-dependent diagnostics.
10. The report states that Sprint 7C explains the observed single-seed result
    and does not prove robustness or biological causality.
11. Report-ready figures are produced for the completed analysis tier(s).
12. Tests pass with `uv run pytest`.

## 13. Risks

- Prediction CSVs do not include `edge_id`; row-level transitions depend on
  proving `row_index` identity.
- Graph C prediction CSVs may have `genome` as NaN; per-genome analysis needs a
  metadata join.
- Existing attention summaries are aggregate by edge kind/layer/head and do not
  support confusion-category attention claims.
- Local checkout intentionally excludes `model.pt`; checkpoint-backed analyses
  may require Drive-held artifacts.
- The result is single-seed/single-split; Sprint 7C must not imply statistical
  stability.
- Test diagnostics can easily become post-hoc model-selection pressure. Sprint
  7C must keep them explanatory only.
- Graph C changes target semantics as well as topology, so the explanation
  cannot isolate "attention alone" without a later controlled ablation.

## 14. Sprint 7D / Sprint 8 Handoff

Potential Sprint 7D ablations, not part of Sprint 7C:

- Graph C GATv2 without `context_similar_to` auxiliary edges.
- Graph C GATv2 with candidate `S5F2_energy` kept in the classifier but removed
  from attention/message passing.
- Graph C GATv2 with target-observation context features masked or ablated,
  if a clean and predeclared control can be designed.

Sprint 8 robustness, not part of Sprint 7C:

- Multi-seed consolidation.
- Guide-level bootstrap CIs.
- Paired-difference intervals for Graph C GCN vs Graph C GATv2 and Graph A GCN
  vs Graph C GATv2.
- No best-seed selection.

Sprint 7C's job is to make the current Graph C GATv2 finding explainable and
auditable enough to decide whether Sprint 7D ablations or Sprint 8 robustness
are worth running next.
