# Execution Plan: Sprint 7D Graph C GATv2 Mechanism Ablation

> Status: IMPLEMENTED / AWAITING RUN RESULTS. This plan scopes Sprint 7D only.
> It is a small, predeclared mechanism/isolation ablation sprint around the
> Sprint 7B Graph C GATv2 result. It is not a model-search sprint and must not
> add new architectures, losses, samplers, feature families, sequence encoders,
> or data regimes.

## 1. Goal

Sprint 7D tests which pre-existing Graph C components are necessary for the
Sprint 7B Graph C GATv2 rare-negative operating-point gain:

1. `context_similar_to` topology,
2. direct `target_observation` context node features,
3. candidate-edge `S5F2_energy` inside GATv2 attention/message passing,
4. or the interaction of those components.

Scientific framing:

> Sprint 7D tests which Graph C GATv2 components are necessary for the observed
> Sprint 7B rare-negative operating-point behavior under the frozen Scheme A,
> guide-disjoint, measured-only contract.

Sprint 7D will:

- Preserve the Sprint 2/3/4/5/6/7/7B label, split, measured-only, loss,
  checkpoint, threshold, and reporting contract.
- Keep Graph C GATv2 as the only newly trained architecture.
- Run three small, predeclared Graph C GATv2 ablations.
- Carry forward the full Graph C GATv2, Graph C GCN, Graph A GCN, and XGBoost F4
  references.
- Produce artifact-level audits proving exactly what each ablation removed or
  masked.
- Produce report-ready diagnostics and figures matching Sprint 4-7C reporting
  conventions.

Sprint 7D will not:

- Tune ablations, thresholds, loss, topology, heads, dropout, hidden dimension,
  or reporting choices from test results.
- Add GAT, GraphSAGE, R-GCN, HGT, HeteroConv, graph transformers, sequence
  encoders, measured-zero rows, new losses, new samplers, or new feature
  families.
- Remove `S5F2_energy` from both attention and final classifier in the
  edge-blind ablation.
- Regenerate `context_similar_to` topology with different k/distance settings.
- Add target identity embeddings after target-feature masking.
- Treat attention weights as biological causal evidence.
- Claim statistical superiority from one seed.
- Describe Graph C as topology-only.

## 2. Frozen Evaluation Contract

Sprint 7D inherits the same headline contract:

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`.
- Split ID: `sprint2_main_seed42`.
- Guide-disjoint split.
- `experiment_id=18` excluded.
- Headline train/validation/test universe: measured-only.
- Test rows: measured-only.
- Graph visibility: `strict_inductive_primary`.
- Feature family: `S5F2_energy` candidate-edge table, 268 columns.
- Loss: Sprint 6 winner `weighted_bce`, `pos_weight: auto` from train labels.
- Optimizer/scheduler/training defaults: inherit Sprint 7B Graph C GATv2 unless
  a technical compatibility change is predeclared before training.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Test split used only for final reporting.
- Primary metric: AUPRC.
- Secondary threshold diagnostics: AUROC, F1, macro F1, MCC, specificity/TNR,
  sensitivity, TN/FP/FN/TP, score distributions, deciles, per-guide diagnostics,
  aggregate attention summaries.
- Test positive prevalence context: `0.900705`; negatives are the rare class.
- Required AUPRC reference: `xgboost_unweighted / F4`, test AUPRC `0.992522`,
  AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

Sprint 7D may report MCC/specificity/TN movement as mechanism evidence, but it
must not rank a model above another on those metrics when AUPRC disagrees.

## 3. Prior Evidence Entering Sprint 7D

Frozen reference rows:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | Specificity | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `S7D_REF_XGB_F4` | XGBoost F4 | `0.992522` | `0.938416` | `0.345198` | n/a | `38/131/21/1512` |
| `S7D_REF_GRAPH_A_GCN` | Graph A GCN + `S5F2_energy` | `0.976935` | `0.819972` | `0.483719` | `0.289941` | `49/120/6/1527` |
| `S7D_REF_GRAPH_A_GATV2` | Graph A GATv2 + `S5F2_energy` | `0.965449` | `0.818282` | `0.291367` | `0.218935` | `37/132/35/1498` |
| `S7D_REF_GRAPH_C_GCN` | Graph C GCN + `GraphCContext+S5F2_energy` | `0.972481` | `0.836219` | `0.274287` | `0.082840` | `14/155/0/1533` |
| `S7D_REF_FULL_GRAPH_C_GATV2` | Graph C GATv2 full setting | `0.969078` | `0.849705` | `0.531774` | `0.372781` | `63/106/12/1521` |

Sprint 7C row-level explanation:

```text
Graph C GATv2 vs Graph C GCN:
FP -> TN recovered negatives = 53
TN -> FP lost negatives      = 4
Net TN gain                  = +49
TP -> FN new false negatives = 12
```

Interpretation entering Sprint 7D:

- Graph C GATv2 gives the best same-contract GNN rare-negative threshold profile
  observed so far.
- It does not beat XGBoost F4 on primary AUPRC.
- It does not beat Graph C GCN or Graph A GCN on AUPRC.
- The useful open question is not whether Sprint 7D can find a better model; it
  is which Graph C GATv2 component explains the observed operating-point gain.

## 4. Graph C Component Definitions

### Full Graph C GATv2 Reference

Full Graph C GATv2 from Sprint 7B has:

- `sgRNA` nodes with guide features.
- `target_observation` nodes with direct context features.
- Candidate edges `sgRNA -> target_observation`.
- Candidate-edge `S5F2_energy` features passed into:
  - GATv2 attention/message passing through `edge_attr` / `edge_dim`,
  - final candidate-edge classifier.
- Bidirectional candidate attention edges with duplicated candidate edge attrs.
- `context_similar_to` target-observation edges.
- `context_similar_to` edges receive zero edge attributes in attention.
- Self-loop edge attributes are zero-filled.

### Context-Similarity Edges

`context_similar_to` edges are label-free Graph C auxiliary topology. They were
constructed before Sprint 7D and must not be regenerated or reparameterized in
Sprint 7D.

Removing them tests only whether the existing auxiliary context-similarity
topology is necessary. It does not test whether all context is unnecessary,
because target-observation node context features remain active.

### Target-Observation Context Node Features

Direct target-observation context node features enter through the Graph C target
node encoder. Masking these features tests direct node-feature contribution.

This does not remove all context semantics if `context_similar_to` edges remain,
because those edges were derived from context. The correct wording is:

> Direct target-observation context node features were masked; context-derived
> topology remains.

### Candidate `S5F2_energy` In Attention

The edge-blind attention ablation removes or zeroes candidate `S5F2_energy` only
inside GATv2 attention/message passing. `S5F2_energy` must remain in the final
edge classifier so the run isolates edge-aware attention flow rather than the
feature family itself.

## 5. Literature Framing

Use literature only to justify why the axes are meaningful:

- Kipf & Welling, 2017, "Semi-Supervised Classification with Graph Convolutional
  Networks": GCN reference lineage through fixed normalized neighborhood
  aggregation.
- Velickovic et al., 2018, "Graph Attention Networks": learned
  attention-weighted neighbor aggregation.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention Networks?":
  GATv2 dynamic attention relative to original static GAT attention.
- Vinodkumar, Ozcinar & Anbarjafari, 2021, "Prediction of sgRNA Off-Target
  Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network":
  motivates CRISPR off-target graph/link-prediction framing only; this project
  does not reproduce that dataset, split, architecture, target, or metrics.
- Mak et al., 2022, "The influence of epigenetic features on CRISPR-Cas9
  off-target activity": motivates epigenetic/nucleosome/context and
  binding-energy feature use; Sprint 7D Scheme A classification is not Mak CA
  regression reproduction.
- Jiang, Li, Xiong & Liu, 2025, "Graph-CRISPR: a gene editing efficiency
  prediction model based on graph neural network with integrated sequence and
  secondary structure feature extraction": broad support for graph/attention
  gene-editing models and interpretation artifacts; it is not this off-target
  Scheme A task.

Do not include piCRISPR as a core Sprint 7D reference unless a later plan
explicitly scopes physically informed deep-learning baselines. It is separate
from Mak 2022 and not required for this ablation.

## 6. Core Run Matrix

Carry-forward references:

| Run ID | Source | Role |
| --- | --- | --- |
| `S7D_REF_XGB_F4` | Sprint 2 XGBoost F4 | Authoritative AUPRC bar; no retrain. |
| `S7D_REF_GRAPH_A_GCN` | Sprint 6 / Sprint 7 carry-forward | Best Graph A GNN reference; no retrain. |
| `S7D_REF_GRAPH_A_GATV2` | Sprint 7 carry-forward | Failed Graph A attention reference; no retrain. |
| `S7D_REF_GRAPH_C_GCN` | Sprint 5B / Sprint 7B carry-forward | Matched Graph C non-attention reference; no retrain. |
| `S7D_REF_FULL_GRAPH_C_GATV2` | Sprint 7B full Graph C GATv2 | Full reference; no retrain unless explicitly predeclared as a same-harness reproducibility check before test inspection. |

New canonical Sprint 7D ablation runs:

| Run ID | Setting | `context_similar_to` edges | Direct target-observation context features | Candidate `S5F2_energy` in GATv2 attention | Candidate `S5F2_energy` in final classifier | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `S7D_R1_no_context_edges` | Graph C GATv2 without context-similarity topology | No | Yes | Yes | Yes | Primary topology ablation. |
| `S7D_R2_edge_blind_attention` | Graph C GATv2 with candidate-edge-blind attention | Yes | Yes | No | Yes | Tests whether candidate `S5F2_energy` must enter message passing. |
| `S7D_R3_mask_target_context_features` | Graph C GATv2 with direct target node features masked | Yes | No | Yes | Yes | Tests direct target-observation node context-feature contribution. |

Deferred / optional only:

| Run ID | Setting | Role |
| --- | --- | --- |
| `S7D_R4_no_context_edges_plus_masked_target_features` | Graph C row-level target nodes with no context edges and masked target features | Artificial lower-bound structural control; run only if the three core ablations leave the mechanism ambiguous and only if predeclared before running. |

Sprint 7D core has exactly three new training runs. `S7D_R4` is not part of the
default acceptance requirement.

## 7. Interpretation Rules

### `S7D_R1_no_context_edges`

If MCC/specificity/TN drops sharply:

> Existing `context_similar_to` topology appears necessary for the rare-negative
> operating point, given target-observation node features and edge-aware
> candidate energy remain available.

If it stays strong:

> Explicit `context_similar_to` edges are not necessary for the observed
> rare-negative gain under this split; the gain may come from direct target
> context features, row-level target-observation semantics, candidate-edge
> message passing, or classifier access to `S5F2_energy`.

Do not say context is irrelevant, because direct target-observation context
features remain.

### `S7D_R2_edge_blind_attention`

If MCC/specificity/TN drops sharply:

> Candidate `S5F2_energy` inside GATv2 message passing likely contributes to
> rare-negative separation.

If it stays strong:

> Candidate `S5F2_energy` inside attention/message passing is not required for
> the observed Graph C rare-negative profile; `S5F2_energy` in the final
> classifier plus Graph C context/semantics may be sufficient.

Do not say energy features are unimportant. The final classifier still receives
`S5F2_energy`, and Sprint 5/Sprint 6 already established it as the strongest GCN
feature family.

### `S7D_R3_mask_target_context_features`

If MCC/specificity/TN drops sharply:

> Direct target-observation context node features appear important for Graph C
> GATv2's rare-negative threshold behavior.

If it stays strong:

> Direct target-observation context node features are not necessary by
> themselves; the gain may come from context-similarity topology, row-level
> target-observation semantics, candidate-edge message passing, or classifier
> access to `S5F2_energy`.

Do not say all context was removed, because context-derived topology remains.

### Metric Outcome Rules

If AUPRC improves but MCC/specificity collapses:

> The ablation improves ranking but loses the rare-negative validation-threshold
> operating point. Since AUPRC is primary, it is a ranking improvement, but it
> does not explain the Sprint 7B rare-negative gain.

If MCC/specificity improves but AUPRC drops:

> The ablation strengthens a secondary threshold operating point but weakens the
> primary ranking metric. It is mechanism evidence, not headline model
> selection evidence.

If no ablation matches full Graph C GATv2:

> The observed gain appears to depend on the full Graph C GATv2 combination.
> This suggests interaction, not biological causality.

## 8. Implementation Slices

### Slice 0: Output And Source Artifact Audit

- Verify Sprint 7B, Sprint 7C, Sprint 6, Sprint 5B, split, and graph artifacts
  are available locally or fail with clear messages.
- Verify full Graph C GATv2 reference metrics are loaded from the committed
  Sprint 7B outputs, not recomputed from smoke artifacts.
- Verify Graph C artifact contains `S5F2_energy`, `target_observation_features`,
  candidate edges, and `context_similar_to` edges.

Expected output:

- `outputs/sprint7d/graph_artifact_provenance.json`
- `outputs/sprint7d/diagnostics/sprint7d_source_artifact_audit.csv`

### Slice 1: Graph C GATv2 Ablation Controls

Add predeclared Graph C GATv2 ablation controls without changing non-Sprint-7D
behavior:

- `drop_context_similarity_edges`: only removes `context_similar_to` edges from
  the homogeneous attention view.
- `edge_blind_candidate_attention`: keeps candidate `S5F2_energy` in the final
  classifier but zeroes/removes candidate edge attrs in GATv2 attention.
- `mask_target_observation_features`: uniformly masks direct
  `target_observation.x` before the target-observation encoder for train,
  validation, and test views.

Controls must be driven by config/run spec, not by test diagnostics.

Required audit fields:

- candidate forward edge count,
- candidate reverse edge count,
- context edge count before/after,
- candidate attention edge_attr dimension,
- candidate attention edge_attr nonzero status,
- classifier edge-feature dimension,
- target feature norm before/after masking,
- self-loop fill,
- parameter count,
- run ID and ablation flags.

### Slice 2: Sprint 7D Runner

Add a script-only runner, no Colab by default:

- `scripts/run_sprint7d_graphc_gatv2_ablation.py`
- `configs/sweeps/sprint7d_graphc_gatv2_ablation.yaml`

The runner:

- loads Graph C artifacts,
- carries forward references,
- trains only the three predeclared ablations,
- writes per-run directories under `outputs/sprint7d/runs/`,
- writes resolved configs, runtime info, metrics, training history, attention
  summaries where applicable,
- excludes `model.pt` from committed evidence but records checkpoint paths in
  manifests if training creates them.

### Slice 3: Diagnostics And Figures

Required tables:

- `outputs/sprint7d/gatv2_graphc_ablation_results.csv`
- `outputs/sprint7d/gatv2_graphc_ablation_run_manifest.json`
- `outputs/sprint7d/diagnostics/sprint7d_ablation_matrix.csv`
- `outputs/sprint7d/diagnostics/sprint7d_component_audit.csv`
- `outputs/sprint7d/diagnostics/sprint7d_threshold_metrics.csv`
- `outputs/sprint7d/diagnostics/sprint7d_deltas_vs_full_graphc_gatv2.csv`
- `outputs/sprint7d/diagnostics/sprint7d_deltas_vs_graphc_gcn.csv`
- `outputs/sprint7d/diagnostics/sprint7d_score_deciles.csv`
- `outputs/sprint7d/diagnostics/sprint7d_predictions.csv`
- `outputs/sprint7d/diagnostics/sprint7d_training_history.csv`
- `outputs/sprint7d/diagnostics/sprint7d_attention_summary.csv`
- `outputs/sprint7d/diagnostics/sprint7d_error_transitions_vs_full_graphc_gatv2.csv`
- `outputs/sprint7d/diagnostics/sprint7d_error_transitions_vs_graphc_gcn.csv`
- `outputs/sprint7d/diagnostics/sprint7d_per_guide_negative_gain.csv`

Required figures:

- `outputs/sprint7d/figures/sprint7d_auprc_comparison.png`
- `outputs/sprint7d/figures/sprint7d_pr_curves.png`
- `outputs/sprint7d/figures/sprint7d_roc_curves.png`
- `outputs/sprint7d/figures/sprint7d_threshold_metrics.png`
- `outputs/sprint7d/figures/sprint7d_confusion_counts.png`
- `outputs/sprint7d/figures/sprint7d_score_distributions.png`
- `outputs/sprint7d/figures/sprint7d_negative_rank_shift.png`
- `outputs/sprint7d/figures/sprint7d_error_transition_heatmap.png`
- `outputs/sprint7d/figures/sprint7d_per_guide_negative_gain.png`
- `outputs/sprint7d/figures/sprint7d_training_curves.png`
- `outputs/sprint7d/figures/sprint7d_attention_by_edge_kind.png`

Per-genome outputs are optional and must be omitted or explicitly blocked unless
a verified metadata join resolves the Graph C missing-genome limitation noted in
Sprint 7C.

### Slice 4: Report And Completion

Write:

- `outputs/sprint7d/gatv2_graphc_ablation_report.md`

The report must include:

- frozen contract,
- run matrix,
- reference rows,
- result table,
- component audit summary,
- matched deltas vs full Graph C GATv2,
- matched deltas vs Graph C GCN,
- error-transition analysis,
- per-guide rare-negative concentration,
- attention summary with interpretation limits,
- metric interpretation rules,
- Sprint 8 robustness handoff.

## 9. Expected Code Changes

Expected implementation files when Sprint 7D is executed:

- `configs/sweeps/sprint7d_graphc_gatv2_ablation.yaml`
- `scripts/run_sprint7d_graphc_gatv2_ablation.py`
- `tests/test_sprint7d_graphc_ablation_model.py`
- `tests/test_sprint7d_graphc_ablation_runner.py`

Likely modified files:

- `src/crispr_gnn/models/gat.py`
- `src/crispr_gnn/training/gcn.py`
- `src/crispr_gnn/models/__init__.py` if new helper exports are needed.

Do not change:

- labels,
- split artifacts,
- graph artifact tables,
- loss implementation,
- sampler implementation,
- feature builders,
- non-Sprint-7D model behavior.

## 10. Acceptance Criteria

Sprint 7D is accepted only if:

1. The run matrix is frozen before implementation.
2. Exactly three new core ablation runs are trained unless a documented
   technical failure blocks one.
3. Full Graph C GATv2, Graph C GCN, Graph A GCN, Graph A GATv2, and XGBoost F4
   are carried forward as references without retraining.
4. Scheme A, `sprint2_main_seed42`, measured-only universe, `experiment_id=18`
   exclusion, weighted BCE, validation-AUPRC checkpoint, and validation max-F1
   threshold are preserved.
5. No model, threshold, topology, loss, sampler, or feature decision is made from
   test diagnostics.
6. Component audits prove exactly which edges/features/edge attrs are active in
   each ablation.
7. `S7D_R2_edge_blind_attention` keeps `S5F2_energy` in the final classifier.
8. `S7D_R3_mask_target_context_features` masks target-observation features
   uniformly across train/validation/test.
9. AUPRC is primary in tables, figures, and conclusions.
10. MCC/specificity/macro F1/TN are labeled secondary threshold diagnostics.
11. Graph C is never described as topology-only.
12. Attention summaries remain interpretation-only and are not described as
    biological causal evidence.
13. Per-genome claims are blocked unless a verified metadata join is implemented.
14. Tests pass with `uv run pytest`.
15. `uv run ruff check` passes on changed code.
16. Final committed evidence excludes `model.pt` checkpoints and `.DS_Store`.

## 11. Sprint 8 Handoff

Sprint 7D remains single-seed mechanism analysis with seed 42. It may identify
which component appears necessary for the observed Sprint 7B behavior under the
locked split, but it must not claim general statistical superiority.

Sprint 8 remains the robustness layer:

- multi-seed consolidation,
- guide-level bootstrap CIs,
- paired-difference intervals,
- no best-seed selection,
- comparison of full Graph C GATv2 and any Sprint 7D ablation worth carrying
  forward.

The correct Sprint 7D closeout language is:

> This ablation suggests which components are necessary for the observed Sprint
> 7B Graph C GATv2 behavior under the locked split.

Not:

> This proves the component is generally necessary or biologically causal.
