# Execution Plan: Sprint 7F Family-Aware Target Context Encoder

> Status: PLANNED. Sprint 7F is a model-improvement sprint guided by Sprint 7D
> and Sprint 7E mechanism evidence. It changes only the Graph C
> `target_observation` encoder. It is not a topology sprint, sequence-encoder
> sprint, hyperparameter search, robustness sprint, or biological causality
> audit.

## 1. Goal

Sprint 7D showed that Graph C GATv2's rare-negative operating-point behavior is
driven mainly by direct `target_observation` features rather than
`context_similar_to` topology. Sprint 7E then showed that, within that direct
context vector, the six experimental epigenetic features are necessary for
reproducing the Graph C/no-context-edge GATv2 rare-negative behavior.

Sprint 7F asks the next modeling question:

> Can Graph C GATv2 represent the Sprint 7E-identified target-observation
> context signal better by using a family-aware target context encoder, while
> keeping topology, edge features, loss, split, checkpointing, and thresholding
> fixed?

Sprint 7F will:

- Keep the frozen Scheme A / guide-disjoint / measured-only / validation-only
  checkpoint-threshold contract.
- Keep Graph C, GATv2, `S5F2_energy`, weighted BCE, seed `42`, and
  `sprint2_main_seed42` fixed.
- Keep candidate `S5F2_energy` active in GATv2 attention/message passing and
  the final edge classifier.
- Use the Sprint 7E primary base: Graph C GATv2 with `context_similar_to` edges
  dropped.
- Vary only the `target_observation` encoder used to map the 212-column context
  vector into the Graph C hidden dimension.
- Add encoder audits showing feature-family column partitioning, branch
  dimensions, parameter counts, and forward-pass activation summaries.
- Produce consolidated CSV, diagnostics, figures, manifest, provenance, and a
  final Markdown report matching Sprint 5-7E output conventions.

Sprint 7F will not:

- Add CRISPR-Net-style CNN/RNN/BiLSTM sequence encoders.
- Add RNA secondary-structure graphs, RNA language-model embeddings, HGT,
  R-GCN, GraphSAGE, graph transformers, HeteroConv, late-fusion XGBoost/GNN
  models, or hybrid tabular models.
- Change Graph C topology or rebuild `context_similar_to` edges.
- Change label scheme, split, measured-only headline universe, loss,
  thresholding, checkpointing, sampler, optimizer, scheduler, GATv2 heads,
  dropout, hidden dimension, or edge-feature policy.
- Choose encoder variants, branch widths, thresholds, reruns, or follow-up
  runs from test diagnostics.
- Treat branch activations, attention weights, or feature-family effects as
  biological causal evidence.
- Claim final statistical superiority from one seed.

## 2. Frozen Evaluation Contract

Sprint 7F inherits the same headline contract:

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`.
- Split ID: `sprint2_main_seed42`.
- Guide-disjoint split.
- `experiment_id=18` excluded.
- Headline train/validation/test universe: measured-only.
- Test rows: measured-only.
- Graph visibility: `strict_inductive_primary`.
- Graph schema: `graph_c_context_observation`.
- Candidate edge feature family: `S5F2_energy`, 268 columns.
- Target node feature table: `target_observation_features`, 212 columns.
- Loss: Sprint 6 winner `weighted_bce`, `pos_weight: auto` from train labels.
- Optimizer/scheduler/training defaults: inherit Sprint 7E unless a technical
  compatibility change is predeclared before training.
- GATv2 policy: 2 layers, hidden dim `128`, 4 heads, concat true, dropout `0.2`,
  attention dropout `0.2`, `share_weights=false`, self-loop edge fill `0.0`.
- Graph C topology policy: `context_similar_to` edges dropped for every
  canonical Sprint 7F run.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Test split used only for final reporting.
- Primary metric: AUPRC.
- Secondary threshold diagnostics: AUROC, F1, macro F1, MCC, specificity/TNR,
  sensitivity, TN/FP/FN/TP, score distributions, deciles, per-guide diagnostics,
  aggregate attention summaries, and encoder activation audits.
- Test positive prevalence context: `0.900705`; negatives are the rare class.
- Required AUPRC reference: `xgboost_unweighted / F4`, test AUPRC `0.992522`,
  AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

MCC/specificity/TN movement may be reported as rare-negative operating-point
evidence, but it must not override AUPRC ranking.

## 3. Prior Evidence Entering Sprint 7F

Carry-forward references:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | Specificity | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `S7F_REF_XGB_F4` | XGBoost F4 | `0.992522` | `0.938416` | `0.345198` | n/a | `38/131/21/1512` |
| `S7F_REF_GRAPH_A_GCN` | Graph A GCN + `S5F2_energy` | `0.976935` | `0.819972` | `0.483719` | `0.289941` | `49/120/6/1527` |
| `S7F_REF_GRAPH_C_GCN` | Graph C GCN + `GraphCContext+S5F2_energy` | `0.972481` | `0.836219` | `0.274287` | `0.082840` | `14/155/0/1533` |
| `S7F_REF_FULL_GRAPH_C_GATV2` | Full Graph C GATv2 | `0.969078` | `0.849705` | `0.531774` | `0.372781` | `63/106/12/1521` |
| `S7F_REF_NO_CONTEXT_EDGE_GATV2` | Graph C GATv2 without context edges | `0.965598` | `0.850137` | `0.517970` | `0.366864` | `62/107/14/1519` |

Sprint 7D mechanism evidence:

- Dropping `context_similar_to` edges preserved most of the full Graph C GATv2
  rare-negative profile.
- Removing `S5F2_energy` only from GATv2 attention/message passing degraded both
  ranking and threshold behavior.
- Masking all direct target-observation features collapsed negative recognition.

Sprint 7E subgroup evidence:

- Masking experimental epigenetic features collapsed the Graph C/no-context-edge
  GATv2 model: AUPRC `0.885321`, AUROC `0.350896`, MCC `-0.011388`,
  specificity `0.000000`, TN/FP/FN/TP `0/169/2/1531`.
- Masking all nonsequence context reproduced the same collapse: AUPRC
  `0.890660`, AUROC `0.389521`, MCC `-0.011388`, specificity `0.000000`,
  TN/FP/FN/TP `0/169/2/1531`.
- Masking target sequence, computed nucleosome aggregates, or computed
  missingness did not collapse the model.
- The per-feature audit showed large positive-minus-negative SMD for `MNase` on
  validation (`0.862604`) and test (`0.813250`), with smaller but visible
  experimental-feature separation for `epigen_h3k4me3`, `epigen_drip`, and
  `epigen_dnase`.

Sprint 7F therefore focuses on representation of the direct target-observation
feature vector, not on adding more topology or a larger sequence model.

## 4. Literature Framing

Use literature to justify the design axis, not to claim reproduction.

- Stortz et al., 2021, "crisprSQL: a novel database platform for CRISPR/Cas
  off-target cleavage assays": motivates observation-level cellular context,
  source/assay/cell-line metadata caution, epigenetic marker traceability, and
  the need to distinguish cellular-context assays from purely sequence-only
  prediction.
- Mak et al., 2022, "Comprehensive computational analysis of epigenetic
  descriptors affecting CRISPR-Cas9 off-target activity": motivates the 6
  experimental epigenetic + 13 computed nucleosome feature families, binding
  energy features, and model-based feature inspection. Mak found computed
  nucleosome features important in their CA regression/SHAP setting, whereas
  Sprint 7E found direct experimental epigenetic features necessary in this
  measured-only Scheme A Graph C/GATv2 setting. Sprint 7F must report this as a
  task/contract difference, not as a contradiction or reproduction claim.
- Lin et al., 2020, "CRISPR-Net: A Recurrent Convolutional Network Quantifies
  CRISPR Off-Target Activities with Mismatches and Indels": motivates that
  richer sequence encoders can matter for CRISPR off-target prediction, but
  Sprint 7F deliberately defers CRISPR-Net-style CNN/RNN/BiLSTM encoders so the
  controlled variable remains target-context representation.
- Jiang et al., 2025, "Graph-CRISPR: a gene editing efficiency prediction model
  based on graph neural network with integrated sequence and secondary structure
  feature extraction": motivates integrated graph + feature extraction modules
  and cautious interpretation artifacts, but it is an on-target efficiency
  setting and must not be described as reproduced by Sprint 7F.
- Velickovic et al., 2018, "Graph Attention Networks": canonical learned
  attention-weighted neighbor aggregation reference.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention
  Networks?": canonical GATv2 dynamic-attention reference.

No dedicated local paper note for Velickovic et al. or Brody et al. is present
under `docs/literature/`; use those as canonical references by title/arXiv only
unless repository notes are added later.

## 5. Primary Design Decision

Sprint 7F's primary base remains:

```text
Graph C GATv2
context_similar_to edges = dropped
S5F2_energy edge-aware attention = active
S5F2_energy final classifier input = active
weighted BCE = active
target_observation features = active
only target_observation encoder architecture varies
```

Reason:

- Sprint 7D showed `context_similar_to` topology is not the main driver.
- Sprint 7E showed direct target-observation features, especially experimental
  epigenetic features, are necessary.
- A better target-context encoder is the smallest model-improvement step that
  follows the evidence without confounding the result with sequence encoders,
  topology changes, loss changes, or hyperparameter search.

Sprint 7F's canonical model family is **family-aware target context encoding**:

```text
target_sequence_one_hot branch
experimental_epigenetic branch
computed_nucleosome_aggregate branch
computed_nucleosome_missingness branch
        ↓
concatenate
        ↓
fusion MLP to hidden_dim
        ↓
Graph C GATv2 message passing + S5F2 edge-aware attention
```

The experimental epigenetic branch is the Sprint 7E-guided branch. It should be
explicit and auditable, but not the only input path in the main model.

## 6. Encoder Variants

### Current Unified Encoder Reference

Existing `GraphCEdgeGATv2` uses one shallow unified encoder:

```text
Linear(212 -> hidden_dim) + ReLU
```

This row is carried forward as `S7F_REF_NO_CONTEXT_EDGE_GATV2`; no retraining is
required unless a same-harness reproducibility check is explicitly predeclared
before test inspection.

### Deeper Unified Encoder

Purpose: test whether the current target context vector needs a modestly more
expressive MLP before message passing, without family-specific inductive bias.

Predeclared shape:

```text
Linear(212 -> hidden_dim)
LayerNorm(hidden_dim)
ReLU
Dropout(0.2)
Linear(hidden_dim -> hidden_dim)
ReLU
```

This is a capacity control for the family-aware encoder. If family-aware wins
only because it has more parameters, this run should narrow that interpretation.

### Family-Aware Encoder

Purpose: test whether separating source-defined feature families improves
context representation and prevents the six experimental epigenetic features
from being diluted inside the 212-column unified input.

Predeclared branch partition:

| Family | Input Columns | Branch Output Dim |
| --- | ---: | ---: |
| `target_sequence_one_hot` | 115 | 32 |
| `experimental_epigenetic` | 6 | 32 |
| `computed_nucleosome_aggregates` | 78 | 48 |
| `computed_nucleosome_missingness` | 13 | 16 |

Concatenate branch outputs (`128` total) and apply a fusion block:

```text
LayerNorm(128)
ReLU
Dropout(0.2)
Linear(128 -> hidden_dim)
ReLU
```

Each branch should be a small MLP:

```text
Linear(input_dim -> branch_output_dim)
LayerNorm(branch_output_dim)
ReLU
```

### Family-Aware Encoder With Experimental Emphasis

Purpose: test the Sprint 7E-specific hypothesis that the experimental
epigenetic signal needs explicit capacity, while preserving all four feature
families.

Predeclared branch partition:

| Family | Input Columns | Branch Output Dim |
| --- | ---: | ---: |
| `target_sequence_one_hot` | 115 | 24 |
| `experimental_epigenetic` | 6 | 48 |
| `computed_nucleosome_aggregates` | 78 | 40 |
| `computed_nucleosome_missingness` | 13 | 16 |

The total concatenated dimension remains `128`, so this is an allocation change
inside the target-context encoder rather than a hidden-dimension expansion.

This run must not be described as "MNase causal" or "epigenetic-only." It is a
predeclared representation allocation motivated by Sprint 7E.

## 7. Predeclared Run Matrix

Carry-forward references:

| Run ID | Source | Role |
| --- | --- | --- |
| `S7F_REF_XGB_F4` | Sprint 2 XGBoost F4 | Authoritative AUPRC bar; no retrain. |
| `S7F_REF_GRAPH_A_GCN` | Sprint 6 S6R0 | Best Graph A GNN reference; no retrain. |
| `S7F_REF_GRAPH_C_GCN` | Sprint 5B | Graph C non-attention reference; no retrain. |
| `S7F_REF_FULL_GRAPH_C_GATV2` | Sprint 7B | Full Graph C GATv2 reference; no retrain. |
| `S7F_REF_NO_CONTEXT_EDGE_GATV2` | Sprint 7D / 7E primary base | Current unified shallow encoder reference; no retrain. |

Canonical Sprint 7F training runs:

| Run ID | Encoder | Context Edges | Role |
| --- | --- | --- | --- |
| `S7F_R1_unified_deep_context_encoder` | deeper unified MLP | dropped | Capacity-control target-context encoder. |
| `S7F_R2_family_aware_context_encoder` | balanced family-aware branches | dropped | Main Sprint 7F candidate. |
| `S7F_R3_family_aware_experimental_emphasis` | family-aware with larger experimental branch | dropped | 7E-guided epigenetic-branch allocation test. |

Default canonical training runs: **3**. Do not add more canonical runs after
seeing test results.

Optional / approval-gated diagnostics only:

| Run ID | Setting | Reason To Defer |
| --- | --- | --- |
| `S7F_R4_experimental_only_context_encoder` | encode only the 6 experimental epigenetic features, zero other target context families | Tests sufficiency of epigenetic features, but changes feature availability and may repeat 7E masking logic rather than improve representation. |
| `S7F_R5_family_aware_with_context_edges` | best predeclared family-aware encoder with `context_similar_to` edges restored | Reintroduces topology confound; only useful after the no-context-edge encoder question is settled. |
| `S7F_R6_multi_seed_best_candidate` | fixed split, predeclared seeds for one selected candidate | Belongs naturally to Sprint 8 robustness unless explicitly approved before test inspection. |

## 8. Implementation Scope

Likely code targets:

- Add target context encoder modules, preferably under
  `src/crispr_gnn/models/target_context_encoder.py` or a narrowly scoped section
  of `src/crispr_gnn/models/gat.py`.
- Extend `GraphCEdgeGATv2` to accept:
  - `target_context_encoder_type`,
  - feature-family indexes/names,
  - branch output dimensions,
  - encoder audit metadata.
- Extend Graph C model dispatch in `src/crispr_gnn/training/gcn.py` without
  changing Graph A, Graph B, or GCN behavior.
- Add config:
  `configs/sweeps/sprint7f_target_context_encoder.yaml`.
- Add runner:
  `scripts/run_sprint7f_target_context_encoder.py`.
- Add Colab runner:
  `colab/sprint7f_target_context_encoder_runner.ipynb`.
- Add tests:
  `tests/test_sprint7f_target_context_encoder.py`.

The Colab notebook must remain a runner only. It may mount Drive, clone/update
the branch, sync dependencies, build/copy required artifacts, call the
repository runner, validate outputs, and copy returned outputs. It must not
define model classes or metric/report logic.

## 9. Encoder Audit Requirements

Every trained Sprint 7F run must write:

```text
runs/<run_id>/target_context_encoder_audit.csv
```

Required audit fields:

- encoder type,
- branch names,
- branch input column counts,
- branch output dimensions,
- resolved feature columns per branch,
- total target encoder parameter count,
- total model parameter count,
- context edges used (`0` for canonical runs),
- candidate `S5F2_energy` attention edge attributes nonzero,
- candidate `S5F2_energy` classifier attributes nonzero,
- target context input absolute sum before encoder,
- branch output mean/std/L2 norm by split,
- final target embedding mean/std/L2 norm by split.

The audit must prove that the controlled variable is target-context encoder
design only.

## 10. Output Contract

Required consolidated outputs:

```text
outputs/sprint7f/target_context_encoder_comparison.csv
outputs/sprint7f/target_context_encoder_report.md
outputs/sprint7f/target_context_encoder_run_manifest.json
outputs/sprint7f/graph_artifact_provenance.json
```

Required diagnostics:

```text
outputs/sprint7f/diagnostics/target_context_encoder_threshold_metrics.csv
outputs/sprint7f/diagnostics/target_context_encoder_deltas.csv
outputs/sprint7f/diagnostics/target_context_encoder_training_history.csv
outputs/sprint7f/diagnostics/target_context_encoder_predictions.csv
outputs/sprint7f/diagnostics/target_context_encoder_score_deciles.csv
outputs/sprint7f/diagnostics/target_context_encoder_per_guide_score_summary.csv
outputs/sprint7f/diagnostics/target_context_encoder_attention_summary.csv
outputs/sprint7f/diagnostics/target_context_encoder_encoder_audit.csv
outputs/sprint7f/diagnostics/target_context_encoder_branch_activation_summary.csv
outputs/sprint7f/diagnostics/target_context_encoder_parameter_counts.csv
```

Required figures:

```text
outputs/sprint7f/figures/target_context_encoder_auprc_comparison.png
outputs/sprint7f/figures/target_context_encoder_threshold_metrics.png
outputs/sprint7f/figures/target_context_encoder_pr_curves.png
outputs/sprint7f/figures/target_context_encoder_roc_curves.png
outputs/sprint7f/figures/target_context_encoder_score_distributions.png
outputs/sprint7f/figures/target_context_encoder_training_curves.png
outputs/sprint7f/figures/target_context_encoder_branch_activation_norms.png
outputs/sprint7f/figures/target_context_encoder_parameter_counts.png
```

Per-run directories must contain:

```text
resolved_config.yaml
runtime.json
training_history.csv
metrics.csv
attention_summary.csv
target_context_encoder_audit.csv
model.pt  # Drive-held / untracked
```

## 11. Report Interpretation Rules

Primary comparisons:

- Compare each trained Sprint 7F row primarily to
  `S7F_REF_NO_CONTEXT_EDGE_GATV2`.
- Also report deltas against full Graph C GATv2, Graph C GCN, Graph A GCN, and
  XGBoost F4.
- AUPRC remains primary.
- MCC, specificity, macro F1, TN/FP/FN/TP remain rare-negative operating-point
  diagnostics.
- Parameter-count differences must be reported next to performance changes.
- Branch activation/attention summaries are interpretation-only artifacts.

Allowed conclusion shapes:

- If family-aware encoder improves AUPRC and keeps/improves MCC/specificity:
  "Family-aware target context encoding is a promising model-improvement
  candidate under the frozen single-seed contract."
- If family-aware encoder improves only MCC/specificity while lowering AUPRC:
  "The encoder improves rare-negative operating-point behavior but not the
  primary ranking metric."
- If deeper unified encoder matches family-aware:
  "The gain is likely capacity-related rather than family partition-specific."
- If experimental-emphasis encoder outperforms balanced family-aware:
  "The Sprint 7E experimental-context signal benefits from explicit branch
  capacity allocation under this split."
- If none improve over the reference:
  "Target context encoder architecture is not the current bottleneck; proceed
  to robustness, metadata confound analysis, or larger sequence/context models
  only with a new plan."

Disallowed claims:

- "Experimental epigenetic features are causal biological evidence."
- "MNase alone explains off-target biology."
- "Sprint 7F reproduces Mak, CRISPR-Net, Graph-CRISPR, GAT, or GATv2."
- "MCC/specificity improvement is an AUPRC improvement."
- "Best seed" or "best rerun" selection.

## 12. Tests And Validation

Required local checks before committing implementation:

```bash
uv run pytest tests/test_sprint7f_target_context_encoder.py -q
uv run pytest tests/test_sprint7b_gatv2_model.py tests/test_sprint7e_target_context_features.py -q
uv run ruff check src/crispr_gnn/models/gat.py src/crispr_gnn/models/target_context_encoder.py src/crispr_gnn/training/gcn.py scripts/run_sprint7f_target_context_encoder.py tests/test_sprint7f_target_context_encoder.py
git diff --check
```

Runner smoke tests should monkeypatch training, as Sprint 7B/7D/7E runner tests
do. Do not start full training locally unless the user explicitly asks and local
GPU/artifacts are available.

## 13. Acceptance Criteria

Sprint 7F is ready to run when:

- The run matrix has exactly the 3 canonical trained rows unless optional rows
  are approved before any Sprint 7F test inspection.
- All canonical runs keep Graph C GATv2, `S5F2_energy`, weighted BCE, split,
  seed, checkpoint policy, threshold policy, and context-edge drop fixed.
- The only canonical controlled variable is `target_context_encoder_type` and
  its predeclared branch allocation.
- Feature-family indexes are resolved from artifact feature names and validated
  against the 212-column target-observation contract.
- Encoder audits prove branch dimensions, feature-family columns, parameter
  counts, branch activations, context-edge drop, and candidate S5F2 activity.
- Output contract includes CSV, report, manifest, provenance, diagnostics, and
  figures.
- Local tests and lint pass.
- Colab notebook contains runner commands only.
- Returned outputs exclude committed `model.pt` checkpoints and `.DS_Store`.

Sprint 7F is complete when:

- Returned Colab outputs are copied locally under `outputs/sprint7f/`.
- The final report interprets results under AUPRC-primary, no-test-tuning,
  single-seed boundaries.
- The plan can be moved to `docs/exec-plans/completed/` only after validated
  outputs and final report are present.

## 14. Deferred Work

Do not include these in Sprint 7F:

- CRISPR-Net-style CNN/RNN/BiLSTM sequence encoder.
- RNA secondary-structure graph construction or RNA language model embeddings.
- Epigenetic-only headline model.
- GATv2 head/dropout/hidden-dimension search.
- Rebuilt context topology per encoder.
- Multi-seed/paired-difference robustness.
- Source/cell-line/assay metadata confound modeling.
- External validation datasets.

Natural follow-ups after Sprint 7F:

- Sprint 7G / sequence-context model if target-context encoder improvements do
  not close the XGBoost AUPRC gap.
- Sprint 8 robustness: multi-seed, guide-level bootstrap, paired differences,
  and confidence intervals for the stabilized candidate model(s).
- Metadata-aware audit for experimental epigenetic source/cell-line/assay
  confounds before any biological interpretation claim.
