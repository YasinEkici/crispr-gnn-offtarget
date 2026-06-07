# Execution Plan: Sprint 7E Target-Observation Context Feature Subgroup Ablation

> Status: ACTIVE IMPLEMENTATION. Sprint 7E is one sprint with two slices:
> local feature profiling/predeclaration, then Colab subgroup-ablation training.
> It is not split into 7E1/7E2. It is not a hyperparameter search, encoder
> search, sequence-model sprint, or robustness sprint.

## 1. Goal

Sprint 7D showed that the Sprint 7B Graph C GATv2 rare-negative operating point
depends primarily on direct `target_observation` context node features, with
candidate `S5F2_energy` edge-aware message passing as an important complementary
signal. Sprint 7E asks the next narrower question:

> Which source-defined `target_observation` feature families carry the Graph C
> GATv2 rare-negative behavior?

Sprint 7E will:

- Keep the frozen Scheme A / guide-disjoint / measured-only / validation-only
  checkpoint-threshold contract.
- Keep Graph C GATv2, `S5F2_energy`, weighted BCE, optimizer, scheduler, seed,
  split, and target-observation semantics fixed.
- Use a single Sprint 7E plan with:
  - Slice 1: local `target_observation` feature-family profiling and run-matrix
    predeclaration.
  - Slice 2: Colab training for the predeclared feature-family masks.
- Vary only which `target_observation.x` feature-family columns are masked
  before the Graph C target-observation encoder.
- Produce artifact-level audits proving which target feature columns were
  masked and that candidate `S5F2_energy` remained active in GATv2 attention and
  the final edge classifier.
- Produce consolidated CSV, diagnostics, figures, manifest, provenance, and a
  final Markdown report matching Sprint 5-7D output conventions.

Sprint 7E will not:

- Add sequence encoders, CRISPR-Net-style CNN/RNN modules, transformer modules,
  HeteroConv/R-GCN/HGT, graph transformers, late-fusion tabular models, or
  hybrid XGBoost-GNN models.
- Tune hidden size, heads, dropout, learning rate, loss, sampler, threshold,
  topology, or graph schema from test results.
- Rebuild Graph C context-similarity topology per feature subgroup.
- Move `target_observation` context onto candidate edge features.
- Remove candidate `S5F2_energy` from GATv2 attention or the final classifier.
- Treat attention weights, feature masks, or context profiles as biological
  causal evidence.
- Claim statistical superiority from one seed.

## 2. Frozen Evaluation Contract

Sprint 7E inherits the same headline contract:

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
- Optimizer/scheduler/training defaults: inherit Sprint 7D Graph C GATv2 unless
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

MCC/specificity/TN movement may be reported as mechanism evidence, but it must
not override AUPRC ranking.

## 3. Prior Evidence Entering Sprint 7E

Reference rows:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | Specificity | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `S7E_REF_XGB_F4` | XGBoost F4 | `0.992522` | `0.938416` | `0.345198` | n/a | `38/131/21/1512` |
| `S7E_REF_GRAPH_A_GCN` | Graph A GCN + `S5F2_energy` | `0.976935` | `0.819972` | `0.483719` | `0.289941` | `49/120/6/1527` |
| `S7E_REF_GRAPH_C_GCN` | Graph C GCN + `GraphCContext+S5F2_energy` | `0.972481` | `0.836219` | `0.274287` | `0.082840` | `14/155/0/1533` |
| `S7E_REF_FULL_GRAPH_C_GATV2` | Full Graph C GATv2 | `0.969078` | `0.849705` | `0.531774` | `0.372781` | `63/106/12/1521` |
| `S7E_REF_NO_CONTEXT_EDGE_GATV2` | Graph C GATv2 without `context_similar_to` edges | `0.965598` | `0.850137` | `0.517970` | `0.366864` | `62/107/14/1519` |

Sprint 7D mechanism evidence:

- Removing `context_similar_to` edges caused only minor degradation.
- Removing `S5F2_energy` only from GATv2 attention/message passing degraded both
  ranking and threshold behavior.
- Masking all direct `target_observation` node features collapsed negative
  recognition: AUPRC `0.893657`, AUROC `0.407817`, MCC `-0.013952`,
  specificity `0.000000`, TN/FP/FN/TP `0/169/3/1530`.

Sprint 7E therefore focuses on the direct `target_observation` feature matrix,
not on adding more topology or larger sequence encoders.

## 4. Target-Observation Feature Families

Graph C `target_observation_features` are built in
`src/crispr_gnn/graph/graph_builder.py`:

1. `target_sequence_one_hot`
   - Source: `target_sequence`.
   - Columns: `feature__target_pos_00_A` ... `feature__target_pos_22_N`.
   - Expected count: `23 * 5 = 115`.
2. `experimental_epigenetic`
   - Source columns: `epigen_ctcf`, `epigen_dnase`, `epigen_rrbs`,
     `epigen_h3k4me3`, `epigen_drip`, `MNase`.
   - Expected count: `6`.
3. `computed_nucleosome_aggregates`
   - Source columns: 13 computed nucleosome arrays, transformed into aggregate
     features.
   - Aggregate suffixes: `mean`, `std`, `min`, `max`, `center`,
     `pam_proximal_mean`.
   - Expected count: `13 * 6 = 78`.
4. `computed_nucleosome_missingness`
   - Missingness indicators for the same 13 computed nucleosome arrays.
   - Expected count: `13`.

Expected total: `115 + 6 + 78 + 13 = 212` target-observation columns.

`target_observation` feature preprocessing is train-only:

- experimental epigenetic and computed nucleosome context are median-imputed on
  train rows only,
- then standard-scaled from train rows only,
- and the same transform is applied to validation/test rows.

Sprint 7E must audit feature names and counts from the materialized artifact
instead of assuming them blindly.

## 5. Literature Framing

Use literature to justify the axes, not to claim reproduction:

- Stortz et al., 2021, "crisprSQL: a novel database platform for CRISPR/Cas
  off-target cleavage assays": motivates the dataset's observation-level
  context, epigenetic annotations, computed nucleosome context, and the need to
  distinguish cellular-context assays from purely sequence-only scoring.
- Mak et al., 2022, "The influence of epigenetic features on CRISPR-Cas9
  off-target activity": motivates context-feature inspection, binding-energy
  features, experimental epigenetic markers, and computed nucleosome features.
  Sprint 7E Scheme A classification is not Mak CA regression reproduction.
- Kipf & Welling, 2017, "Semi-Supervised Classification with Graph
  Convolutional Networks": GCN reference lineage through fixed normalized
  neighbor aggregation.
- Velickovic et al., 2018, "Graph Attention Networks": learned
  attention-weighted neighbor aggregation.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention
  Networks?": GATv2 dynamic attention relative to static GAT attention.
- Vinodkumar, Ozcinar & Anbarjafari, 2021, "Prediction of sgRNA Off-Target
  Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network":
  motivates CRISPR off-target graph/link-prediction framing only; this project
  does not reproduce that setup.
- Lin et al., 2020, "CRISPR-Net: A Recurrent Convolutional Network Quantifies
  CRISPR Off-Target Activities with Mismatches and Indels": motivates sequence
  encoder and position-sensitivity ideas as future scope. Sprint 7E will not
  add CRISPR-Net-style CNN/RNN encoders because that would confound whether
  gains come from target context families, sequence modeling, or attention.
- Jiang, Li, Xiong & Liu, 2025, "Graph-CRISPR: a gene editing efficiency
  prediction model based on graph neural network with integrated sequence and
  secondary structure feature extraction": broad support for graph/attention
  gene-editing models and interpretation artifacts; it is not this off-target
  Scheme A task.

## 6. Primary Design Decision

Sprint 7E's primary ablation base is:

```text
Graph C GATv2
S5F2_energy edge-aware attention = active
weighted BCE = active
target_observation node features = active except predeclared masked family
context_similar_to edges = dropped
```

Reason: Sprint 7D showed `context_similar_to` topology is not the main driver
and that the no-context-edge run nearly preserves the full rare-negative
operating point. Dropping those edges in Sprint 7E makes subgroup masking
cleaner: if a target context feature family is masked, its information is less
likely to remain indirectly available through context-similarity topology that
was built from the full context vector.

Carry-forward full Graph C GATv2 remains a reference, not the primary 7E
ablation base. This avoids the confound:

> "Feature family was masked on target nodes, but topology still encodes the
> original full-context similarity relation."

## 7. Predeclared Run Matrix

Carry-forward references:

| Run ID | Source | Role |
| --- | --- | --- |
| `S7E_REF_XGB_F4` | Sprint 2 XGBoost F4 | Authoritative AUPRC bar; no retrain. |
| `S7E_REF_GRAPH_A_GCN` | Sprint 6 / Sprint 7 carry-forward | Best Graph A GNN reference; no retrain. |
| `S7E_REF_GRAPH_C_GCN` | Sprint 5B / Sprint 7B carry-forward | Graph C non-attention reference; no retrain. |
| `S7E_REF_FULL_GRAPH_C_GATV2` | Sprint 7B full Graph C GATv2 | Full Graph C attention reference; no retrain. |
| `S7E_REF_NO_CONTEXT_EDGE_GATV2` | Sprint 7D R1 | Primary 7E base reference; no retrain unless explicitly predeclared as same-harness reproducibility before test inspection. |

New canonical Sprint 7E training runs:

| Run ID | Base | Masked `target_observation` family | Expected masked columns | Role |
| --- | --- | --- | ---: | --- |
| `S7E_R1_mask_target_sequence` | no context edges | `target_sequence_one_hot` | 115 | Tests whether target-node sequence one-hot is necessary beyond candidate-edge S5F2 sequence/mismatch features. |
| `S7E_R2_mask_experimental_epigenetic` | no context edges | `experimental_epigenetic` | 6 | Tests direct experimental epigenetic scalar contribution. |
| `S7E_R3_mask_computed_nucleosome_aggregates` | no context edges | `computed_nucleosome_aggregates` | 78 | Tests computed nucleosome aggregate value contribution. |
| `S7E_R4_mask_computed_nucleosome_missingness` | no context edges | `computed_nucleosome_missingness` | 13 | Tests missingness/availability indicator contribution. |
| `S7E_R5_mask_all_nonsequence_context` | no context edges | `experimental_epigenetic + computed_nucleosome_aggregates + computed_nucleosome_missingness` | 97 | Tests whether target-node sequence alone can preserve the 7D no-context-edge profile. |

Default canonical training runs: **5**. The run list is source-family-defined
and must be frozen before Colab training. Local profiling may describe feature
distributions but must not add, remove, or reorder these canonical runs from
test diagnostics.

Optional / approval-gated only:

| Run ID | Setting | Role |
| --- | --- | --- |
| `S7E_R6_keep_only_nonsequence_context` | no context edges; target sequence masked, nonsequence context active | Complement to R5; run only if predeclared before any 7E training and compute budget allows. |
| `S7E_R7_mask_top_profiled_family` | no context edges; deterministic train/validation-only selected family | Only if a deterministic train/validation-only profiling rule is written before test inspection; not part of default 7E. |

## 8. Interpretation Rules

### `S7E_R1_mask_target_sequence`

If the run remains close to `S7E_REF_NO_CONTEXT_EDGE_GATV2`, target-node
sequence one-hot is not the main direct target-observation signal. That would
be plausible because candidate `S5F2_energy` already carries guide/target
sequence and mismatch information on candidate edges.

If it drops sharply, target-node sequence representation is materially
supporting Graph C GATv2's observation-level behavior.

### `S7E_R2_mask_experimental_epigenetic`

If it drops sharply, direct experimental epigenetic scalars carry important
Graph C signal even though raw experimental epigenetic edge additions did not
improve Graph A in Sprint 5.

If it remains close to reference, experimental epigenetic scalars are not the
main Sprint 7D context-feature driver.

### `S7E_R3_mask_computed_nucleosome_aggregates`

If it drops sharply, computed nucleosome aggregate values are a major direct
target-observation signal. This would align with the project's broader context
motivation but must not be called biological causality.

If it remains close, the direct context gain likely comes from another family
or from feature interactions.

### `S7E_R4_mask_computed_nucleosome_missingness`

If it drops sharply, missingness/availability indicators are predictive under
the split and must be interpreted cautiously because missingness can encode
dataset coverage or assay/source structure, not biological cleavage mechanism.

If it remains close, missingness is not the main driver.

### `S7E_R5_mask_all_nonsequence_context`

If R5 collapses while R1 is stable, nonsequence context is the likely direct
target-observation signal. If R5 remains close to reference, target-node
sequence plus candidate-edge `S5F2_energy` may explain much of the behavior,
and the Sprint 7D all-target-feature mask collapse may reflect removing
sequence plus context together.

## 9. Slice 1 - Local Profiling And Predeclaration

Purpose: produce a traceable feature-family map and descriptive context profile
before Colab training. Slice 1 does not train a model.

Implementation target:

- Add `scripts/analyze_sprint7e_target_context_features.py`.
- Add tests under `tests/test_sprint7e_target_context_features.py`.

Inputs:

- Graph C S5F2 artifact:
  `data/processed/graphs/sprint5b/graph_c_context_observation/`.
- Sprint 7D outputs:
  `outputs/sprint7d/graphc_gatv2_mechanism_ablation.csv`,
  `outputs/sprint7d/diagnostics/graphc_gatv2_mechanism_predictions.csv`,
  `outputs/sprint7d/diagnostics/graphc_gatv2_component_ablation_audit.csv`.
- Optional prior explanation artifacts:
  `outputs/sprint7c/diagnostics/sprint7c_error_transitions.csv`,
  `outputs/sprint7c/diagnostics/sprint7c_per_guide_error_gain.csv`.

Required Slice 1 outputs:

```text
outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_family_map.csv
outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_group_summary.csv
outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_distribution_by_split_label.csv
outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_profile_report.md
outputs/sprint7e/context_feature_profiling/sprint7e_context_feature_profile_manifest.json
outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_missingness.png
outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_distribution.png
```

Slice 1 must:

1. Load Graph C `target_observation_features`.
2. Classify all 212 feature columns into the four source-defined families in
   Section 4.
3. Assert expected counts: 115, 6, 78, 13, total 212.
4. Record source columns and feature-name matching rules.
5. Summarize train/validation/test distribution, missingness indicators, and
   label-conditioned summaries by feature group.
6. Use Sprint 7C/7D predictions only for descriptive diagnostics, not to change
   the canonical run matrix.
7. Write an explicit "run matrix freeze" section stating that Slice 2 run IDs
   are source-family-defined and not selected from test diagnostics.

Slice 1 must not:

- Retrain a model.
- Choose new feature groups from test performance.
- Produce a "best feature family" claim.
- Use target labels or test error transitions to alter Slice 2 run IDs.

## 10. Slice 2 - Colab Subgroup Ablation Training

Purpose: run the predeclared 7E feature-family masks under the frozen Graph C
GATv2 contract.

Implementation targets:

- Add config:
  `configs/sweeps/sprint7e_target_context_subgroup_ablation.yaml`.
- Add runner:
  `scripts/run_sprint7e_target_context_subgroup_ablation.py`.
- Add Colab runner:
  `colab/sprint7e_target_context_subgroup_ablation_runner.ipynb`.
- Add model/training support for a named target-observation feature mask:
  - either mask by target feature family before `target_observation_encoder`,
  - or pass stable feature-name/feature-index masks resolved from
    `data["target_observation"].feature_names`.
- Add tests:
  - feature-family mapping count tests,
  - target-feature mask tensor tests,
  - runner output-contract tests,
  - no-context-edge + feature-mask audit tests.

Model/training behavior:

- Use `GraphCEdgeGATv2`.
- Keep candidate `S5F2_energy` active in attention/message passing and final
  classifier for every 7E run.
- Drop `context_similar_to` edges for every canonical 7E training run.
- Apply the target-feature mask uniformly to train/validation/test views.
- Do not mutate graph artifact parquet files in place.
- Write per-run `target_feature_mask_audit.csv` proving:
  - masked family name,
  - masked column names,
  - masked column count,
  - masked feature absolute sum after masking = 0,
  - unmasked feature absolute sum remains nonzero where expected,
  - context edges used = 0,
  - candidate attention edge attributes remain nonzero,
  - classifier candidate edge attributes remain nonzero.

Required Slice 2 outputs:

```text
outputs/sprint7e/target_context_subgroup_ablation.csv
outputs/sprint7e/target_context_subgroup_ablation_report.md
outputs/sprint7e/target_context_subgroup_ablation_run_manifest.json
outputs/sprint7e/graph_artifact_provenance.json
outputs/sprint7e/diagnostics/target_context_subgroup_predictions.csv
outputs/sprint7e/diagnostics/target_context_subgroup_training_history.csv
outputs/sprint7e/diagnostics/target_context_subgroup_threshold_metrics.csv
outputs/sprint7e/diagnostics/target_context_subgroup_deltas.csv
outputs/sprint7e/diagnostics/target_context_subgroup_mask_audit.csv
outputs/sprint7e/diagnostics/target_context_subgroup_attention_summary.csv
outputs/sprint7e/diagnostics/target_context_subgroup_per_guide_score_summary.csv
outputs/sprint7e/diagnostics/target_context_subgroup_score_deciles.csv
outputs/sprint7e/figures/target_context_subgroup_auprc_comparison.png
outputs/sprint7e/figures/target_context_subgroup_threshold_metrics.png
outputs/sprint7e/figures/target_context_subgroup_pr_curves.png
outputs/sprint7e/figures/target_context_subgroup_roc_curves.png
outputs/sprint7e/figures/target_context_subgroup_score_distributions.png
outputs/sprint7e/figures/target_context_subgroup_attention_by_edge_kind.png
```

Per-run directories must contain:

```text
resolved_config.yaml
runtime.json
training_history.csv
metrics.csv
attention_summary.csv
target_feature_mask_audit.csv
model.pt  # Drive-held / untracked
```

## 11. Colab Runner Requirements

Colab is a runner only. The notebook may:

- mount Drive,
- clone/update `sprint7/gat-gatv2`,
- install/sync with `uv`,
- copy raw data and processed artifacts from Drive,
- build Sprint 5B Graph C S5F2 artifact if missing,
- run the repository script,
- copy outputs back to Drive,
- validate output contract.

The notebook must not:

- define model classes,
- implement masking logic,
- implement metric/report logic,
- choose run IDs interactively,
- edit configs based on intermediate results.

Expected command:

```bash
uv run python scripts/run_sprint7e_target_context_subgroup_ablation.py \
  --config configs/sweeps/sprint7e_target_context_subgroup_ablation.yaml \
  --run-id "$RUN_ID"
```

## 12. Output Interpretation Rules

Primary report logic:

- AUPRC remains the primary metric.
- AUPRC below `0.900705` indicates collapse below positive-prevalence floor.
- Specificity, MCC, macro F1, TN/FP/FN/TP describe rare-negative threshold
  behavior only.
- Compare each 7E run primarily to `S7E_REF_NO_CONTEXT_EDGE_GATV2`.
- Also report deltas against `S7E_REF_FULL_GRAPH_C_GATV2`, Graph A GCN, Graph C
  GCN, and XGBoost F4.
- Do not call a subgroup "biologically causal".
- Do not call a subgroup the "best model" unless AUPRC supports it under the
  primary metric and the no-test-tuning contract.

Expected conclusion shapes:

- If masking one family sharply degrades AUPRC/MCC/specificity/TN, that family
  is a plausible direct target-observation mechanism under this split.
- If masking all nonsequence context degrades but no single family does, the
  signal may be distributed or interaction-driven.
- If target sequence masking degrades more than nonsequence context masking,
  the Sprint 7D "context feature" collapse may partly reflect target-node
  sequence representation rather than epigenetic/nucleosome context.
- If no subgroup mask degrades as strongly as the all-target-feature mask from
  Sprint 7D, the direct target-observation signal is likely distributed across
  multiple families.

## 13. Tests And Validation

Required local checks before committing implementation:

```bash
uv run pytest tests/test_sprint7e_target_context_features.py -q
uv run pytest tests/test_sprint7b_gatv2_model.py tests/test_sprint7d_gatv2_mechanism_runner.py -q
uv run ruff check src/crispr_gnn/models/gat.py src/crispr_gnn/training/gcn.py scripts/run_sprint7e_target_context_subgroup_ablation.py scripts/analyze_sprint7e_target_context_features.py tests/test_sprint7e_target_context_features.py
git diff --check
```

Runner smoke tests should monkeypatch training, as Sprint 7B/7D runner tests do.
Do not start full training locally unless the user explicitly asks and local
GPU/artifacts are available.

## 14. Acceptance Criteria

Sprint 7E is ready to run when:

- The feature-family map classifies exactly 212 target-observation columns.
- The run matrix has exactly the 5 canonical 7E trained runs unless the user
  explicitly approves optional runs before training.
- All 7E canonical runs keep Graph C GATv2, `S5F2_energy`, weighted BCE, split,
  seed, checkpoint policy, and threshold policy fixed.
- All canonical 7E runs drop `context_similar_to` edges.
- Per-run audits prove the intended target feature family was masked and that
  candidate `S5F2_energy` remains active in both attention and classifier.
- Output contract includes CSV, report, manifest, provenance, diagnostics, and
  figures.
- Local tests and lint pass.
- Colab notebook contains runner commands only.
- Returned outputs exclude committed `model.pt` checkpoints and `.DS_Store`.

Sprint 7E is complete when:

- Returned Colab outputs are copied locally under `outputs/sprint7e/`.
- The final report interprets results under AUPRC-primary, no-test-tuning,
  single-seed boundaries.
- The plan can be moved to `docs/exec-plans/completed/` only after validated
  outputs and final report are present.

## 15. Deferred Work

Do not include these in Sprint 7E:

- CRISPR-Net-style CNN/RNN sequence encoder.
- Larger target context encoder variants.
- GATv2 head/dropout/hidden-dimension search.
- Hybrid tabular-GNN or late-fusion XGBoost/GNN models.
- Rebuilt context topology per subgroup.
- Multi-seed/paired-difference robustness.
- External validation datasets.

Natural follow-ups after Sprint 7E:

- Sprint 7F / model-improvement sprint: target context encoder or edge encoder
  improvements based on the 7E subgroup result.
- Sprint 8 robustness: multi-seed, guide-level bootstrap, paired differences,
  and confidence intervals for the stabilized candidate model(s).
