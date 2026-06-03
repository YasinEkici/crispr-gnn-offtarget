# Execution Plan: Sprint 5 Epigenetic Feature Ablation

## 1. Goal

Sprint 5 is the project's main biological novelty experiment: measure how much
epigenetic and computed nucleosome feature families add under the frozen
guide-level evaluation contract.

Sprint 5 will:

- Run a fixed-topology Graph A GCN feature ladder ablation.
- Keep labels, split, evaluation universe, graph topology, target-node
  semantics, seed policy, checkpoint policy, and threshold policy fixed.
- Vary only the candidate-edge feature bundle across the primary ablation.
- Produce consolidated result tables, diagnostics, figures, provenance records,
  and a Markdown report.
- Use Google Colab only as a runner for full GPU runs, following the Sprint 4
  Drive returned-output pattern.

Sprint 5 will not:

- Use Graph C as the primary feature ablation model.
- Tune feature sets, hyperparameters, thresholds, or model variants from test
  diagnostics.
- Claim reproduction of Mak et al. 2022 CA/regression results.
- Move row-varying context features onto shared physical target nodes.
- Introduce GAT/GATv2, GraphSAGE, R-GCN, HGT, or edge-aware message passing as
  the primary Sprint 5 scope.

## 2. Inputs

- Dataset: Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome data.
- Dataset config: `configs/data/mak2022.yaml`.
- Primary label: Scheme A, `int(cleavage_freq > 1e-5)`.
- Locked split manifest: `outputs/splits/sprint2_guides.json`.
- Locked split ID: `sprint2_main_seed42`.
- Sprint 2 strongest reference baseline: `xgboost_unweighted / F4`.
- Sprint 4 primary graph baseline: `gcn_graph_a`.
- Sprint 3 graph artifact directory: `data/processed/graphs/sprint3/`.
- Sprint 3 handoff report: `outputs/sprint3/graph_schema_report.md`.
- Sprint 4 comparison report:
  `outputs/sprint4/gcn_sprint4_comparison_report.md`.
- Policy sources: `docs/EVALUATION_PROTOCOL.md`, `docs/FEATURE_PARSING.md`,
  `docs/LABEL_SCHEMES.md`, `docs/DECISIONS.md`, and
  `docs/PROJECT_CONTEXT.md`.

Reference metrics under the locked test universe:

| Model | Feature/schema | Test AUPRC | Test MCC | Test TN/FP/FN/TP |
| --- | --- | ---: | ---: | --- |
| `xgboost_unweighted` | `F4` | `0.992522` | `0.345198` | `38/131/21/1512` |
| `gcn_graph_a` | Graph A, `S1_pair+F1` | `0.966287` | `0.300781` | `26/143/11/1522` |
| `gcn_graph_b` | Graph B, `S1_pair+F1` | `0.966570` | `0.126559` | `3/166/0/1533` |
| `gcn_graph_c` | Graph C context observation | `0.961586` | `0.453738` | `43/126/5/1528` |

The locked test positive rate is approximately `0.900705`, and must appear in
Sprint 5 AUPRC interpretation.

## 3. Frozen Evaluation Contract

Sprint 5 must preserve the Sprint 2/Sprint 3/Sprint 4 contract:

- Use Scheme A exactly: `cleavage_freq > 1e-5`.
- Exclude NaN `cleavage_freq` rows from supervised labels.
- Retain negative `cleavage_freq` values as below-threshold labels.
- Retain `cleavage_freq > 1` rows as positive labels without clipping.
- Reuse `sprint2_main_seed42`.
- Assert guide-disjoint train, validation, and test splits.
- Keep headline train, validation, and test measured-only.
- Exclude `experiment_id=18` from headline train, validation, and test.
- Do not introduce `measured=0` rows into headline results.
- Fit imputation and scaling on training rows only.
- Use validation only for checkpoint and threshold selection.
- Use test only for final reporting.
- Use AUPRC as the primary metric.
- Report secondary metrics including AUROC, binary F1, macro F1, MCC,
  TN/FP/FN/TP, specificity, Precision@K, recall at fixed FPR, per-guide,
  per-genome, and cell-line breakdowns where possible.

## 4. Primary Model Decision

Sprint 5 primary ablation uses Graph A fixed topology.

Reason:

- Graph A topology can remain unchanged while candidate-edge feature bundles
  vary.
- Graph A keeps row-varying sequence, mismatch, energy, experimental
  epigenetic, and computed nucleosome inputs on candidate-pair edge features.
- Graph A physical target nodes remain featureless zero/type representations,
  so physical target identity, coordinates, row IDs, and context values do not
  become predictive target-node tensors.
- This isolates feature-family contribution more cleanly than Graph C.

Graph C is secondary only.

Reason:

- Graph C context-similarity topology is constructed from context features.
- A lower-context Graph C run would still expose context through fixed
  context-similarity edges unless topology is rebuilt.
- Rebuilding Graph C topology per feature set changes the experiment from a
  pure feature ablation into a feature-plus-topology ablation.
- Graph C also changes target semantics from shared physical targets to
  feature-bearing target observations.

Graph B is not part of the primary Sprint 5 ablation. It remains a Sprint 4
bounded topology control and may be referenced as prior context.

## 5. Feature Set Ladder

Sprint 5 uses new names instead of overloading Sprint 2 `F1`-`F4`.

Existing `S1_pair` includes guide one-hot, target one-hot, and an aligned
mismatch channel. It is not strict sequence-only. Sprint 5 must implement a
true sequence-only view for `S5F0`.

| ID | Slug | Contents |
| --- | --- | --- |
| `S5F0_SEQ` | `seq_only` | Aligned guide and target sequence channels only; no explicit mismatch channel, no engineered mismatch columns, no energy, no epigenetic/context features. |
| `S5F1_SEQ_MIS` | `seq_mismatch` | `S5F0` plus explicit aligned mismatch channel and engineered mismatch/sequence-pair features. |
| `S5F2_SEQ_MIS_ENERGY5` | `seq_mismatch_energy5` | `S5F1` plus `energy_1` through `energy_5`. |
| `S5F3_SEQ_MIS_ENERGY5_EPI6` | `seq_mismatch_energy5_epi6` | `S5F2` plus six experimental epigenetic scalar features. |
| `S5F4_SEQ_MIS_ENERGY5_EPI6_COMP13_AGG` | `seq_mismatch_energy5_epi6_comp13_agg` | `S5F3` plus aggregated computed nucleosome features and missingness indicators. |
| `S5F5_SEQ_MIS_ENERGY5_EPI6_COMP13_AGG_POS23` | `seq_mismatch_energy5_epi6_comp13_agg_pos23` | `S5F4` plus full `13 * 23 = 299` position-resolved computed nucleosome values. |

Optional later sensitivity:

- `S5F5B_SEQ_MIS_ENERGY5_EPI6_COMP13_POS23_NOAGG`: position-resolved computed
  features without aggregate scalars.

This optional set must not be added after looking at test results. If used, it
must be declared before the run batch begins.

Sprint 5B secondary sensitivity:

- `GraphCContext+S5F2_energy`: fixed Graph C context-similarity topology and
  target-observation context node semantics, with the candidate-edge feature
  table changed to Sprint 5 `S5F2_energy`.

This is not a primary feature ablation. Graph C already carries context through
`target_observation_features` and `context_similar_to` edges, so Sprint 5B can
only be interpreted as a narrow sensitivity comparing the Sprint 5 energy-heavy
edge setting against the established Graph C context representation.

## 6. Required Repository Changes

Planning and documentation:

- `docs/exec-plans/active/005-sprint5-epigenetic-ablation.md`
- `docs/DECISIONS.md`
- `docs/COMMANDS.md`
- `docs/PROJECT_CONTEXT.md` after Sprint 5 is complete
- `colab/README.md`

Likely configs:

- `configs/experiments/gcn_graph_a_sprint5_ablation.yaml`
- `configs/sweeps/sprint5_graph_a_feature_ablation.yaml`

Likely source/script changes:

- Add Sprint 5 feature registry/builders under `src/crispr_gnn/features/`
  following existing module style.
- Extend Graph A feature-table generation or materialization so S5F0-S5F5 edge
  tensors can be loaded while preserving Graph A topology.
- Add or extend GCN training orchestration so a declared feature ladder can run
  without duplicating model/training logic.
- Add Sprint 5 reporting/plotting support using existing evaluation and plotting
  patterns where possible.
- Add a repository-owned Colab runner notebook under `colab/` that invokes repo
  commands only.

Candidate scripts:

- `scripts/build_sprint5_features.py`
- `scripts/run_sprint5_feature_ablation.py`
- `scripts/validate_sprint5_outputs.py`

Use fewer scripts if existing `scripts/train.py` can cleanly dispatch the same
responsibilities without duplicating logic.

Implemented mapping:

- Feature registry/builders: `src/crispr_gnn/features/sprint5.py`
- Graph A Sprint 5 artefact builder:
  `scripts/build_sprint5_graph_a_features.py`
- Feature-ablation sweep runner:
  `scripts/run_sprint5_feature_ablation.py`
- Sweep config:
  `configs/sweeps/sprint5_graph_a_feature_ablation.yaml`
- Colab runner:
  `colab/sprint5_graph_a_feature_ablation_runner.ipynb`
- Metric/reporting extensions:
  `src/crispr_gnn/evaluation/metrics.py`,
  `src/crispr_gnn/evaluation/diagnostics.py`, and
  `src/crispr_gnn/evaluation/plots.py`
- Sprint 5B Graph C energy-sensitivity artefact builder:
  `scripts/build_sprint5b_graph_c_energy_features.py`
- Sprint 5B Graph C energy-sensitivity config:
  `configs/sweeps/sprint5b_graph_c_energy_sensitivity.yaml`
- Sprint 5B Colab runner:
  `colab/sprint5b_graph_c_energy_sensitivity_runner.ipynb`

## 7. Output Contract

Local tracked/scientific outputs:

```text
outputs/sprint5/graph_a_primary/
  sprint5_graph_a_feature_ablation_results.csv
  sprint5_graph_a_feature_ablation_report.md
  sprint5_graph_a_feature_catalog.md
  sprint5_graph_a_run_manifest.json
  diagnostics/
  figures/
  runs/
```

Diagnostics should include:

```text
sprint5_graph_a_predictions_all_feature_sets.csv
sprint5_graph_a_metrics_by_feature_set.csv
sprint5_graph_a_delta_metrics.csv
sprint5_graph_a_fixed_threshold_metrics.csv
sprint5_graph_a_confusion_matrices.csv
sprint5_graph_a_score_deciles.csv
sprint5_graph_a_per_guide_metrics.csv
sprint5_graph_a_per_genome_metrics.csv
sprint5_graph_a_per_cell_line_metrics.csv
sprint5_graph_a_feature_missingness_by_split.csv
sprint5_graph_a_preprocessing_audit.csv
sprint5_graph_a_parameter_counts.csv
```

Figures should be limited to report-useful visualizations. Avoid decorative or
redundant plots; every figure must answer a specific interpretation question.

Required figures:

```text
sprint5_graph_a_auprc_ablation.png
sprint5_graph_a_confusion_matrices.png
sprint5_graph_a_pr_curves.png
sprint5_graph_a_roc_curves.png
sprint5_graph_a_training_curves.png
sprint5_graph_a_decile_lift.png
sprint5_graph_a_per_genome_metrics.png
sprint5_graph_a_per_cell_line_metrics.png
sprint5_graph_a_feature_missingness.png
```

Optional interpretation figures, only if implemented cleanly without broadening
the sprint:

```text
sprint5_graph_a_sequence_position_sensitivity.png
sprint5_graph_a_computed_position_sensitivity.png
```

Sprint 5B secondary outputs:

```text
outputs/sprint5b/graph_c/
  gcn_graph_c_results.csv
  gcn_graph_c_report.md
  diagnostics/
  figures/
  <run_id>/
```

Sprint 5B returned outputs should use a Drive folder named after the run ID,
for example:

```text
returned_outputs/sprint5b_graph_c_energy_sensitivity_seed42_<timestamp>/
```

Run-specific artifacts:

```text
outputs/sprint5/graph_a_primary/runs/<feature_set>/<run_id>/
  resolved_config.yaml
  runtime.json
  graph_artifact_provenance.json
  preprocessing_provenance.json
  training_history.csv
  metrics.json
  model.pt
```

`model.pt`, copied graph artifacts, raw data, caches, and Drive-local folders
must remain untracked.

Drive returned-output structure should follow the observed Sprint 4 pattern:

```text
/content/drive/MyDrive/crispr_gnn_offtarget/returned_outputs/
  sprint5_graph_a_feature_ablation_seed42_<yyyymmdd>/
    diagnostics_sprint5_graph_a_feature_ablation/
    figures_sprint5_graph_a_feature_ablation/
    sprint5_graph_a_feature_ablation_seed42_<yyyymmdd>/
    sprint5_graph_a_feature_ablation_results.csv
    sprint5_graph_a_feature_ablation_report.md
```

The nested run directory stores provenance, resolved configs, runtime records,
training histories, and checkpoints.

## 8. Colab Runner Workflow

Colab is a runner only. Notebook cells may clone/update the repo, mount Drive,
sync dependencies, copy artifacts, call repository commands, copy outputs back
to Drive, and verify returned files. They must not implement model classes,
feature builders, preprocessing, evaluation, plotting, or scientific decisions.

Runner outline:

```bash
pip install uv
git clone https://github.com/YasinEkici/crispr-gnn-offtarget.git crispr-gnn-offtarget
cd crispr-gnn-offtarget
git checkout <approved-sprint5-commit>
uv sync
uv run python -c "import torch, torch_geometric; print(torch.__version__); print(torch_geometric.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

mkdir -p data/processed/graphs
cp -r /content/drive/MyDrive/crispr_gnn_offtarget/data/processed/graphs/sprint3 data/processed/graphs/

uv run python scripts/validate_graph_artifacts.py \
  --artifact-dir data/processed/graphs/sprint3 \
  --approved-source drive_sprint3_handoff \
  --output outputs/sprint5/graph_a_primary/provenance/graph_artifact_provenance.json

uv run python scripts/run_sprint5_feature_ablation.py \
  --config configs/sweeps/sprint5_graph_a_feature_ablation.yaml

uv run python scripts/validate_sprint5_outputs.py \
  --output-dir outputs/sprint5/graph_a_primary
```

Copy-back uses the Sprint 4 pattern, with one top-level returned-output folder
per Colab run batch.

## 9. Tests

Feature set tests:

- Assert S5F0 contains no explicit mismatch channel or engineered mismatch
  columns.
- Assert S5F1 adds explicit mismatch representation and engineered mismatch
  features.
- Assert S5F2 adds exactly the five binding-energy features.
- Assert S5F3 adds exactly the six experimental epigenetic scalar features.
- Assert S5F4 adds aggregate computed nucleosome features plus missingness
  indicators.
- Assert S5F5 adds exactly 299 position-resolved computed columns.
- Assert column order is deterministic.
- Assert forbidden predictive/reporting columns are absent.

Preprocessing tests:

- Assert imputation/scaling fit uses train rows only.
- Assert validation/test transforms reuse train-fitted statistics.
- Assert missingness indicators are generated before imputation.
- Assert preprocessing provenance records feature columns and split missingness.

Graph A invariant tests:

- Assert S5F0-S5F5 preserve Graph A schema, node counts, candidate edge IDs,
  labels, split masks, measured-only universe, `experiment_id=18` exclusion,
  and visibility policy.
- Assert only edge-feature tensor width changes.

Leakage/config tests:

- Reject random-edge final split.
- Reject wrong split ID or label scheme.
- Reject test-set checkpoint or threshold selection.
- Reject measured-zero validation/test rows.
- Reject `experiment_id=18` in headline rows.
- Reject Graph C as an unlabeled primary Sprint 5 ablation.

Output tests:

- Assert required CSV, report, diagnostics, figures, resolved configs, runtime
  records, training histories, and provenance files exist for completed runs.

## 10. Risks And Mitigations

Graph C confounds feature ablation:

- Mitigation: use Graph A fixed topology as primary. Any Graph C run must be
  secondary and explicitly labeled as fixed-full-context topology or
  feature-matched topology.

Strict sequence-only is not current `S1_pair`:

- Mitigation: implement S5F0 without the aligned mismatch channel. Use S5F1 for
  explicit mismatch representation.

Position-resolved computed features add many dimensions:

- Mitigation: keep model/training policy fixed, report parameter counts, and do
  not tune only the high-dimensional feature set. If instability forces a
  revised config, rerun all feature sets under the revised predeclared config.

Train/test leakage through preprocessing:

- Mitigation: fit preprocessing on train only per feature set and write
  `preprocessing_provenance.json`.

Test-set tuning:

- Mitigation: predeclare all feature sets and report every completed run.
  Test diagnostics are final interpretation only.

High positive prevalence:

- Mitigation: include positive prevalence, TN/FP/FN/TP, specificity,
  MCC, macro F1, binary F1, precision/recall diagnostics, and AUPRC. AUPRC
  remains the primary comparison metric.

Edge features may not participate in GCN message passing:

- Mitigation: state that Sprint 5 measures feature-family contribution under
  the Sprint 4 Graph A GCN edge-classification architecture. Do not claim that
  epigenetic features improve message passing unless the architecture is changed
  to use edge features inside propagation.

## 11. Acceptance Criteria

- Sprint 5 plan is written before implementation.
- `docs/DECISIONS.md` records Graph A as the primary fixed-topology ablation.
- S5F0-S5F5 are explicitly defined and tested.
- All headline runs use `sprint2_main_seed42`.
- Headline train, validation, and test rows are measured-only and exclude
  `experiment_id=18`.
- NaN `cleavage_freq` rows are excluded from supervised labels.
- Imputation/scaling are train-only.
- Graph A topology, candidate edge IDs, labels, split masks, and visibility are
  identical across feature sets.
- Checkpoint and threshold selection use validation only.
- No test diagnostic changes feature sets, topology, hyperparameters, seeds,
  thresholds, or reporting choices.
- All six feature sets are run, or skipped/failed runs are clearly documented
  with technical reasons.
- Consolidated report includes positive prevalence, Sprint 4 Graph A, Graph B
  and Graph C as prior context where useful, and `xgboost_unweighted / F4`.
- Consolidated report includes macro F1, confusion matrices, TN/FP/FN/TP, and
  specificity for every feature set, while keeping AUPRC as the primary metric.
- Figures are meaningful and bounded: no extra plots unless they directly
  support the feature-ablation, thresholded-classification, subgroup, or
  missingness interpretation.
- Report does not claim Mak et al. reproduction.
- Required diagnostics, figures, provenance, configs, runtime records, and
  training histories exist.
- Colab notebook remains runner-only.
- Output validation passes before results are treated as final.

## 12. Implementation Slices

### Slice 0: Planning And Decision Freeze

- Finalize this active execution plan.
- Update `docs/DECISIONS.md` with Graph A primary, Graph C secondary, S5F0-S5F5
  naming, position-resolved feature artifact policy, and no Mak reproduction
  claim.
- Update `docs/COMMANDS.md` with planned Sprint 5 commands.

Exit gate: feature sets, graph policy, output contract, and no-test-tuning rule
are documented before code changes.

### Slice 1: Feature Registry And Preprocessing

- Implement S5F0-S5F5 feature builders.
- Implement position-resolved computed nucleosome feature builder.
- Implement train-only preprocessing/provenance.
- Generate Sprint 5 feature catalog.

Exit gate: feature-set and preprocessing tests pass. No model training yet.

### Slice 2: Graph A Materialization With Variable Feature Bundles

- Extend Graph A loader/materializer to expose S5F0-S5F5 candidate-edge tensors
  while preserving topology and masks.
- Add Graph A invariant tests.
- Run a tiny CPU smoke training path.

Exit gate: topology invariance tests pass across all feature sets.

### Slice 3: Reporting And Output Contract

- Implement consolidated result tables, diagnostics, figures, Markdown report,
  and output validation.
- Add output contract tests.

Exit gate: mocked or smoke outputs produce the required Sprint 5 directory
structure.

### Slice 4: Colab Runner Preparation

- Create runner-only Sprint 5 Colab notebook/instructions.
- Preserve base configs and generate run-specific resolved configs.
- Include graph artifact provenance gate and returned artifact checklist.

Exit gate: runner workflow is documented; no full result is claimed.

### Slice 5: Full Graph A Primary Ablation

- Run all six Graph A feature sets on Colab GPU through repository commands.
- Copy returned outputs to Drive.
- Validate returned artifacts locally before integrating tracked reports.

Exit gate: all six runs are present or failures are documented; output
validation passes; no test-driven reruns occurred.

### Slice 6: Optional Graph C Secondary Sensitivity

- Only after Graph A primary ablation is closed, run one predeclared Graph C
  secondary policy if still useful.

Exit gate: Graph C report explicitly states it is not a clean primary feature
ablation.

### Slice 7: Final Sprint 5 Report Freeze

- Finalize reports, result CSVs, diagnostics, figures, and status docs.
- Move this plan to `docs/exec-plans/completed/` after acceptance.

Final interpretation must say:

```text
Under the locked Scheme A, guide-level, measured-only, experiment_id=18-excluded
protocol and fixed Graph A topology, feature family X changed AUPRC by Y
relative to feature set Z. This is not Mak et al. reproduction and did not use
test-driven model selection.
```
