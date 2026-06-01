# Execution Plan: Sprint 4 GCN Baseline + Colab Training + Visual Reporting

## 1. Goal

Sprint 4 is the first graph-model training and evaluation sprint. It consumes the
validated Sprint 3 typed graph artifacts without redefining their schemas,
visibility policy, labels, or split assignments, and it implements the first
literature-aligned GCN edge-classification/link-prediction baseline.

Sprint 4 will:

- Materialize validated Graph A, Graph B, and Graph C typed artifacts into a
  model-facing representation with fail-fast contract checks.
- Train and evaluate Graph A first as the minimal physical-target GCN path.
- Train and evaluate Graph C second as the primary context-enriched graph
  comparison.
- Run Graph B only as one bounded secondary control after the Graph A pipeline
  is stable.
- Execute full training on Google Colab GPU infrastructure through repository
  commands only.
- Produce reproducible results, diagnostics, and report-ready figures.
- Compare graph results primarily against the frozen
  `xgboost_unweighted / F4` Sprint 2 baseline under the same evaluation
  contract.

Sprint 4 will not:

- Implement GAT, GATv2, GraphSAGE, heterogeneous message-passing sweeps, or
  other advanced graph architectures.
- Expand into Sprint 5 systematic context-feature ablation.
- Expand into Sprint 6 systematic imbalance-method comparisons.
- Claim a stronger predictive baseline unless a graph model improves on the
  strong Sprint 2 F4 XGBoost result under the locked contract.

Vinodkumar et al. motivate an initial GCN/link-prediction model and reporting of
iterative training behavior. Their evaluation protocol is not copied: final
Sprint 4 reporting remains guide-level and leakage-controlled rather than
random-edge based.

## 2. Inputs

- Primary dataset: Mak et al. 2022 crisprSQL-derived
  epigenetic/nucleosome dataset.
- Dataset config: `configs/data/mak2022.yaml`.
- Primary label: Scheme A, `int(cleavage_freq > 1e-5)`.
- Locked split manifest: `outputs/splits/sprint2_guides.json`.
- Locked split ID: `sprint2_main_seed42`.
- Sprint 2 strongest reference baseline: `xgboost_unweighted / F4`.
- Sprint 2 feature/preprocessing definitions: F1-F4 and S1, as documented in
  `outputs/features/sprint2_feature_catalog.md` and implemented under
  `src/crispr_gnn/features/`.
- Sprint 3 schema config: `configs/sweeps/graph_schema_ablation.yaml`.
- Sprint 3 canonical handoff report: `outputs/sprint3/graph_schema_report.md`.
- Sprint 3 typed graph artifact location:
  `data/processed/graphs/sprint3/`.
- Policy sources: `CRISPR_GNN_PROJECT_PLAN.md`,
  `docs/EVALUATION_PROTOCOL.md`, `docs/DECISIONS.md`,
  `docs/PROJECT_CONTEXT.md`, and `PROJECT_FOLDER_STRUCTURE.md`.
- Colab and command conventions: `docs/COMMANDS.md`, `notebooks/README.md`,
  and `colab/README.md`.

Frozen Sprint 2 reference result:

| Reference model | Feature bundle | Test AUPRC | Test AUROC | Test MCC |
| --- | --- | ---: | ---: | ---: |
| `xgboost_unweighted` | `F4` | approximately `0.992522` | approximately `0.938416` | approximately `0.345198` |

The locked test positive rate is approximately `0.9007`; that prevalence is a
required reference for interpreting all Sprint 4 AUPRC and classification
results.

## 3. Frozen Sprint 2 Contract Sprint 4 Must Preserve

- Preserve Scheme A exactly:
  `label = int(cleavage_freq > 1e-5)`.
- Exclude NaN `cleavage_freq` rows from supervised graph-model evaluation.
- Retain negative `cleavage_freq` values as below-threshold labels while
  retaining the documented raw-label quality caveat.
- Retain `cleavage_freq > 1` rows as positive labels without clipping for this
  binary-classification task.
- Reuse the locked guide-level split `sprint2_main_seed42`.
- Assert that no guide occurs in more than one of train, validation, and test.
- Keep the main train, validation, and test evaluation universe measured-only.
- Require validation and test rows to remain `measured=1`.
- Exclude `experiment_id=18` from main Sprint 4 evaluation.
- Do not introduce `measured=0` rows into headline model results.
- Do not use test diagnostics, labels, scores, or figures to choose graph
  schema, model architecture, inputs, epoch count, checkpoint, hyperparameters,
  thresholds, or reporting choices.
- Use AUPRC as the primary metric and report test positive prevalence.
- Report secondary metrics consistently where available: AUROC, F1, MCC,
  Precision@K, Recall at fixed FPR, and confusion matrices.
- Preserve per-guide and per-genome interpretation because the locked split has
  uneven guide sizes and genome composition.
- Compare primarily against `xgboost_unweighted / F4`, not only against weaker
  sequence baselines.

## 4. Frozen Sprint 3 Graph Contract Sprint 4 Must Preserve

Sprint 4 loaders must consume the serialized typed tables and manifests
produced in Sprint 3. They must not rebuild graph topology from raw records,
relocate scientifically meaningful features, or relax strict-inductive
visibility.

### Graph A: Minimal Physical-Target Schema

- Graph A is the minimal physical-target graph schema.
- `sgRNA` nodes use the established guide key and permitted guide-only
  representation.
- `physical_target_site` nodes retain genome-aware coordinate identity:
  `genome + target_chr + target_start + target_end + target_strand`.
- Candidate-pair relations are keyed by source row `id`.
- Labels and split assignments remain candidate-edge properties.
- Row-varying target sequence, mismatch, binding energy, experimental
  epigenetic context, and computed nucleosome context must not be moved onto
  shared physical target nodes.
- Model-facing materialization must preserve this placement without silently
  enriching physical target nodes.
- The default Sprint 4 Graph A baseline must not introduce learned
  per-target-ID embeddings for `physical_target_site` nodes. If featureless
  physical targets need a model-facing representation, use a documented
  zero/type or equivalent strategy that does not encode target identity,
  coordinates, row IDs, or row-varying context as predictive tensors.

### Graph B: Guide-Similarity Bounded Control

- Graph B inherits Graph A candidate relations, features, and physical target
  representation.
- It adds only deterministic, label-free guide-sequence similarity relations.
- Validation and test held-out guides may connect only to training guides in
  strict-inductive evaluation views.
- Graph B is one bounded secondary control; Sprint 4 must not use it as an
  open-ended schema-sweep track.

### Graph C: Context-Observation Schema

- Graph C uses `target_observation` nodes keyed by source row `id`.
- Context features remain attached at the observation level according to the
  Sprint 3 artifact contract rather than being collapsed onto physical target
  coordinates.
- Candidate edges preserve Scheme A labels and locked split membership.
- Context similarity remains label-free, deterministic, and built only from
  allowed context inputs with preprocessing fit on training observations.
- Validation observations may connect only to training observations.
- Test observations may connect only to training observations and must never
  use validation observations as neighbors.
- Equal-distance similarity ties retain deterministic stable source-row-ID
  resolution from Sprint 3.
- Graph C changes both topology and target semantics relative to Graph A. Any
  Graph A versus Graph C result must be described accordingly.

Mak-derived row-varying context cannot be collapsed onto shared physical
target nodes because assay/context observations associated with a coordinate
are not guaranteed to share the same feature values. Graph C exists to
represent that observation-level context explicitly.

### Expected Artifact Counts Before Training

The model loader must assert the report and artifact manifests before any
training run:

| Schema | Node/relation count | Expected value |
| --- | --- | ---: |
| Graph A | `sgRNA` | 150 |
| Graph A | `physical_target_site` | 9880 |
| Graph A | `candidate_pair` | 11446 |
| Graph B | `sgRNA` | 150 |
| Graph B | `physical_target_site` | 9880 |
| Graph B | `candidate_pair` | 11446 |
| Graph B | `sequence_similar_to` | 1208 |
| Graph C | `sgRNA` | 150 |
| Graph C | `target_observation` | 11446 |
| Graph C | `candidate_pair` | 11446 |
| Graph C | `context_similar_to` | 91754 |

Expected supervised candidate-edge split counts:

| Split | Candidate edges |
| --- | ---: |
| train | 8010 |
| validation | 1734 |
| test | 1702 |
| total | 11446 |

No earlier nondeterministic Graph C count is an acceptable input to Sprint 4.

## 5. Scope

- Decide the minimum approved graph-model dependency and materialization
  approach.
- Implement typed-artifact loading into the selected training representation,
  provisionally PyTorch Geometric `HeteroData`.
- Implement strict-inductive train, validation, and test graph views at loader
  level with manifest-backed assertions.
- Implement a minimal GCN edge classifier/link predictor.
- Train Graph A first, then Graph C, with Graph B allowed only as a bounded
  control after Graph A is stable.
- Define configuration-driven training, deterministic seeding, checkpointing,
  early stopping, validation-only threshold selection, and result recording.
- Define a Google Colab GPU runner workflow that invokes repository code only.
- Produce numerical outputs, diagnostics, reports, graph-data sanity
  visualizations, and model-result visualizations.
- Add focused model-loader, leakage, model, training-smoke, evaluation,
  reporting, visualization, configuration, and runner-boundary tests.

## 6. Out Of Scope

- Rebuilding or changing Sprint 3 graph schemas or graph artifacts.
- Rebuilding, balancing, or replacing the locked Sprint 2 split.
- Changing Scheme A labels or the main supervised row universe.
- Adding `measured=0` rows to headline evaluation.
- Reintroducing `experiment_id=18` into headline evaluation.
- Test-set-driven graph, model, feature, epoch, threshold, or hyperparameter
  selection.
- GAT, GATv2, GraphSAGE, HeteroConv, R-GCN, HGT, or other advanced graph
  architectures.
- Sprint 5 systematic epigenetic/context feature ablation or full
  context-attribution analysis.
- Sprint 6 systematic imbalance-method evaluation.
- Scheme B paper reproduction or Scheme C robustness analysis.
- CRISPRoffT external validation.
- Full 299-dimensional position-resolved computed nucleosome experimentation
  unless separately reopened later.
- Core model, preprocessing, evaluation, or plotting logic implemented only in
  Colab notebooks.

## 7. Graph Materialization And Dependency Decision

### Proposed Decision Gate

Sprint 4 implementation should adopt PyTorch Geometric only after approving
the minimum dependency and Colab/CUDA compatibility approach. The proposed
representation is typed PyG `HeteroData`, subject to that approval, because:

- Sprint 3 already serializes typed node and relation tables.
- Graph A, Graph B, and Graph C require consistent handling of heterogeneous
  relation types.
- A typed representation can retain node/relation identity and visibility
  assertions while providing a direct GCN training interface.

This plan does not add PyTorch Geometric. During implementation, the approved
minimum additions to `pyproject.toml` and `uv.lock` must be recorded and
reviewed before full GPU training.

### Materialization Requirements

The loader/materializer must:

- Read Sprint 3 typed tables and manifests rather than reconstructing topology
  from the raw Mak dataset.
- Validate schema ID, label scheme, split ID, visibility policy, feature
  bundles, preprocessing scope, relation identity, and expected counts before
  exposing training tensors.
- Preserve candidate-edge source-row identity for auditing while preventing raw
  IDs from becoming predictive input tensors.
- Preserve Graph A edge-level placement of row-varying fields.
- Preserve Graph C observation-level feature placement.
- Produce strict-inductive train, validation, and test graph views:
  - Training view: training candidate supervision and training-visible
    auxiliary topology only.
  - Validation view: training-visible topology plus permitted validation
    query/fragments, without test visibility.
  - Test view: training-visible topology plus permitted test query/fragments,
    without validation becoming a test neighbor source.
- Fail fast rather than silently rebuilding, dropping, or adding relations
  when artifacts drift.

### Dependency And Colab/CUDA Policy

- Confirm compatible versions of installed PyTorch, proposed PyTorch
  Geometric packages, Python, and the Colab GPU/CUDA runtime during later
  implementation.
- Record any approved PyG/CUDA installation rule in versioned repository
  documentation, not only in notebook cells.
- Use `uv` as the documented project dependency and execution workflow.
- Determine whether Colab-specific setup requires `colab/README.md` updates;
  add them only if a reproducible GPU workflow needs them.
- Do not accept an undocumented ad hoc `pip install torch-geometric` cell as
  the final workflow.

## 8. Model And Training Design To Implement Later

### Model Role And Schema Sequence

- Implement a minimal GCN edge classifier/link predictor over candidate
  guide-target relations.
- Stabilize the Graph A training/evaluation path first.
- Evaluate Graph C second as the primary context-enriched comparison.
- Run Graph B once, at most, as a bounded guide-similarity control after the
  Graph A pipeline passes contract checks.

### Input Feature Policy

The implemented configuration must explicitly fix the feature bundle used by
each schema before headline training:

- Graph A may consume its permitted guide representation and candidate-edge
  sequence/pair feature bundles without relocating row-varying context to
  physical targets.
- Graph C may consume the corresponding permitted guide/pair inputs together
  with its observation-level context representation and context-similarity
  topology.
- Graph B uses the approved Graph A feature path plus its label-free
  guide-similarity relation only.
- The final report must state exactly whether aligned sequence information,
  engineered pair features, and/or context features were used.
- The Graph A model-facing target-node representation policy must be fixed
  before headline training and recorded with each run. Featureless physical
  targets must remain featureless in the artifact contract; any internal
  model representation must not become a covert target-ID or coordinate
  feature.
- Earlier training slices must emit enough structured metadata for later
  reporting and visualization: feature bundle, target-node representation
  policy, training history, prediction split, validation-selected threshold,
  checkpoint selection source, prevalence, and baseline-reference fields.

Graph A versus Graph C results must not be presented as isolating topology
alone, because Graph C changes target representation and permits
observation-level context inputs.
Any Graph C result must explicitly distinguish Graph A's featureless
physical-target semantics from Graph C's observation-level target semantics.

### Fixed Training Policy

- Use weighted binary cross-entropy as the fixed initial Sprint 4 loss policy,
  consistent with the project plan.
- Do not treat loss choice as an imbalance-method sweep; alternatives belong
  to Sprint 6.
- Keep optimizer, learning rate, hidden dimensions, number of layers, dropout,
  maximum epochs, patience, checkpoint rule, random seeds, and device
  selection configuration-driven.
- Use validation AUPRC for early stopping/checkpoint selection.
- Select any classification threshold using validation data only, following
  the established Sprint 2 evaluation policy.
- Evaluate the test set once for each finalized model/schema run.
- Treat post-run test diagnostics as interpretation outputs only, not
  decision inputs.

### Placeholder Configuration Gate

`configs/experiments/gcn_minimal.yaml` must not contain a placeholder/debug
split rule for headline evaluation. Final runs require `sprint2_main_seed42`.

If a random-edge or reduced debug run is retained for technical smoke testing,
it must be visibly identified as debug-only, excluded from final comparison
tables and figures, and rejected as a headline configuration by validation.

## 9. Colab GPU Runner Workflow

Google Colab is a compute runner, not a source of scientific implementation.

Repository ownership requirements:

- Loader, preprocessing, model, training, evaluation, plotting, and reporting
  logic live under `src/crispr_gnn/` and `scripts/`.
- Experiment parameters live under `configs/`.
- Scientific decisions and command instructions live in versioned
  documentation.
- Notebook cells may invoke repository commands but must not replace them.

The later Colab runner workflow must:

1. Clone or update the repository and check out the approved Sprint 4 branch
   or exact commit.
2. Record the git commit SHA used for each run.
3. Mount Google Drive only as needed for data/artifact input and output
   persistence.
4. Install/sync dependencies using the documented approved workflow.
5. Copy required raw/processed inputs or Sprint 3 typed graph artifacts to
   Colab local disk when needed for fast I/O.
6. Execute repository commands through `uv run ...`.
7. Copy generated checkpoints, logs, metrics, diagnostics, reports, and
   figures back to durable Drive storage.
8. Record the resolved experiment config, seed, device/runtime information,
   graph schema, feature bundle, split ID, and visibility policy.

The Colab runner must not:

- Define model classes.
- Implement scientific preprocessing or topology creation.
- Implement evaluation calculations or final plotting.
- Silently modify labels, split manifests, graph visibility, or experiment
  configuration.
- Serve as the sole record of dependency workarounds.

Potential later Colab-support files, only if required by the selected runner
workflow:

- `colab/README.md`.
- A runner-only notebook such as `notebooks/05_train_gcn_runner.ipynb`, or
  the equivalent location consistent with repository conventions.

## 10. Evaluation And Comparison Policy

- Headline results use only the locked guide-level split
  `sprint2_main_seed42`.
- AUPRC is the primary metric, always interpreted with positive prevalence.
- Report AUROC, F1, MCC, Precision@K, Recall at fixed FPR, and confusion
  matrices where implemented consistently.
- Select thresholds from validation only.
- Do not tune against test outputs.
- Preserve per-guide and per-genome diagnostic reporting.
- Mark any technical debug/random-edge run as debug-only and exclude it from
  headline model performance.

The consolidated Sprint 4 comparison must include:

- Dummy/prevalence baseline.
- `xgboost_unweighted / F4`.
- Relevant Sprint 2 sequence baseline context where useful for interpretation.
- GCN Graph A.
- GCN Graph C.
- GCN Graph B only if the bounded control run is completed.

Interpretation constraints:

- The approximate test positive rate of `0.9007` makes prevalence context
  essential for interpreting AUPRC and F1.
- A graph model that does not beat `xgboost_unweighted / F4` under the frozen
  contract is not a stronger predictive baseline, although it may remain a
  scientifically useful graph ablation.
- An improvement for Graph C cannot automatically be attributed to topology
  because Graph C also changes target semantics and context placement.
- The project keeps its guide-level leakage-controlled evaluation rather than
  reproducing a paper's random-edge evaluation protocol.

## 11. Visualization And Interpretation Contract

Sprint 4 is not complete with only metric CSV files and a Markdown report.
Visualizing both graph data/materialized views and model behavior is required.

### Required Core Figure Outputs

```text
outputs/sprint4/graph_a/figures/gcn_graph_a_graph_schema_auprc_comparison.png
outputs/sprint4/graph_a/figures/gcn_graph_a_pr_curves.png
outputs/sprint4/graph_a/figures/gcn_graph_a_roc_curves.png
outputs/sprint4/graph_a/figures/gcn_graph_a_training_curves.png
outputs/sprint4/graph_a/figures/gcn_graph_a_score_distributions.png
outputs/sprint4/graph_a/figures/gcn_graph_a_confusion_matrices.png
outputs/sprint4/graph_a/figures/gcn_graph_a_decile_lift.png
outputs/sprint4/graph_a/figures/gcn_graph_a_per_genome_metrics.png
outputs/sprint4/graph_a/figures/gcn_graph_a_view_sanity_example.png
```

Conditional interpretation output:

```text
outputs/sprint4/graph_a/figures/gcn_graph_a_sequence_position_sensitivity.png
```

The conditional sequence-position output is required only if the stable Sprint
4 GCN contract consumes an aligned guide-target sequence representation such
as S1 or another explicitly position-aligned sequence bundle.

### Graph-Data Visualization Requirements

- `graph_view_sanity_example.png` must depict a small, readable
  strict-inductive loader/materialized view or local subgraph example.
- The sanity figure must make node and relation types legible and show allowed
  auxiliary visibility for the selected view.
- It must be generated from the materialized loader view or an equivalent
  mocked loader fixture in tests, not by rebuilding topology from raw rows.
- It must label whether the illustrated view is train, validation, or test,
  and captions/report text must state that the visualization is bounded and
  non-exhaustive.
- It must not plot an unreadable full Graph C network.
- It must not be treated as a performance figure.
- When illustrating Graph C, captions/report text must distinguish
  `target_observation` and context-similarity relations from Graph A physical
  target nodes and candidate edges.
- Additional compact graph-structure diagnostics may be implemented only when
  they support loader validation or clear reporting without broadening scope.

### Model Evaluation Visualization Requirements

- `gcn_graph_schema_auprc_comparison.png` must include positive prevalence,
  `xgboost_unweighted / F4`, Graph A, Graph C, and Graph B only if its bounded
  control is run.
- PR and ROC curves must be labeled by graph schema and contain headline
  guide-level results only.
- Training curves must display training and validation behavior for iterative
  GCN fitting; they must not use test tracking for selection.
- Score distributions and decile-lift outputs are diagnostics, not selection
  tools.
- Confusion matrices or fixed-threshold summaries must use
  validation-selected thresholds only; plotting code must not select a
  threshold from test scores.
- Per-genome metrics must accompany aggregate generalization claims.
- Per-guide diagnostic tables or compact figures should be produced where
  practical because guide sample sizes are uneven.
- Smoke or mocked visualization outputs must not be presented as final Sprint
  4 model performance. They are allowed only for testing the reporting path.
- Schema-comparison figures must include `graph_schema` in the plotted
  identity so Graph A, Graph C, and Graph B cannot be visually collapsed into
  one row when model or feature names overlap.

### Interpretation Boundaries

- If aligned sequence inputs are used, position sensitivity may adapt the
  CRISPR-Net position-level occlusion/replacement precedent.
- Kipf and Vinodkumar motivate reporting iterative GCN training behavior, so
  Sprint 4 training curves should show training loss and validation behavior.
  They must not track test performance across epochs.
- Vinodkumar motivates a CRISPR GCN/link-prediction baseline, but its
  random-edge evaluation setup must not appear in Sprint 4 headline figures.
- DeepCRISPR-style saliency motivates interpretable diagnostic views, but
  Circos output is not a core Sprint 4 requirement without an approved
  genome-wide end-user prediction scope.
- Mak-style context distribution and comprehensive SHAP/feature-contribution
  reporting remain primarily Sprint 5 work; Sprint 4 may include only a
  minimal, clearly labeled sanity view if needed.
- Graph C or context-related Sprint 4 visualizations must not become full
  Mak-style context contribution analysis. Systematic context distribution,
  SHAP, feature-contribution, and context-gain-by-condition work remains a
  Sprint 5 boundary.
- Gao/Guan imbalance literature motivates prevalence-aware PR reporting and
  positive-retrieval interpretation; it does not expand Sprint 4 into Sprint
  6 method comparisons.
- Graph C visualizations and comparisons must not describe any improvement as
  topology-only, because Graph C changes both topology and target semantics.
- Weighted BCE is the fixed Sprint 4 starting loss policy. Focal loss,
  SMOTE, sampling methods, and systematic positive-retrieval variability
  analysis remain Sprint 6 work unless a later plan explicitly reopens them.
- Perturbation, attribution, SHAP, and future attention views are model
  interpretation signals, not causal biological evidence.
- Test-based visualizations are interpretation-only and must not affect
  schema, feature, architecture, checkpoint, threshold, or hyperparameter
  choices.
- Debug/random-edge figures, if generated, must be explicitly labeled
  debug-only and excluded from headline figures.

## 12. Required Outputs And Artifact Policy

### Tracked Scientific And Reporting Outputs

Subject to repository artifact conventions, validated Sprint 4 completion
should produce:

```text
outputs/sprint4/graph_a/gcn_graph_a_results.csv
outputs/sprint4/graph_a/gcn_graph_a_report.md
outputs/sprint4/graph_a/figures/*.png
outputs/sprint4/graph_a/diagnostics/*.csv
```

Slice 4A does not produce final scientific results. It may add or update only
repository-owned runner documentation, command references, provenance checks,
or runner-only notebook support. The files above become final Sprint 4
artifacts only after a real manual Slice 4B run and Slice 4C validation.

### Run Outputs Normally Kept Untracked Or Stored In Google Drive

```text
outputs/sprint4/graph_a/<run_id>/
outputs/sprint4/graph_a/<run_id>/model.pt
outputs/sprint4/graph_a/<run_id>/optimizer/
```

Large prediction dumps, Colab-local caches, copied raw data, copied processed
graph Parquet files, checkpoints, and Drive-specific paths must not be added
to version control as scientific reporting artifacts.

### Required Run Provenance

Each full run must record at least:

- Run ID.
- Git commit SHA.
- Resolved configuration.
- Random seed.
- Device/runtime information.
- Sprint 3 graph artifact source path and checksum/provenance summary.
- Graph schema.
- Feature bundle.
- Model-facing target-node representation policy.
- Label scheme.
- Split ID.
- Visibility policy.
- Primary and secondary metrics.
- Positive prevalence.
- Checkpoint selection rule.
- Threshold selection source.
- Training-history artifact path or run record.

Before Graph C or Graph B results are written to
`outputs/sprint4/<schema_label>/gcn_<schema_label>_results.csv`, the result upsert identity must include
`graph_schema` or an equivalent schema-disambiguating field. Graph A, Graph C,
and Graph B rows must not overwrite each other if model name or feature bundle
names overlap.

## 13. Expected Future Implementation Files

The implementation phase should use the smallest coherent file set supported
by current repository patterns. Candidate files are listed here for planning;
not all must be created if an existing module can be extended cleanly.

### Likely Core Implementation Files

- `src/crispr_gnn/graph/pyg_dataset.py` or equivalently named typed
  artifact loader/materializer.
- `src/crispr_gnn/models/gcn.py`.
- `src/crispr_gnn/training/gcn.py`, or a narrowly justified extension of the
  current training package.
- `src/crispr_gnn/evaluation/plots.py`.
- `src/crispr_gnn/evaluation/diagnostics.py`.
- `scripts/train.py`, preserving the existing config-dispatch entry-point
  convention.
- `scripts/evaluate.py` only if a separate evaluation entry point is justified
  by the final output workflow.
- `configs/experiments/gcn_minimal.yaml`.
- An additional Graph C/schema-comparison config only if it makes the fixed
  run contract clearer than config overrides.

### Likely Focused Tests

- `tests/test_graph_loader.py`.
- `tests/test_gcn_model.py`.
- `tests/test_gcn_training_smoke.py`.
- `tests/test_gcn_evaluation_contract.py`.
- `tests/test_gcn_visualization_outputs.py`.
- `tests/test_config_loads.py` updates for final/debug configuration rules.

### Dependency Updates Only After Approval

- `pyproject.toml`.
- `uv.lock`.

### Documentation And Colab Updates Only As Needed

- `docs/DECISIONS.md`.
- `docs/EVALUATION_PROTOCOL.md`.
- `docs/COMMANDS.md`.
- `README.md`.
- `docs/PROJECT_CONTEXT.md`.
- `colab/README.md`.
- A runner-only notebook under `notebooks/` or `colab/`, only if useful for
  executing documented commands on Colab.

### Generated Outputs After Approved Training

- `outputs/sprint4/graph_a/gcn_graph_a_results.csv`.
- `outputs/sprint4/graph_a/gcn_graph_a_report.md`.
- `outputs/sprint4/graph_a/figures/`.
- `outputs/sprint4/graph_a/diagnostics/`.

## 14. Required Tests And Validation Checks

### Loader And Materialization Tests

- Load Sprint 3 artifacts without altering manifest identity or relation
  definitions.
- Assert `split_id == sprint2_main_seed42`.
- Assert label scheme is Scheme A.
- Assert supervised candidate counts: train `8010`, validation `1734`, test
  `1702`, total `11446`.
- Assert Graph A, Graph B, and Graph C node/relation counts equal their
  manifests.
- Assert materialized feature placement matches each Sprint 3 schema.
- Assert strict-inductive visibility remains intact after model-facing
  loading/materialization.
- Assert validation and test candidate edges cannot enter the supervised
  training view.
- Assert Graph B held-out guides link only to training guides.
- Assert Graph C validation and test observations use only training context
  neighbors and that test views do not use validation observations.

### Leakage And Evaluation Tests

- Assert no forbidden metadata or outcome fields become predictive tensors.
- Prevent label, `cleavage_freq`, `measured`, experiment ID, split value, raw
  ID, coordinate, genome, or cell-line metadata from silently entering
  predictive feature bundles.
- Assert NaN, negative, and greater-than-one cleavage-frequency behavior
  remains consistent with Scheme A.
- Assert `experiment_id=18` remains excluded.
- Assert no `measured=0` rows enter headline evaluation.
- Assert train-fitted preprocessing remains train-only where materialization
  applies it.
- Assert threshold selection is validation-only.
- Ensure test results cannot drive checkpoint, schema, or model selection.
- Assert canonical GCN result writing cannot collapse Graph A, Graph C, and
  Graph B rows into the same upsert identity; schema must be part of the
  result identity before multi-schema outputs are written.

### Model And Training Smoke Tests

- Assert the GCN forward pass returns the expected candidate-edge prediction
  shape.
- Run one minimal CPU smoke-training fixture or tiny debug artifact path.
- Validate deterministic seeding at a reasonable smoke-test level.
- Validate checkpoint metadata and selection source.
- Reject placeholder/debug configuration as a headline final run.

### Reporting And Visualization Tests

- Assert the results table and report carry graph schema, feature bundle,
  label scheme, split ID, prevalence, baseline reference, and visibility
  policy fields.
- Assert a mocked or tiny smoke-report path emits all required core figure
  filenames.
- Assert headline schema comparison includes positive prevalence and
  `xgboost_unweighted / F4`.
- Assert debug outputs, when present, are visibly separated from final
  reporting.
- Require position-sensitivity output only when the finalized model uses
  aligned sequence inputs.
- Assert the graph-view sanity figure is generated from a bounded readable
  loader view rather than an uncontrolled full-network plot.

### Colab Workflow Validation

- If a runner notebook is later added, assert or review that it invokes
  repository commands only.
- Ensure no final model, preprocessing, evaluation, or plotting logic exists
  only in notebook cells.
- Require Colab run artifacts to include commit SHA and resolved config.

## 15. Risks

### Scientific And Evaluation Risks

- Drift from Scheme A or `sprint2_main_seed42`.
- Accidental use of `measured=0` rows as main evaluation ground truth.
- Reintroduction of `experiment_id=18`.
- Comparison against weak baselines while omitting
  `xgboost_unweighted / F4`.
- Overclaiming Graph C as a topology-only improvement.
- Reporting AUPRC without its approximately `0.9007` positive prevalence
  context.
- Ignoring guide-size or genome-composition effects.

### Graph And Leakage Risks

- Reconstructing topology differently inside the graph-model loader.
- Loss of strict-inductive view restrictions during PyG materialization.
- Exposing validation/test candidate or similarity relations during training.
- Introducing raw IDs, coordinates, genome, cell-line, experiment ID,
  `measured`, cleavage values, labels, or split indicators into predictive
  tensors.
- Applying preprocessing globally rather than fitting permitted transforms on
  training data only.

### Training And Selection Risks

- Selecting schema, model, epoch, checkpoint, threshold, or hyperparameters
  from test results.
- Treating Graph B as an uncontrolled schema sweep.
- Pulling Sprint 5 context ablations or Sprint 6 imbalance experiments into
  the GCN baseline sprint.

### Dependency And Colab Risks

- Incompatibility among PyTorch Geometric, PyTorch, CUDA, and the selected
  Colab runtime.
- Undocumented installation workarounds in notebook cells.
- Running reported results from a different git commit or resolved config than
  recorded.
- Leaving checkpoints or run outputs only on ephemeral Colab disk.
- Committing raw data, processed graph Parquet artifacts, model checkpoints,
  caches, or Drive-specific paths.

### Visualization And Reporting Risks

- Producing only headline numbers without graph/data or model diagnostics.
- Presenting random-edge/debug figures as final results.
- Using test diagnostic plots to make model decisions.
- Treating perturbation or future attention outputs as causal biological
  evidence.
- Attempting to visualize an unreadable full Graph C network rather than a
  bounded, interpretable graph-view sanity example.

## 16. Step-By-Step Later Implementation Plan

1. Reconfirm the frozen Sprint 2 label/split/evaluation contract, the Sprint 3
   manifests, and corrected node/relation/split counts.
2. Approve the minimum PyTorch Geometric dependency and documented
   Colab/CUDA installation policy.
3. Implement typed-artifact-to-model-loader materialization with fail-fast
   manifest, schema, feature-placement, split, and visibility checks.
4. Implement strict-inductive train, validation, and test graph views and add
   loader/leakage tests.
5. Replace or revise the placeholder `gcn_minimal.yaml` so headline runs use
   `sprint2_main_seed42`, while any debug path is visibly non-final.
6. Implement the minimum GCN edge-classification path for Graph A.
7. Implement deterministic training, validation-selected checkpointing,
   validation-only thresholding, metric generation, result metadata, and
   output paths.
8. Implement required report and visualization generation, including bounded
   graph-view sanity visualization.
9. Run local CPU smoke tests and focused contract/reporting tests.
10. Prepare the Colab runner workflow with repository commands only,
    documented artifact persistence/provenance, and a pre-training graph
    artifact checksum/provenance gate.
11. Manually run Graph A on Colab GPU using the documented runner workflow.
    This step is performed by the project owner/operator, not claimed by the
    coding agent unless real returned artifacts are available.
12. Inspect returned Graph A artifacts, validate provenance/checksums,
    confirm result/report/figure completeness, and finalize Graph A reporting
    only if the real run passes all gates.
13. After Graph A is validated, run Graph C as the primary context-enriched
    comparison.
14. Run Graph B only if the bounded secondary control remains required by the
    approved scope.
15. Produce consolidated same-contract comparisons against positive prevalence
    and `xgboost_unweighted / F4`.
16. Generate sequence-position sensitivity only if the implemented GCN
    consumes aligned sequence input.
17. Validate that all results preserve no-test-tuning, strict-inductive
    visibility, artifact provenance, and figure/report boundaries.
18. Update documentation and conduct a Sprint 4 validation review before
    considering merge or Sprint 5 work.

### Implementation Slices And Gates

The ordered steps above remain the authoritative implementation sequence.
Implementation tasks should select one bounded slice at a time rather than
interpreting the whole sprint as one coding task.

| Slice | Plan steps | Bounded implementation scope | Exit gate before proceeding |
| --- | --- | --- | --- |
| Slice 1: Graph Materialization Foundation | 1-5 | Reconfirm contracts; approve and add only the minimum graph dependency; implement typed-artifact materialization, strict-inductive views, fail-fast contract assertions, final-config gate, and loader/config/leakage tests. | Graph artifacts load under the frozen contract, all materialization/visibility tests pass, and no GCN training path has been implemented or run. |
| Slice 2: Graph A Minimal GCN Path | 6-7 | Implement the Graph A edge-classifier path, fixed starting loss, deterministic training, validation-only selection, metrics/result metadata, and focused model/training/evaluation tests. Keep `physical_target_site` featureless in the artifact contract and avoid learned per-target-ID embeddings as the default baseline. | A local CPU smoke run for Graph A succeeds under the locked contract; result metadata records feature bundle, target-node representation policy, training history availability, validation-only threshold source, and baseline-reference fields; no full GPU training or Graph C/B model run has begun. |
| Slice 3: Reporting And Graph Visualization | 8-9, reporting portion | Implement results/report/diagnostic/figure generation, including a bounded graph-view sanity visualization and reporting-output tests. Consume Slice 2 structured training/prediction metadata instead of recomputing scientific decisions from plots, and verify that metadata is sufficient for required figures. | Required reporting outputs can be generated from a tiny or mocked Graph A path, graph-view visualization is local/readable rather than full-network, figure metadata includes graph schema, feature bundle, target-node representation, training history, validation-selected threshold, prevalence, and baseline reference, and figure contract tests pass before full Colab training. |
| Slice 4A: Colab Runner Infrastructure Preparation | 10 | Implement/document runner-only Colab workflow, approved PyTorch/PyG/CUDA setup notes, Google Drive copy-in/copy-out policy, run provenance capture, and pre-training Sprint 3 graph artifact checksum/provenance validation. Do not execute the full Graph A GPU run. | Colab workflow is repository-documented, uses repository commands only, records commit/config/runtime/schema/split/visibility/artifact provenance, distinguishes smoke/debug from canonical outputs, and has a pre-training checksum/provenance gate; no final Graph A result is claimed. |
| Slice 4B: Manual Full Graph A GPU Run | 11 | The project owner/operator runs the documented Colab workflow manually, produces real Graph A outputs, persists run artifacts to durable Google Drive storage, and returns artifacts for validation. Coding agents may provide instructions but must not fabricate or claim completion without real outputs. | Real Graph A result, diagnostic, report, figure, checkpoint/run, config, runtime, and provenance artifacts exist in durable storage; the run used the documented command path and no test-driven tuning was performed. |
| Slice 4C: Returned Artifact Integration And Graph A Reporting Finalization | 12 | Inspect returned Graph A artifacts, validate commit/config/provenance/checksums, verify required result/report/diagnostic/figure completeness, and update status/reporting only after real outputs pass the gates. Do not alter model/training choices based on test diagnostics. | Graph A full-run artifacts are contract-validated, complete, durable, and documented; per-genome/practical per-guide diagnostics are supported or fail fast with a documented reason; Graph A is ready as the same-contract baseline for Graph C planning. |
| Slice 5: Graph C Primary Comparison | 13, 15-17 as applicable to Graph C | Implement Graph C model-facing support, enforce observation/context visibility constraints, execute the Graph C run, and consolidate Graph A/Graph C/F4 comparisons. Explicitly distinguish topology changes from target-semantics/context-representation changes. | Same-contract Graph A versus Graph C comparison is complete and Graph C is reported as changing topology and target semantics, not as a topology-only intervention. |
| Slice 6: Optional Graph B And Sprint Closure | 14-18 as applicable after primary comparison | Run Graph B once only if retained as the bounded control; complete consolidated reporting, documentation, final audits, and Sprint 4 validation review. | Sprint 4 handoff is validated without opening Sprint 5/6 work or uncontrolled schema/model sweeps. |

Gate rules:

- Do not begin Slice 2 until Slice 1 has established that model-facing graph
  views preserve the Sprint 3 manifests and strict-inductive visibility.
- Do not treat reporting as complete until Slice 3 has produced and tested
  graph-data sanity visualization as well as model-result figure paths.
- Do not treat Slice 3 reporting as ready unless Slice 2 outputs provide the
  metadata required for figures and reports: graph schema, feature bundle,
  target-node representation, training history, validation-selected threshold,
  prevalence, and baseline reference.
- Do not begin Slice 4A until local Graph A smoke training and reporting paths
  have passed their focused tests.
- Do not begin the manual Slice 4B Colab GPU run until Slice 4A has documented
  the runner workflow, durable Drive artifact policy, approved dependency
  setup, run-provenance capture, and graph artifact checksum/provenance gate.
- Do not publish or treat a full Graph A run as report-complete until Slice 4C
  has inspected real returned artifacts and confirmed that
  prediction outputs carry the metadata needed for required per-genome and
  practical per-guide diagnostics, or the reporting path fails fast with a
  documented reason. Placeholder "metadata unavailable" figures are acceptable
  for Slice 3 smoke/mock validation only, not for final Sprint 4 claims.
- Do not run canonical `scripts/train.py --config
  configs/experiments/gcn_minimal.yaml` as a smoke command unless output paths
  are explicitly redirected to non-canonical locations; the canonical command is
  a real artifact-producing run.
- Do not begin the manual Slice 4B Colab GPU run until the copied Sprint 3
  graph artifacts pass an explicit checksum/provenance validation against the
  approved canonical artifact source.
- Do not write multi-schema GCN results until the result upsert key includes
  `graph_schema` or another schema-disambiguating field.
- Do not begin Graph C or Graph B runs until Slice 4C has validated real Graph
  A same-contract artifacts and reporting outputs.
- Do not use Graph B as a branch for adaptive schema tuning; it remains one
  optional bounded secondary control.
- Do not permit a slice-level task to broaden into later slices unless the
  plan is explicitly revised and approved.

## 17. Acceptance Criteria

Sprint 4 implementation will be complete only when it:

- States and enforces the exact frozen Sprint 2 label, split, evaluation, and
  baseline comparison contract.
- States and enforces the exact Sprint 3 Graph A/B/C artifact, feature
  placement, visibility, and count contract.
- Resolves and documents the approved PyTorch Geometric/materialization and
  Colab/CUDA setup policy.
- Uses repository-owned logic with Colab as a runner only.
- Trains Graph A first, Graph C second, and Graph B only as a bounded control
  if completed.
- Uses a fixed initial Sprint 4 loss/training policy without broadening into
  Sprint 6.
- Uses validation-only early stopping/checkpointing/thresholding and prevents
  test-driven choices.
- Reports comparison against `xgboost_unweighted / F4` and positive
  prevalence.
- Produces required results, reports, diagnostics, graph-data visualization,
  model-evaluation figures, and any conditionally required sequence
  sensitivity output.
- Passes loader, leakage, smoke-training, evaluation, visualization,
  configuration, and Colab workflow validation checks.
- Records run provenance and follows storage/version-control policy for
  reports, figures, checkpoints, raw data, graph artifacts, and Colab/Drive
  outputs.
- Leaves systematic context-feature ablation to Sprint 5 and systematic
  imbalance comparisons to Sprint 6.

Planning-task completion criterion: this execution-plan document records the
scope and requirements only. It does not add dependencies, implement models,
modify graph artifacts/configurations/tests/notebooks/output reports, or train
any model.

## 18. Commands To Plan For Later Implementation

These are planned command patterns only. They are not to be run as part of
creating this plan.

### Local Development And Validation Pattern

```bash
uv sync
uv run ruff check scripts src tests
uv run pytest -q
uv run python scripts/build_graph.py --config configs/data/mak2022.yaml --schema-config configs/sweeps/graph_schema_ablation.yaml
uv run pytest -q
```

### Later Sprint 4 Training And Evaluation Pattern

The final entry-point choice must follow the implemented repository convention.
The preferred current pattern is the existing training dispatcher:

```bash
uv run python scripts/train.py --config configs/experiments/gcn_minimal.yaml
```

This local command writes canonical Sprint 4 outputs. It must be run as an
artifact-producing command only after Slice 4A runner/provenance gates are
ready. In the manual Slice 4B Colab workflow, the runner should not mutate this
base config in place. Instead it should generate
`outputs/sprint4/graph_a/<run_id>/resolved_config.yaml` with runtime-only
fields such as `run_id` and `training.device`, then run:

```bash
uv run python scripts/train.py --config outputs/sprint4/graph_a/<run_id>/resolved_config.yaml
```

Coding agents must not claim that this command completed unless real returned
artifacts exist. For smoke/debug use, output paths must be redirected to
non-canonical locations. Any `--debug` or smoke mode for `sprint4_gcn` must be
visibly non-final and must not write to headline result paths by default.

Use a separate evaluation entry point only if implementation justifies it:

```bash
uv run python scripts/evaluate.py --config configs/experiments/gcn_minimal.yaml
```

### Slice 4A Colab Runner Preparation Pattern

Slice 4A documents the exact runner commands and Drive paths, but does not
execute the full Graph A GPU run. The runner documentation should follow this
shape:

```bash
git clone <repository-url>
cd crispr-gnn-offtarget
git checkout <approved-sprint4-branch-or-commit>
uv sync
uv run python scripts/train.py --config outputs/sprint4/graph_a/<run_id>/resolved_config.yaml
```

Exact Colab/PyG/CUDA setup commands cannot be finalized until the dependency
decision is approved. Once resolved, they must be documented in repository
files before final training, and notebook runners must invoke those documented
commands without hiding scientific logic or configuration.

### Slice 4B Manual Colab Run Pattern

The full Graph A GPU run is manual. The operator runs the Slice 4A documented
commands in Colab, copies input artifacts from Google Drive to Colab local disk
as documented, copies generated outputs back to durable Drive storage, and
returns artifacts for Slice 4C validation. The run record must include commit
SHA, resolved config, seed, device/runtime, graph schema, feature bundle, split
ID, visibility policy, and Sprint 3 graph artifact provenance.

### Slice 4C Returned Artifact Validation Pattern

Returned artifacts should be inspected before any Graph C/B work begins. The
validation should confirm that:

- the reported commit and resolved config match the approved run;
- copied Sprint 3 graph artifacts match the expected checksum/provenance gate;
- `outputs/sprint4/graph_a/gcn_graph_a_results.csv`,
  `outputs/sprint4/graph_a/gcn_graph_a_report.md`,
  `outputs/sprint4/graph_a/figures/`, and
  `outputs/sprint4/graph_a/diagnostics/` are complete;
- prediction outputs support required per-genome and practical per-guide
  diagnostics;
- no test diagnostic changed model, schema, epoch, threshold, feature, or
  reporting choices.

## 19. Documentation Updates Required Later

After approved implementation and validated results, update only the
documentation necessary to record actual choices and results:

- `docs/DECISIONS.md`:
  - PyTorch Geometric/`HeteroData` decision.
  - Colab/CUDA dependency workflow.
  - Fixed Sprint 4 model/training policy.
  - Final sequence-position sensitivity interpretation decision, if
    applicable.
- `docs/EVALUATION_PROTOCOL.md` only if materialized visibility, thresholding,
  or reporting details require clarification.
- `docs/COMMANDS.md` with reproducible local commands, Slice 4A Colab runner
  commands, and returned-artifact validation guidance.
- `README.md` with Sprint 4 status and result locations after Slice 4C
  validation, not after runner preparation alone.
- `docs/PROJECT_CONTEXT.md` with Sprint 4 handoff status once validated GCN
  results exist.
- `colab/README.md` during Slice 4A if a documented GPU runner or Drive
  artifact workflow is implemented.
- `outputs/sprint4/graph_a/gcn_graph_a_report.md` as the canonical Graph A result report only
  after real returned artifacts pass Slice 4C validation.
- `outputs/sprint4/graph_a/figures/` as the required Graph A report-ready visualization set
  only after real returned artifacts pass Slice 4C validation.

This plan does not authorize Sprint 4 implementation until dependency,
materialization, and training work is separately undertaken under the frozen
contracts above.
