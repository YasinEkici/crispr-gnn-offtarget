# Decisions

## 2026-04-30 - Initialize uv-first repository

Decision: use `uv`, `pyproject.toml`, and `uv.lock` as the dependency source of truth.

Reason: the project needs a reproducible workflow that works locally and in Colab without a manually maintained `requirements.txt`.

## 2026-04-30 - Keep Sprint 0 ML-free

Decision: create only scaffold, config loading, label helpers, and smoke scripts.

Reason: PyTorch and PyTorch Geometric are deferred until the graph-model sprint so initialization stays lightweight and reviewable.

## 2026-05-21 - Use Wayback Mak 2022 dataset snapshot as Phase 1 source

Decision: use the local `data/raw/260520_putative_nucleosomal.parquet` file recovered from the Internet Archive Wayback snapshot of `https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz` as the Phase 1 dataset.

Reason: the original crisprSQL URL is unavailable, while the Wayback snapshot of the same original URL provides the working Mak et al. 2022 crisprSQL-derived epigenetic/nucleosome dataset used by this project.

Alternatives considered:

- Reopen source discovery.
- Switch immediately to raw crisprSQL.
- Switch immediately to CRISPRoffT.

Outcome: keep Mak et al. 2022 as the first working dataset; keep raw crisprSQL as fallback and CRISPRoffT as stretch external validation.

## 2026-05-21 - Adopt Scheme A as the primary binary label

Decision: use `cleavage_freq > 1e-5` as the primary binary label scheme.

Reason: Sprint 1 audit confirmed `cleavage_freq` is present and transformed `CA` is absent. After excluding 78 NaN `cleavage_freq` rows from supervised label generation, Scheme A has 310,064 label-eligible rows: 21,365 positives and 288,699 negatives. The threshold is aligned with the paper's assay-accuracy boundary.

Alternatives considered:

- Scheme B: Mak CA / Box-Cox reproduction.
- Scheme C: `cleavage_freq > 1e-3`.
- Continuous regression target.

Outcome: Scheme A is primary for binary guide-level AUPRC. Scheme C is reserved for robustness sensitivity. Scheme B is deferred to a later paper-comparison track only.

## 2026-05-21 - Keep Mak CA reproduction out of the project center

Decision: do not center the project on reproducing Mak et al.'s transformed CA target.

Reason: the audited raw dataset does not contain a transformed `CA` column. Reproducing Scheme B would require per-study Box-Cox transformation, standardization, and clipping. That is useful for paper comparison but not required for the main project contribution.

Outcome: main track remains binary off-target classification plus epigenetic/context-aware GNN evaluation under guide-level AUPRC.

## 2026-05-21 - Set `cleavage_freq` outlier policy for binary labels

Decision:

- NaN `cleavage_freq`: exclude from supervised binary train/validation/test label generation; do not silently impute as negative.
- Negative `cleavage_freq`: below-threshold for binary sensitivity counts, but flagged as raw-label quality issue.
- `cleavage_freq > 1`: positive for binary thresholds; do not clip for binary classification.

Reason: Sprint 1 audit found 78 NaN values, 685 negative values, and 298 values above 1. Silent imputation, clipping, or dropping would make later labels hard to audit.

Outcome: label generation must preserve these policies and report affected counts. `src/crispr_gnn/data/labels.py` rejects missing or NaN `cleavage_freq` by default so these rows cannot silently become negative labels.

## 2026-05-21 - Restrict test rows to measured experimental data

Decision: final test rows must contain only `measured=1` rows.

Reason: `measured=0` rows are putative off-target candidates and are not ground-truth test labels. Sprint 1 audit found 25,632 `measured=1` rows and 284,510 `measured=0` rows.

Outcome:

- Test: `measured=1` only.
- Validation: prefer `measured=1`.
- Training: may include `measured=0` rows only as optional noisy negatives.
- Report measured composition for every split.

## 2026-05-21 - Treat `experiment_id=18` as a main-evaluation risk

Decision: keep `experiment_id=18` out of main evaluation or report it as a separate no-cell-line sensitivity subset.

Reason: Sprint 1 audit found all 14,108 missing `cell_line` rows concentrated in `experiment_id=18`, and computed nucleosome missingness is also heavily concentrated there.

Outcome: future split and evaluation docs must avoid hiding this subset inside main reported performance.

## 2026-05-23 - Exclude `experiment_id=18` from Sprint 2 main baselines

Decision: Sprint 2 main baseline runs will exclude `experiment_id=18` from train, validation, and test rows. The subset may be used only for a separately labeled no-cell-line / high-missingness sensitivity analysis if enough label-eligible measured rows exist.

Reason: Sprint 1 identified `experiment_id=18` as the concentrated source of missing `cell_line` values and a major source of computed nucleosome missingness. Sprint 2 is intended to create clean, fair non-graph baselines for later GNN comparison, not to maximize data volume by mixing a known high-missingness subset into the primary benchmark.

Alternatives considered:

- Allow `experiment_id=18` in training while excluding it from validation/test.
- Include it in all splits and report a flag.
- Exclude it only from feature sets that use computed nucleosome features.

Outcome: the active Sprint 2 execution plan must define a `main_clean` evaluation universe that excludes `experiment_id=18` before splitting. Any reported `experiment_id=18` result is sensitivity-only and must not be mixed into the headline baseline table.

## 2026-05-23 - Use XGBoost as the Sprint 2 boosted-tree baseline

Decision: add `xgboost` as a Sprint 2 dependency and use it as the official boosted-tree tabular baseline. Random Forest remains an optional scikit-learn fallback or sanity baseline.

Reason: the project plan explicitly calls for "XGBoost or Random Forest" in Sprint 2, and Mak et al. used XGBoost in their modeling track. XGBoost is a classical ML dependency, not a graph or PyTorch dependency, and gives the tabular baseline more credibility than relying only on a Random Forest.

Alternatives considered:

- Use only `RandomForestClassifier` from scikit-learn.
- Use scikit-learn `HistGradientBoostingClassifier` as an XGBoost-like substitute.
- Defer boosted trees until later.

Outcome: dependency updates must be made through `uv` / `pyproject.toml`, not `requirements.txt`. If XGBoost installation or runtime fails in a target environment, the fallback must be documented and Random Forest or a scikit-learn boosted-tree model may be used as a clearly labeled fallback.

## 2026-05-23 - Report unweighted XGBoost primary and balanced XGBoost sensitivity

Decision: Sprint 2 reports `xgboost_unweighted` as the primary boosted-tree baseline. `xgboost_balanced_train_weights` is reported as a separately labeled training-weight sensitivity, not as a replacement chosen from test performance.

Reason: the measured-only main split is positive-heavy, but the negative class is scientifically important. The unweighted run is the clearest primary comparison to the unweighted Logistic Regression debug baseline. The balanced train-weight run is useful for checking whether inverse-frequency training weights improve negative-class recognition, but it changes the training objective and must therefore be interpreted separately.

Outcome: XGBoost runs use the same Scheme A labels, locked guide-level split, measured-only validation/test rows, `experiment_id=18` exclusion, F1-F4 feature ladder, train-only preprocessing, validation-only threshold selection, and no test-set tuning. Test diagnostics are for interpretation only, not feature-set or hyperparameter selection.

## 2026-05-23 - Add PyTorch in Sprint 2 for sequence baselines only

Decision: Sprint 2 will introduce PyTorch for non-graph neural baselines, specifically the required CnnCrispr-inspired sequence CNN/BiLSTM baseline. PyTorch Geometric remains deferred until the graph-model sprint.

Reason: the current project plan makes a same-split sequence-based deep learning baseline a Sprint 2 must-have. An sklearn MLP is a tabular neural baseline and does not satisfy the sequence-based DL requirement. Adding PyTorch now resolves that requirement while still keeping graph-specific dependencies and graph-model implementation out of Sprint 2.

Alternatives considered:

- Defer PyTorch and replace the sequence DL baseline with an sklearn MLP.
- Add both PyTorch and PyTorch Geometric immediately.
- Use only published paper-reported sequence baseline metrics.

Outcome: Sprint 2 may add PyTorch and implement non-graph sequence models, but must not add PyTorch Geometric or start graph construction/modeling work. Any CUDA or Colab-specific setup notes must be documented rather than handled through ad-hoc install commands.

## 2026-05-23 - Use scikit-learn MLPClassifier for the Sprint 2 tabular MLP

Decision: implement the Sprint 2 tabular MLP baseline with `sklearn.neural_network.MLPClassifier`, trained through the existing config-driven `scripts/train.py` path.

Reason: the tabular MLP is a neural non-graph tabular baseline, not the required sequence deep-learning baseline. Keeping it in scikit-learn avoids unnecessary PyTorch/XGBoost native runtime interaction in the tabular baseline process and keeps the same train-only preprocessing, validation-only early stopping, feature audit, diagnostics, and result-table schema used by the other Sprint 2 tabular baselines. PyTorch remains available for the later CnnCrispr-inspired sequence baseline where it is necessary.

Alternatives considered:

- Implement the tabular MLP in PyTorch immediately.
- Defer the tabular MLP until the sequence-model PyTorch infrastructure exists.
- Omit the tabular MLP and rely only on Logistic Regression plus XGBoost.

Outcome: `tabular_mlp_unweighted` runs on F1-F4. `tabular_mlp_balanced_train_weights` is a focused F3/F4 sensitivity. XGBoost remains the current strongest tabular baseline unless later diagnostics show otherwise.

## 2026-05-23 - Use S1 sequence-pair input for pure sequence baselines

Decision: Sprint 2 sequence-only CNN/BiLSTM baselines use `S1_sequence_pair`: aligned guide and target sequence inputs encoded as guide-base one-hot channels, target-base one-hot channels, and one aligned mismatch channel over 23 positions.

Reason: the sequence baseline should answer how well a neural model performs from guide-target sequence relationship alone. It must not receive binding-energy scalars, epigenetic/context scalars, computed nucleosome features, genome/cell-line labels, experiment IDs, guide IDs, target coordinates, measured flags, labels, or cleavage values. Keeping this pure sequence input separates sequence-model evidence from the context-rich F3/F4 tabular baselines.

Alternatives considered:

- Add F3/F4 late-fusion context immediately.
- Use only engineered F1 mismatch features instead of raw aligned sequence channels.
- Include genome or cell-line metadata as sequence-model context.

Outcome: `sequence_cnn_*` and `sequence_bilstm_*` rows are reported with feature set `S1`. Late-fusion sequence + F3/F4 context models remain optional and must be separately labeled if added later.

## 2026-05-23 - Limit sequence late fusion to a small CNN + F3/F4 Sprint 2 slice

Decision: Sprint 2 may include only the small unweighted CNN late-fusion slice: `sequence_cnn_plus_F3_late_fusion_unweighted` and `sequence_cnn_plus_F4_late_fusion_unweighted`.

Reason: pure sequence CNN/BiLSTM baselines are weak, while XGBoost shows that F3/F4 context features carry most of the useful non-graph signal. A small CNN late-fusion run answers whether adding the same tabular context families helps the neural sequence baseline. It should not expand into a BiLSTM-fusion, balanced-fusion, or architecture-sweep track unless there is a clear validation-side reason.

Alternatives considered:

- Add BiLSTM + F3/F4 late fusion immediately.
- Run balanced late-fusion variants.
- Skip late fusion and freeze Sprint 2 after pure sequence.

Outcome: late-fusion rows are reported separately from pure sequence rows with feature sets `S1+F3` and `S1+F4`. They are context-fusion neural baselines, not pure sequence baselines, because F3/F4 include engineered sequence/mismatch features and context features.

## 2026-05-23 - Use named Sprint 2 feature ladder with train-only F4 imputation

Decision: Sprint 2 uses a named feature ladder: `F1` sequence/mismatch features, `F2` adds binding energy, `F3` adds experimental epigenetic scalars, and `F4` adds aggregated computed nucleosome features plus missingness indicators. Missing computed aggregate values in `F4` are imputed during model preprocessing using train-only statistics. Rows are not dropped from the main `F1`-`F4` comparison because computed nucleosome arrays are missing.

Reason: Sprint 2 needs fair same-split baselines that can show whether each feature family adds value. Dropping rows only for `F4` would change the evaluation population and make `F1`-`F4` comparisons harder to interpret. Keeping missingness indicators makes imputed computed-context values explicit.

Alternatives considered:

- Drop rows missing computed nucleosome arrays for `F4`.
- Omit computed nucleosome features from required Sprint 2 baselines.
- Use full 299-dimensional position-resolved computed features as the required Sprint 2 context representation.

Outcome: `F4` uses aggregated computed features as the required Sprint 2 context feature set. Full 299-dimensional position-resolved computed features remain optional/later. Raw identifiers, genome labels, cell-line labels, and coordinates are not predictive features in Sprint 2 main baselines.

## 2026-05-21 - Keep non-human genomes by default with explicit reporting

Decision: do not drop non-`hg19` genomes by default.

Reason: the audited dataset includes `hg19`, `hg38`, `rn5`, `mm10`, and `mm9`. Dropping non-human genomes without an explicit experiment would narrow the project and could change the dataset distribution.

Outcome: later evaluations should report per-genome breakdowns and avoid human-only claims unless the experiment intentionally filters to human genomes.

## 2026-05-21 - Use strict parser for computed nucleosome arrays

Decision: computed nucleosome features must parse as exactly 23 numeric values. Missing values are tracked separately from malformed arrays.

Reason: Sprint 1 audit confirmed all 13 computed nucleosome features share the same parser status: 294,989 valid rows, 15,153 missing rows, 0 malformed-length rows, and 0 non-numeric rows.

Outcome:

- Parser behavior lives in `src/crispr_gnn/data/parsers.py`.
- Tests live in `tests/test_feature_parsers.py`.
- Later feature builders must choose an explicit missingness policy before using computed features.

## 2026-05-25 - Build typed Sprint 3 graph artifacts without PyTorch Geometric

Decision: Sprint 3 serializes dependency-light typed node, relation, and feature tables plus graph manifests. PyTorch Geometric and `HeteroData` materialization are deferred until Sprint 4 model training is approved.

Reason: graph construction and leakage validation do not require a graph-training dependency. Typed artifacts allow the graph schema, feature placement, and strict-inductive visibility policy to be tested before selecting a GNN architecture.

Outcome: graph tables are generated under `data/processed/graphs/sprint3/` and the tracked handoff artifact is `outputs/sprint3/graph_schema_report.md`.

## 2026-05-25 - Keep Graph A context on candidate edges and use genome-aware target keys

Decision: Graph A physical target nodes use the key `genome + target_chr + target_start + target_end + target_strand`. Their predictive content is restricted to identity/type representation; S1/F1-F4 row-varying inputs remain candidate-pair edge features.

Reason: in the locked measured-only Sprint 2 universe, repeated physical target coordinates occur across supervised splits and carry varying experimental and computed context values. Coordinates are also assembly-specific, so target identity must include genome. Storing observation context on a shared physical node would blur distinct assay observations and risk leakage across held-out guides.

Outcome: Graph A is the minimal physical-target schema while preserving the Sprint 2 edge-level feature contract and train-only F4 imputation policy.

## 2026-05-25 - Make Graph C the context-observation schema and Graph B a bounded control

Decision: Graph C uses one `target_observation` node per source row `id`, adds context-only target-observation similarity, and follows a strict-inductive visibility policy. Graph B adds guide-sequence Hamming similarity to Graph A but is retained only as a bounded secondary control.

Reason: Mak/crisprSQL context describes a measured guide-target observation and may differ at the same physical target site. Graph C represents this without forcing incompatible values into one physical node. Strict-inductive construction ensures validation/test observations connect only to training observations using train-fitted, label-free context processing. Graph B isolates inexpensive topology enrichment without expanding Sprint 4 into unnecessary schema sweeps.

Outcome: Graph A and Graph C are the primary Sprint 4 schemas; Graph B may receive one controlled later run after Graph A training is stable. All later results must compare against `xgboost_unweighted / F4` under the locked Sprint 2 protocol.

## 2026-05-26 - Require reproducible visual reporting for model-evaluation sprints

Decision: every model-training or model-evaluation sprint from Sprint 4 onward must produce report-ready performance and diagnostic figures alongside metrics tables and Markdown reports. Graph-model comparisons must visually include positive prevalence and the frozen `xgboost_unweighted / F4` reference where applicable.

Reason: aggregate metric tables alone are insufficient under the positive-heavy test set and uneven guide/genome composition. Visual PR/ROC, training-history, score/threshold, and subgroup diagnostic outputs make the result interpretable while preserving the same scientific contract. The literature notes additionally support position-level perturbation views for sequence-bearing neural predictions (CRISPR-Net), feature-distribution and SHAP-style context contribution analysis (Mak et al. 2022), and positive-retrieval/variability reporting when comparing imbalance interventions (Gao 2020; Guan 2024).

Outcome: Sprint 4-7 deliverables explicitly include figures under `outputs/<sprint_name>/<model>/figures/`. Sprint 4 adds a focused position-level sensitivity artifact when its trained GCN consumes aligned sequence input; Sprint 5 adds context distribution and model-contribution artifacts; Sprint 6 adds positive-retrieval and across-guide variability artifacts. Figures remain subject to the locked guide-level split, Scheme A, measured-only main evaluation, `experiment_id=18` exclusion, validation-only threshold selection, and no test-driven model or schema selection. Random-edge or exploratory figures must be labeled debug-only. SHAP, perturbation, and attention diagnostics are interpretation-only and must not be claimed as causal biological evidence.

## 2026-05-28 - Materialize Sprint 3 typed artifacts with minimal PyG `HeteroData`

Decision: add `torch-geometric>=2.7.0` as the only new graph-specific dependency
for Sprint 4 Slice 1 and materialize the validated Sprint 3 typed graph tables
as strict-inductive PyG `HeteroData` views.

Reason: Graph A/B/C are already persisted as typed node, relation, feature,
and manifest tables. `HeteroData` provides a typed model-facing container
without requiring topology reconstruction from raw dataset rows. The official
PyTorch Geometric installation guidance states that basic PyG use requires
only PyTorch and `torch_geometric`; optional compiled extension packages should
be introduced only when a demonstrated later model/runtime need requires them.

Outcome:

- `src/crispr_gnn/graph/pyg_dataset.py` reads serialized Sprint 3 artifacts
  and validates the frozen Scheme A, `sprint2_main_seed42`, manifest count,
  feature-placement, preprocessing-scope, and strict-inductive visibility
  contracts before exposing PyG views.
- Train, validation, and test views retain their permitted relation fragments;
  the training view cannot include held-out candidate supervision.
- Raw identifiers and reporting metadata remain audit information and are not
  inserted into predictive `x` or `edge_attr_*` tensors.
- `configs/experiments/gcn_minimal.yaml` declares the locked headline protocol
  rather than a debug split. Configuration validation rejects debug or
  random-edge settings as headline evaluation.
- This slice does not implement a GCN model, training loop, Colab GPU run, or
  reporting figures.

Reference:

- PyTorch Geometric installation documentation:
  `https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html`

## 2026-05-30 - Enable torch.compile, bfloat16 AMP, and extended epochs for A100 GPU run

Decision: enable `torch.compile`, bfloat16 mixed-precision autocast, and extended
training epochs for the Sprint 4 Graph A GPU run on A100 hardware.

Reason: A100 GPU provides tensor-core-accelerated bfloat16 and PyTorch 2.x
compile support that are not available on CPU. These changes improve training
efficiency without altering the frozen evaluation contract.

Outcome:

- `use_compile: true` in config — applies `torch.compile(model)` before the
  training loop when `device == cuda`. Skipped silently on CPU. Adds JIT
  compilation overhead on the first epoch; subsequent epochs are faster.
- `use_amp: true` in config — wraps training and inline val-loss forward passes
  with `torch.autocast("cuda", dtype=torch.bfloat16)`. Loss inputs are cast
  to float32 before `BCEWithLogitsLoss` to preserve numerical stability.
  Skipped silently on CPU. Eval forward passes in `_scores_for_view` remain
  float32 for precise metric computation.
- `max_epochs: 300`, `patience: 15` — more training time; early stopping on
  val_auprc still protects against overfitting.
- These changes do not affect the frozen label, split, visibility, threshold
  selection, or evaluation contract. Results table gains `use_compile` and
  `use_amp` provenance fields. CPU runs with `use_compile: true` and
  `use_amp: true` in config are safe — both flags are gated on `device.type == cuda`.

## 2026-05-30 - Add gradient clipping, LR scheduling, LayerNorm, and encoder activation to GCN baseline

Decision: Apply five code-validated training and architecture improvements to the
Sprint 4 Graph A GCN baseline before the first headline GPU run.

Reason: Systematic validation against the actual training loop revealed five
confirmed problems: (1) no gradient clipping after `loss.backward()`, creating
instability risk on GPU float32; (2) fixed learning rate with no scheduler,
reducing convergence quality; (3) no nonlinear activation after `sgrna_encoder`,
collapsing the encoder and first GCNConv linear into one linear transformation;
(4) no inter-layer LayerNorm, reducing training stability with multiple conv
layers; (5) validation loss not tracked in history, limiting diagnostics.
Two initially raised concerns were rejected: edge features not entering message
passing is the documented GCNConv architectural choice (Sprint 7 GAT addresses
this), and input feature normalization belongs to Sprint 5 feature ablation.

Outcome:

- `src/crispr_gnn/models/gcn.py`: `sgrna_encoder` changed to
  `nn.Sequential(Linear, ReLU)`; `norms` ModuleList of `nn.LayerNorm` added and
  applied post-conv in the forward loop.
- `src/crispr_gnn/training/gcn.py`: `clip_grad_norm_(max_norm=1.0)` added before
  `optimizer.step()`; `ReduceLROnPlateau(mode="max")` scheduler steps on
  `val_auprc`; `val_loss` and `lr` added to per-epoch history dict.
- `GCNRunConfig` extended with `clip_grad_norm`, `scheduler`,
  `scheduler_factor`, `scheduler_patience`, `scheduler_min_lr` — all
  config-driven with defaults matching approved Sprint 4 policy.
- `configs/experiments/gcn_minimal.yaml` updated with the new training fields.
- `tests/test_gcn_training_smoke.py` extended to assert `val_loss` and `lr`
  presence and non-negativity.
- These changes do not alter the frozen Sprint 2/3 label, split, visibility, or
  evaluation contract. Loss function remains weighted BCE only (Sprint 6 scope
  for focal loss). Edge feature message passing remains Sprint 7 scope.

## 2026-05-30 - Use Colab as a runner with pre-training graph artifact provenance

Decision: Sprint 4 full GPU training may run on Google Colab, including Colab
Pro when available, but Colab remains a runner only. Repository code and
configs remain the source of truth. Before any headline Graph A run, the
copied Sprint 3 graph artifacts must pass a repository-owned provenance gate
that validates the frozen loader contract and records SHA256 checksums.

Reason: Colab provides practical GPU runtime capacity, but notebook-local model
logic, ad hoc dependency fixes, and unverified Drive copies would weaken the
frozen Sprint 2/Sprint 3 contract. A checksum/provenance record makes the
copied graph artifact identity explicit before training and gives Slice 4C a
concrete returned artifact to inspect.

Outcome:

- `colab/README.md` documents the runner-only workflow, Drive copy-in/copy-out
  policy, PyTorch/PyG/CUDA version check, required returned artifacts, and the
  boundary between debug and canonical output paths.
- `scripts/validate_graph_artifacts.py` validates the copied Sprint 3 graph
  artifacts through the Sprint 4 loader and writes
  `outputs/sprint4/graph_a/<run_id>/graph_artifact_provenance.json`.
- A Graph A Colab result without a passing provenance record is provisional or
  debug-only and must not enter headline Sprint 4 reporting.
- Any Colab-specific dependency workaround must be documented in repository
  files before the run can support a final claim.

## 2026-06-01 - Organize generated outputs by sprint and schema

Decision: Track small scientific reports, result tables, diagnostics, and
figures under sprint-scoped output directories instead of flat
`outputs/reports/`, `outputs/results/`, `outputs/figures/`, and
`outputs/diagnostics/` folders. Sprint 4 graph-model outputs use a
schema-specific layout such as `outputs/sprint4/graph_a/`, with run artifacts
stored below `outputs/sprint4/graph_a/<run_id>/`.

Reason: Sprint-scoped directories make handoff artifacts easier to audit and
avoid mixing baseline, graph-construction, and graph-model outputs. The
schema-level Sprint 4 layout prevents Graph A, Graph B, and Graph C files from
overwriting one another while preserving direct comparison under the locked
Sprint 2/Sprint 3 contract.

Outcome:

- Sprint 1 audit artifacts live under `outputs/sprint1/`.
- Sprint 2 baseline reports, results, diagnostics, and figures live under
  `outputs/sprint2/`.
- Shared Sprint 2 split and feature handoff artifacts remain under
  `outputs/splits/` and `outputs/features/`.
- Sprint 3 tracked graph handoff report lives at
  `outputs/sprint3/graph_schema_report.md`; large typed graph tables remain
  under ignored `data/processed/graphs/sprint3/`.
- Sprint 4 Graph A outputs live under `outputs/sprint4/graph_a/`.
- Large run directories, checkpoints, copied graph tables, caches, and
  Colab-local artifacts remain untracked; `.gitignore` ignores model checkpoint
  extensions.
- Colab full runs should preserve the repository base config and execute a
  run-specific `resolved_config.yaml` stored under the run directory.

## 2026-06-01 - Graph A Slice 4C validation passed; Graph A is the validated same-contract GCN baseline

Decision: The real Colab GPU Graph A run (commit `9f17e4f`, run ID
`sprint4_graph_a_gcn_seed42_20260601`) has passed Slice 4C artifact and
provenance validation. Graph A is the validated first GCN baseline under
the frozen Sprint 2/Sprint 3 contract. It does not beat `xgboost_unweighted
/ F4`.

Reason: All required artifacts are present and complete: canonical CSV,
report, nine core figures, six diagnostic tables, and a run directory
containing `resolved_config.yaml`, `runtime.json`,
`graph_artifact_provenance.json`, `training_history.csv`, and `model.pt`.
The provenance record confirms the Sprint 3 graph artifact checksums, split
`sprint2_main_seed42`, Scheme A labels, strict-inductive visibility, seed
42, CUDA device, and no test-driven tuning.

Outcome:

- Test AUPRC `0.9663`, test AUROC `0.7451`, test F1 `0.9518`, test MCC
  `0.3008`; positive prevalence `0.9007`.
- Graph A does not beat `xgboost_unweighted / F4` (test AUPRC `0.9925`).
  It is a valid same-contract graph baseline, not a stronger predictive one.
- Graph C planning may now begin. Graph B remains a bounded control pending
  a separate approval.
- No model, schema, epoch, threshold, or feature choice was revised from
  test diagnostics.

## 2026-06-01 - Defer sequence-position sensitivity figure to Sprint 5

Decision: The `gcn_graph_a_sequence_position_sensitivity.png` conditional
figure is deferred. It will not be produced as part of Sprint 4 Slice 4C.

Reason: `S1_pair` is confirmed position-aligned (23 positions × 11 channels,
columns `s1_pos_{pp:02d}_channel_{cc:02d}`). The exec plan §11 makes this
figure conditional on an aligned sequence input, so the condition is met.
However, generating a per-position occlusion or masking sensitivity map
requires a dedicated inference pass with position-level perturbation logic
not currently implemented in the Sprint 4 reporting path. Implementing it
within Slice 4C would broaden the slice scope. Sprint 5 systematic feature
ablation is the approved location for position-level attribution analysis.

Outcome:

- `outputs/sprint4/graph_a/figures/gcn_graph_a_sequence_position_sensitivity.png`
  is not produced in Sprint 4 Slice 4C.
- The nine core figures listed in exec plan §11 remain complete.
- Position-sensitivity analysis is deferred to Sprint 5 feature ablation.
- This deferral does not affect the Graph A headline metrics, provenance
  validation, or the Slice 4C exit gate.

## 2026-06-01 - Graph C Slice 5C validation passed; Graph C is a validated same-contract GCN comparison

Decision: The real Colab GPU Graph C run (commit `3d18bec`, run ID
`sprint4_graph_c_gcn_seed42_20260601`) has passed returned-artifact and
provenance validation. Graph C is a validated same-contract Sprint 4 GCN
comparison against Graph A and `xgboost_unweighted / F4`.

Reason: The returned Graph C artifacts include the canonical result CSV,
report, nine required figures, diagnostic tables, and a run directory with
`resolved_config.yaml`, `runtime.json`, `graph_artifact_provenance.json`,
`training_history.csv`, and an ignored `model.pt` checkpoint. The provenance
record confirms Scheme A, `sprint2_main_seed42`, strict-inductive visibility,
and the expected Graph C counts: `sgRNA=150`, `target_observation=11446`,
`candidate_pair=11446`, and `context_similar_to=91754`.

Outcome:

- Graph C test AUPRC `0.9616`, test AUROC `0.7599`, test F1 `0.9589`, test
  MCC `0.4537`; positive prevalence `0.9007`.
- Graph C does not beat `xgboost_unweighted / F4` on primary test AUPRC
  (`0.9925`) and does not beat Graph A on primary test AUPRC (`0.9663`).
- Graph C improves MCC relative to Graph A in this run, but MCC is secondary
  and cannot override the primary AUPRC comparison or drive test-based model
  changes.
- Graph C is not a topology-only comparison. Relative to Graph A, it changes
  topology through `context_similar_to` relations and target semantics through
  feature-bearing `target_observation` nodes instead of featureless shared
  `physical_target_site` nodes.
- Consolidated Sprint 4 comparison artifacts live under `outputs/sprint4/`.
  Graph B remains optional as a bounded control and must not become an
  uncontrolled schema sweep.

## 2026-06-01 - Run Graph B as bounded topology-ablation control for thesis ablation story

Decision: Graph B was run as a bounded secondary control after validated
Graph A and Graph C results. It is not a primary result, not a tuning branch,
and does not affect any model, threshold, feature, schema, or reporting choice.

Reason: The three-way ablation (Graph A → Graph B → Graph C) isolates topology
contribution from Graph C's combined topology + target-semantics change. Graph A
uses featureless physical targets with candidate edges only. Graph B adds
label-free guide-similarity edges (`sequence_similar_to`, 1208 edges) without
changing target representation or candidate features. Graph C changes both
topology (context-similarity edges) and target semantics (feature-bearing
`target_observation` nodes). Without Graph B, a reviewer cannot distinguish
topology contribution from feature contribution in the Graph A → Graph C gap.
This ablation is scientifically necessary for the thesis comparison table.

Outcome:

- Graph B provides a clean topology-only reference point in the Graph A/B/C
  ablation. It was run once and its results are interpretation-only.
- Graph B must not be treated as a primary result or used to open new
  schema tuning.

## 2026-06-01 - Graph B Slice 6C validation passed; Graph B is a validated bounded secondary control

Decision: The real Colab GPU Graph B run (commit `1eb494aa`, run ID
`sprint4_graph_b_gcn_seed42_20260601`) has passed returned-artifact and
provenance validation. Graph B is included in the consolidated Sprint 4
comparison as a bounded secondary control only.

Reason: The returned Graph B artifacts include the canonical result CSV,
report, nine required figures, seven diagnostic tables, and a run directory
with `resolved_config.yaml`, `runtime.json`, `graph_artifact_provenance.json`,
`training_history.csv`, and an ignored `model.pt` checkpoint. The provenance
record confirms Scheme A, `sprint2_main_seed42`, strict-inductive visibility,
`zero_type_feature` target representation, and the expected Graph B counts:
`sgRNA=150`, `physical_target_site=9880`, `candidate_pair=11446`,
`sequence_similar_to=1208`.

Outcome:

- Graph B test AUPRC `0.9666`, test AUROC `0.7436`, test F1 `0.9486`, test
  MCC `0.1266`; positive prevalence `0.9007`.
- Graph B does not beat `xgboost_unweighted / F4` (test AUPRC `0.9925`).
- Graph B AUPRC is similar to Graph A (`0.9663`), confirming that guide-
  similarity topology alone does not substantially improve AUPRC over the
  minimal physical-target baseline.
- The low test MCC (`0.1266`) reflects the validation-selected threshold
  (`0.0785`) producing near-total positive classification; this is
  interpretation-only and did not drive any model or reporting decision.
- No model, schema, threshold, or feature choice was revised from Graph B
  test diagnostics.
- Consolidated Sprint 4 comparison artifacts updated under `outputs/sprint4/`
  to include Graph B as a bounded secondary control row.

## 2026-06-01 - Use a single fixed guide-disjoint split instead of k-fold cross-validation for GCN evaluation

Decision: Sprint 4 GCN models (Graph A, Graph B, Graph C) are evaluated on
the single fixed split `sprint2_main_seed42` rather than k-fold
cross-validation. Variance across seeds or folds is not reported.

Reason:

1. **Baseline comparison consistency.** All Sprint 2 baselines
   (`xgboost_unweighted / F4`, CNN/BiLSTM, MLP) were evaluated on the same
   locked split. Applying k-fold CV to GCN models while keeping the single-
   split baselines would make the comparison apples-to-oranges. Fair
   comparison requires the same evaluation protocol across all models; re-
   running all Sprint 2 baselines with CV would reopen the locked Sprint 2
   contract.

2. **GNN fold complexity.** For graph models, each fold requires a different
   graph materialization: the strict-inductive train/val/test views change,
   auxiliary edges (`sequence_similar_to`, `context_similar_to`) must
   respect the new fold's guide assignments, and Graph C's train-only
   preprocessing (median imputation, standard scaling) must be re-fit for
   each fold. This is significantly more complex than re-training a tabular
   model on different row subsets.

3. **Computational cost.** Five folds × three schemas = 15 GPU Colab runs
   versus the 3 runs executed. Each full GPU run takes multiple hours. This
   cost is not justified given that the primary scientific question (topology
   vs. context ablation) does not require variance estimates to reach a
   conclusion.

4. **Primary metric is threshold-free.** AUPRC — the primary metric — is
   insensitive to threshold selection and relatively stable across folds at
   this positive prevalence (~90%). The main conclusion (no GCN schema beats
   `xgboost_unweighted / F4`) holds across any reasonable fold partitioning
   given the ~0.03 AUPRC gap. Threshold-dependent metrics (MCC, F1) carry
   higher fold-to-fold variance and are therefore treated as secondary
   interpretation outputs only; this is consistent with Gao et al. (2020)
   who recommend PR-AUC over threshold-dependent metrics for imbalanced
   CRISPR off-target data.

Outcome:

- Single-seed, single-split evaluation is reported for all Sprint 4 GCN
  schemas.
- Variance is acknowledged as a thesis limitation; multi-seed or CV
  evaluation is deferred to future work.
- MCC results — especially Graph B's `test_mcc=0.127` — should be
  interpreted with caution as they are highly sensitive to the threshold
  selected from the validation set and the specific negative distribution of
  this split.
- Threshold-free AUPRC remains the authoritative comparison metric.

## 2026-06-03 - Sprint 5 primary epigenetic ablation uses fixed-topology Graph A

Decision: Sprint 5's primary biological ablation varies candidate-pair edge
feature tables on Graph A only. The graph topology, target-node semantics,
Scheme A label, `sprint2_main_seed42` guide-disjoint split, measured-only
universe, `experiment_id=18` exclusion, checkpoint policy, and validation-only
threshold policy remain fixed.

Reason: Graph A can isolate feature-family contribution because row-varying
sequence, mismatch, energy, experimental epigenetic, and computed nucleosome
features are candidate-edge tensors while physical target nodes remain
featureless zero/type representations. Graph C is not suitable as the primary
feature-ablation graph because its context-similarity topology and
observation-level target semantics already encode context; using it for
feature ablation would mix feature and topology effects.

Outcome:

- Sprint 5 adds `S5F0_seq` through `S5F5_computed_pos` feature tables under
  `data/processed/graphs/sprint5/graph_a_minimal_physical_target/`.
- `S5F0_seq` is true sequence-only guide/target one-hot input and does not
  include Sprint 4 `S1_pair`'s explicit mismatch channel.
- Full GPU execution uses `colab/sprint5_graph_a_feature_ablation_runner.ipynb`
  as a runner only.
- Reporting must include AUPRC as the primary metric plus AUROC, binary F1,
  macro F1, MCC, specificity, and TN/FP/FN/TP. Confusion matrices use each
  feature set's validation-selected threshold.

## 2026-06-03 - Add one Sprint 5B Graph C energy-focused secondary sensitivity

Decision: before moving to Sprint 6 imbalance experiments, add one predeclared
Sprint 5B run: fixed Graph C context-similarity topology and fixed
target-observation context semantics, with candidate-edge features set to
Sprint 5 `S5F2_energy`.

Reason: Sprint 5 Graph A results showed `S5F2_energy` as the strongest
candidate-edge feature setting, while raw experimental epigenetic and computed
context edge-feature additions did not improve the Graph A result. Sprint 4
Graph C still remains important because it represents context relationally
through target-observation nodes and context-similarity edges. A single
energy-focused Graph C run checks whether the best Sprint 5 edge setting is
compatible with Graph C's context representation before opening Sprint 6. This
is not hyperparameter tuning and not a Graph B/C feature-ablation ladder.

Outcome:

- Add `scripts/build_sprint5b_graph_c_energy_features.py` to materialize
  Graph C with the additional `S5F2_energy` candidate-edge feature table while
  preserving the established Graph C topology and target-observation features.
- Add `configs/sweeps/sprint5b_graph_c_energy_sensitivity.yaml` with
  `edge_feature_sets: [s5f2_energy]`, `weighted_bce`, `sprint2_main_seed42`,
  and the same headline guide-level evaluation contract.
- Add `colab/sprint5b_graph_c_energy_sensitivity_runner.ipynb` as a runner-only
  notebook. Full training must be executed in Colab; local tests validate the
  config, notebook contract, and model feature-tensor wiring.
- Interpret Sprint 5B as secondary sensitivity only. It must not replace the
  primary Sprint 5 Graph A ablation and must not be used to tune thresholds,
  features, topology, or hyperparameters from test diagnostics.

## 2026-06-03 - Sprint 5B Graph C energy sensitivity completed; move imbalance work to Sprint 6

Decision: treat the Sprint 5B Graph C energy-sensitivity run as completed
interpretation evidence, not as a trigger for more feature or hyperparameter
tuning in Sprint 5.

Reason: Sprint 5B tested the one predeclared question: whether the strongest
Sprint 5 Graph A candidate-edge setting (`S5F2_energy`) is compatible with the
established Graph C context-observation representation. The run improved Graph
C's threshold-free AUPRC relative to Sprint 4 Graph C, but did not outperform
Graph A `S5F2_energy` and showed poor negative-class recognition under the
validation-selected threshold. This points to imbalance/threshold/loss behavior
rather than missing feature families as the next controlled axis.

Outcome:

- Sprint 5B `GraphCContext+S5F2_energy` result:
  test AUPRC `0.972481`, test AUROC `0.836219`, test F1 `0.951878`,
  test macro F1 `0.552442`, test MCC `0.274287`, specificity `0.082840`,
  TN/FP/FN/TP `14/155/0/1533`.
- Relative to Sprint 4 Graph C, AUPRC improves from `0.961586` to `0.972481`.
  This supports that binding-energy edge features are useful in Graph C too.
- Relative to Sprint 5 Graph A `S5F2_energy`, AUPRC drops from `0.976585` to
  `0.972481`, and MCC drops from `0.477933` to `0.274287`. Graph C context
  representation does not add a clear advantage over the fixed-topology Graph A
  energy setting under the current GCN architecture.
- The lower MCC and macro F1 are explained by the validation-selected threshold
  classifying almost all test rows as positive: zero false negatives but only
  14 true negatives out of 169 negatives. AUPRC remains the primary metric, but
  this confusion profile strengthens the case for Sprint 6 imbalance,
  threshold, and loss analysis.
- Literature interpretation remains mixed: Mak et al. 2022 supports binding
  energy and computed nucleosome scores as meaningful model inputs, while raw
  experimental epigenetic scalars are weak. The current project result is
  consistent with strong binding-energy signal, but does not show that the
  current GCN formulation can exploit additional context features better than
  Graph A `S5F2_energy`.

## 2026-06-06 - Sprint 6 freezes Graph A + S5F2_energy and varies only loss/sampling

Decision: Sprint 6 (imbalance/loss comparison) holds the Sprint 5 best setting
fixed and treats the training objective as the only controlled variable. The
fixed setting is: graph schema `graph_a_minimal_physical_target`, feature set
`S5F2_energy` (268 edge-feature columns), `GraphAEdgeGCN` architecture (2-layer
GCNConv, hidden 128, LayerNorm+ReLU, dropout 0.2), AdamW, `ReduceLROnPlateau` on
`val_auprc`, grad clip 1.0, seed 42, split `sprint2_main_seed42`. Only the loss
function and/or training-time sampling change.

Reason: Sprint 5 established `S5F2_energy` as the strongest stable GCN feature
setting (test AUPRC `0.976585`, MCC `0.477933`, TN/FP/FN/TP `48/121/6/1527`) and
the strongest *non-degenerate* confusion profile among the feature ladder. To
isolate the loss effect cleanly, every other axis must be frozen, otherwise a
metric change cannot be attributed to the objective. `S5F2_energy` is also Graph A
topology, so this continues the Sprint 4 Graph A baseline lineage rather than
abandoning it. The Sprint 4 Graph A and `xgboost_unweighted / F4` rows remain the
comparison baselines; `S5F2_energy` is the carried-forward operating point. These
are different roles, not competing choices.

Outcome:

- Sprint 6 exec plan: `docs/exec-plans/completed/006-sprint6-imbalance-loss-comparison.md`.
- The frozen feature set must not be re-tuned from Sprint 6 loss diagnostics.
- Validation-only checkpoint (`val_auprc`) and threshold (`validation_max_f1`)
  selection are unchanged so all Sprint 6 rows stay same-contract comparable to
  Sprints 2-5 and to XGBoost F4.

## 2026-06-06 - Predeclared Sprint 6 loss set and hyperparameters (no test tuning)

Decision: Sprint 6 predeclares its full loss/sampling run list and all
hyperparameters before any training. Headline runs (all on the frozen Graph A +
`S5F2_energy` setting): `S6R0` weighted BCE (`pos_weight = negatives/positives`,
data-derived ≈0.1267); `S6R1` unweighted BCE (control); `S6R2` focal γ=2, α=0.25;
`S6R3` focal γ=1, α=0.25; `S6R4` focal γ=2, α=0.50; `S6R5` generalized Dice
(ε=1.0, class weights from train frequency); `S6R6` Tversky α=0.70, β=0.30
(ε=1.0); `S6R7` unweighted BCE + measured-only balanced supervised-edge
subsampling (1:1). Optional/approval-gated: `S6R8` class-balanced BCE (Cui 2019,
β=0.999), `S6R9` hard-negative mining.

Reason (literature basis, axis_2 notes + primary sources):

- Focal γ=2, α=0.25 is the experimentally validated default (Lin et al. 2017;
  robust over γ∈[0.5,5]; Guan et al. 2024 reports focal loss as the best/most
  stable cost-sensitive method across CRISPR off-target models). In the Guan
  eq.3 form, α weights the positive (y=1) term and (1-α) the negative (y=0) term,
  so α=0.25 places 0.75 weight on this project's rare negative class. α>0.5 was
  excluded because it would down-weight the rare negatives (wrong direction).
- Dice/generalized Dice (Sudre et al. 2017) is overlap-based and robust to
  learning rate, but is known to yield high precision / low recall on the rare
  class — which is exactly why Tversky is included as its targeted generalization.
- Cost-sensitive (loss) methods are preferred over resampling for deep nets with
  a small minority pool; the measured-only train set has only 901 negatives
  across 98 guides, so balanced sampling can only upsample those few negatives
  (overfitting risk). Sampling (`S6R7`) is therefore secondary, not headline.

Reason (Tversky direction is the deliberate inverse of the literature default):

- Tversky index `TI = TP/(TP + α·FP + β·FN)`, loss = 1-TI. The standard Salehi
  et al. 2017 recommendation is α=0.3, β=0.7 — it up-weights false negatives to
  recover the rare *positive/foreground* class, because Dice gives high
  precision / low recall on rare positives.
- This project is inverted: the rare class is the *negative*, and the failure
  mode is excess false positives (S5F2_energy: FP=121 vs FN=6; specificity
  collapses to ~0). Recovering the rare negative class requires penalizing FP
  more, so we set α=0.70, β=0.30 — the literature value flipped. This inversion
  is predeclared and justified by the confusion profile, not chosen from test
  results.

Outcome:

- Hyperparameters are frozen on review and cannot be changed from Sprint 6 test
  diagnostics. Every completed run is reported (winners and losers).
- The report must state AUPRC first (primary), then negative-class threshold
  metrics (specificity, TNR, MCC, macro F1), and must not present MCC/macro-F1
  gains as AUPRC gains.
- Residual threshold collapse across all losses would implicate architecture or
  feature distribution (edge features do not enter GCN message passing in the
  current model) rather than the loss alone, and points to Sprint 7.

Direction validation (literature review, 2026-06-06):

- These are segmentation/detection losses **adapted** to an inverted binary
  classification (rare class = negative), not reproductions of their source
  experiments.
- Tversky α>β reducing false positives / raising specificity is the documented,
  intended use of the parameter; α=0.70/β=0.30 is mathematically equivalent to
  Salehi's standard α=0.3/β=0.7 applied with the negative (minority) class as
  foreground. The inversion is therefore literature-endorsed, not ad-hoc.
- Focal is **not** inverted: γ is class-agnostic (focuses on hard/misclassified
  examples, here the rare negatives) and α=0.25 is kept (under the Guan eq.3 form
  it already up-weights the negative class). α=0.25 is a directionally-correct
  *transferred* value, not re-optimized for the inverted structure; `S6R4`
  (α=0.5) hedges α-sensitivity.
- Dice (`S6R5`) must be **generalized Dice** (inverse-volume class weights);
  plain single-class Dice on the majority-positive class is degenerate at ~90%
  prevalence.
- A unit-test **direction guard** must confirm that the implemented
  foreground/weight convention actually penalizes the FP-type errors (negatives
  predicted positive), since the effect depends on the implementation convention.

## 2026-06-06 - Sprint 6 headline stays measured-only; measured=0 screening regime deferred

Decision: Sprint 6's headline loss comparison uses the locked measured-only,
guide-level universe only. Any use of `measured=0` putative rows is deferred to a
separately named, approval-gated secondary track (`putative_augmented_screening`
/ `genome_wide_candidate_filtering`), not part of default Sprint 6 scope.

Reason:

- `measured=0` rows are the switch into the low-prevalence genome-wide screening
  regime described by Gao et al. 2020 and Guan et al. 2024. The measured-only
  universe is ~90% positive (negatives rare); adding the ~284K putative
  `measured=0` candidates flips the dataset to ~7% positive (negatives dominant,
  ≈1:13). Only in that second regime do the literature's positive-oversampling /
  SMOTE recommendations apply, because there the *positive* class is rare.
- Mixing `measured=0` into Sprint 6 would change two things at once (loss and
  data regime), making the loss effect unidentifiable, and would break
  same-contract comparability with XGBoost F4 and Sprints 2-5 (all measured-only).
- `measured=0` rows are putative/unmeasured candidates and must never be labeled
  true negatives or enter validation/test (Sprint 1 / Evaluation Protocol rule).

Alternatives considered:

- Add `measured=0` negatives via balanced sampling/hard-negative mining inside
  Sprint 6 to relieve negative scarcity — rejected (regime mixing, leakage of
  putative labels into the headline contract, loss of F4 comparability).
- Switch the whole project to the genome-wide regime — rejected; the measured-only
  benchmark is the project's defensible, leakage-controlled, experimentally
  labeled contract.

Outcome:

- If the screening regime is later explored, it is a separate track: `measured=0`
  used as training-only noisy negatives, never in validation/test, reported
  separately from the headline loss table, following the Sprint 5B secondary
  sensitivity precedent. The current benchmark must continue to be described as
  measured-only / guide-level / leakage-controlled, not as a full genome-wide
  off-target screening benchmark.

## 2026-06-06 - Sprint 6 headline loss comparison outcome

Decision: Keep weighted BCE (`S6R0_wbce`) as the headline Sprint 6 Graph A +
`S5F2_energy` objective reference. Do not revise the predeclared loss
hyperparameters from the returned test diagnostics, and do not promote optional
`S6R8`/`S6R9`/`S6S1` runs into the headline table.

Outcome:

- Colab batch `sprint6_loss_comparison_seed42_20260606_182812` completed exactly
  the predeclared headline runs `S6R0`-`S6R7`; returned artifacts validated under
  `outputs/sprint6/loss_comparison/`.
- AUPRC-first ranking: `S6R0_wbce` test AUPRC `0.976935`; `S6R7_balanced_sampling`
  `0.976205`; focal variants `0.956803`-`0.963497`; Tversky `0.955804`;
  generalized Dice `0.871174`.
- The best run is only `+0.000350` AUPRC over the Sprint 5 Graph A
  `S5F2_energy` reference (`0.976585`) and remains below `xgboost_unweighted` /
  F4 (`0.992522`) by `-0.015587`.
- Negative-class recognition remains limited under the validation-max-F1
  threshold: `S6R0` retrieves 49/169 negatives, `S6R7` retrieves 42/169, and
  generalized Dice retrieves 0/169. These threshold metrics are diagnostic and
  must not be reported as AUPRC gains.

Interpretation:

- The expected Gao/Guan-style imbalance benefit did not transfer cleanly to this
  measured-only headline regime because the class structure is inverted
  (positive prevalence `0.900705`; negatives are rare), the validation-max-F1
  threshold favors positive predictions, and the current `GraphAEdgeGCN` uses
  `S5F2_energy` only in the edge-classifier head rather than in message passing.
- Therefore residual threshold collapse is not attributed to loss alone. Further
  work should be framed as architecture/regime investigation (for example an
  edge-aware Sprint 7 or separately approved screening-regime Slice 5), not as
  post-hoc retuning of Sprint 6 losses.

## 2026-06-06 - Open optional Sprint 8 (Robustness); proceed to Sprint 7 next

Decision: defer the project's uncertainty/variance-quantification work to a new
OPTIONAL Sprint 8 ("Robustness") and proceed directly to Sprint 7 (GAT/GATv2)
without first retrofitting robustness across earlier sprints. Sprint 6 is treated
as complete at Slice 4 (headline validated); Slice 5 remains separately
approval-gated.

Reason: Slice 4 localized the binding constraint to architecture (in the current
`GraphAEdgeGCN`, `S5F2_energy` edge features enter only the classifier head, not
message passing). The most informative next experiment is therefore the
edge-aware architecture (Sprint 7), which is also a must-have on the critical
path. Robustness work (bootstrap CIs, paired comparisons, multi-seed) is valuable
but interpretation-only and does not block Sprint 7; bundling it as an optional
sprint keeps the roadmap moving while preserving the work.

Scope of Sprint 8 (predeclared, interpretation-only, no test-driven tuning):

- Guide-level (cluster) bootstrap CIs for all reported results (Sprints
  4/5/5B/6/7) from saved per-row predictions; no retraining. Resample guides, not
  rows (rows within a guide are correlated). AUPRC primary; threshold metrics at
  the frozen validation threshold. BCa preferred (AUPRC bounded near the
  `0.900705` floor); B >= 2000 (e.g. 5000).
- Paired-difference bootstrap for headline comparisons; overlapping independent
  CIs do not establish significance (overlap fallacy). Comparing to
  `xgboost_unweighted / F4` requires regenerating F4 per-row test predictions on
  the locked split (cheap, CPU, reproduces `0.992522`) — Sprint 2 did not save
  them.
- Multi-seed (fixed split) for headline model-selection configs only (best GCN vs
  GAT): predeclared seeds, report mean +/- std, no best-seed selection. May be
  run inline in Sprint 7 and consolidated in Sprint 8.

Alternatives considered:

- Retrofit bootstrap + multi-seed across all sprints now, before Sprint 7 —
  rejected: delays the architecture experiment that Slice 4 pointed to, and
  multi-seed on locked ablation cells reopens documented numbers for little
  information gain.
- Bake multi-seed into Sprint 7 only and skip a robustness sprint — viable; in
  that case Sprint 8 simply consolidates and adds bootstrap/paired CIs.

Outcome:

- `CRISPR_GNN_PROJECT_PLAN.md` adds Sprint 8 (optional/stretch) plus a Stretch
  bullet; `README.md` roadmap marks Sprint 6 complete, Sprint 7 next, Sprint 8
  optional.
- `scripts/compute_sprint6_bootstrap_ci.py` (guide-level cluster percentile
  bootstrap, B=2000) is the Sprint 6 prototype; Sprint 8 generalizes it (BCa,
  paired-difference, all sprints) into a tested `src/crispr_gnn/evaluation/`
  module.
- Multi-seed is NOT applied retroactively to locked ablation cells; guide-level
  bootstrap CIs (no retraining) provide the uniform uncertainty layer instead.
- Literature anchors: Boyd 2013 (AUPRC CIs); cluster/block bootstrap for
  correlated data; paired-difference bootstrap / overlapping-CI fallacy; BCa.

## 2026-06-10 - Re-scope Sprint 8 as model improvement (8A/8B); defer robustness to Sprint 9

Decision: Sprint 8 is re-scoped from "Robustness" to a small, predeclared,
mechanism-driven **model-improvement** sprint, split into Sprint 8A
(target-context + context-edge interaction) and Sprint 8B (sequence-context
encoder). The robustness work (guide-level bootstrap CIs, paired-difference
bootstrap, multi-seed fixed-split variance) is moved to a new **Sprint 9**. This
supersedes the 2026-06-06 "Open optional Sprint 8 (Robustness)" decision for the
sprint *numbering and scope* only; the robustness *methodology* in that entry is
unchanged and simply relabelled Sprint 9.

Reason: Sprint 7F localized the strongest same-contract GNN signal to
target-context representation (7D: direct `target_observation` features are
critical; 7E: experimental epigenetic features are necessary; 7F: a family-aware
encoder beat a larger unified-deep encoder, i.e. structure mattered more than raw
capacity). The most informative next step is therefore a small model-improvement
sprint on that axis, not robustness. Robustness is interpretation-only, requires
no architecture change, and does not block model improvement, so it consolidates
cleanly as a later Sprint 9.

Scope and locked decisions (predeclared before any Sprint 8 training; carried in
`docs/exec-plans/active/008-sprint8a-target-context-interaction.md` and
`...008b-sprint8b-sequence-context-encoder.md`):

- Frozen evaluation contract is inherited verbatim from Sprint 7F: `scheme_a`,
  `sprint2_main_seed42`, guide-disjoint, measured-only headline, `experiment_id=18`
  excluded, train-only preprocessing, validation-only checkpoint and threshold,
  no test-driven selection, AUPRC primary, XGBoost F4 bar test AUPRC `0.992522`,
  Graph C GATv2 + `S5F2_energy`, weighted BCE, seed 42, `context_similar_to`
  edges dropped.
- **Canonical base = Sprint 7F R3** (`family_aware_experimental_emphasis`, branch
  dims 24/48/40/16), chosen by the highest **validation** AUPRC (`0.987522`; R2
  `0.977541`, R1 `0.976594`) so the base selection never touches the test set.
  Sprint 7F R2 is retained only as a carry-forward rare-negative reference row,
  not as the base.
- **Sprint 8A run matrix (exactly 5 canonical Graph C GATv2 runs):** R0 base
  reference; R1 SENET-style learned family gate over the four target-context
  family branches (115/6/78/13); R2 head-only FiLM interaction (context embedding
  → γ,β → candidate `S5F2_energy` edge embedding) before classification; R3 gate
  + FiLM; R4 regularized experimental-epigenetic branch (bottleneck +
  feature-dropout). The **frozen GATv2 attention/message passing is unchanged** —
  the interaction is applied in the classifier head only (GNN-FiLM/ECC cited as
  principle, adapted head-only to preserve the contract). Interaction-MLP over
  `[edge, context, edge*context]` is the predeclared fallback if FiLM
  underperforms on validation; full bilinear is excluded (overfit risk at ~900
  train negatives / single seed).
- **Selection rule:** validation AUPRC primary, validation MCC/macro F1
  tie-break; test metrics reported only; every predeclared run reported;
  `parameter_count` reported next to performance as a capacity-confound control
  (Sprint 7F family-aware already beat the larger unified-deep encoder with fewer
  parameters). Optional axis-4 hyperparameter refinement is bounded, predeclared,
  validation-only, and applied only to the single validation-AUPRC winner — the
  sole sanctioned exception to the otherwise-frozen GATv2/training defaults.
- **Sprint 8B** re-implements a CRISPR-Net-adapted Conv+BiLSTM sequence encoder
  over the Sprint 2 `S1` sgRNA/target pair, trained **from scratch on the locked
  split**. Externally-pretrained CRISPR/genomic weights (CRISPR-Net/DeepCRISPR
  checkpoints; RNA-FM/DNABERT-2) must not be used as same-contract results
  (leakage); any transfer experiment is a separately-labelled, approval-gated
  slice. No reproduction claims (data/split/target/metric differ from the source
  papers).

Literature: a new axis `docs/literature/axes/axis_4_model_architecture_components/`
was added with the design-justification papers (4A SENet; 4B FiLM, GNN-FiLM,
FiBiNET; 4C Lengerich dropout-as-interaction-regularizer, Geirhos shortcut
learning; 4D Schneider et al. overtuning, Kapoor & Narayanan leakage; 4E Brody
GATv2 backbone, Dwivedi same-parameter-budget benchmarking), plus CRISPR-IP under
`axis_1/1B`. All are cited as adaptations/precedents, never as reproductions.

Outcome:

- `CRISPR_GNN_PROJECT_PLAN.md` and `README.md` roadmaps updated: Sprint 8 =
  model-improvement (8A/8B); Sprint 9 = robustness (optional/stretch). Robustness
  deliverables move from `outputs/sprint8/` to `outputs/sprint9/`.
- Sprint 8A outputs live under `outputs/sprint8a/`, Sprint 8B under
  `outputs/sprint8b/`.
- No label, split, dataset, loss, or evaluation-rule change — this is a
  roadmap/scope decision plus predeclared architecture deltas under the existing
  frozen contract.

## 2026-06-11 - Sprint 8A selects R2 by validation AUPRC; defer superiority to Sprint 9

Decision: close Sprint 8A Slice 6 with `S8A_R2_context_edge_film` as the
validation-selected canonical candidate, skip Slice 7 hyperparameter refinement
unless a separate preapproved methodological reason is added, and defer any
superiority claim to Sprint 9 robustness.

Reason:

- The predeclared selection rule was validation AUPRC. The authoritative
  consolidated batch `sprint8a_target_context_interaction_seed42_20260611_011416`
  selected R2 with validation AUPRC `0.987496`. R2's test diagnostics were AUPRC
  `0.982757`, AUROC `0.910575`, MCC `0.563656`, TN/FP/FN/TP `88/81/39/1494`.
- R2 improved the rare-negative operating point versus the Sprint 8A R0 harness
  base, but no Sprint 8A variant surpassed the carry-forward XGBoost F4 bar
  (test AUPRC `0.992522`). R0 also did not reproduce the S7F R3 carry-forward
  reference exactly, so single-seed harness variance remains a material
  interpretation risk.
- R3 (`gate+FiLM`) underperformed R2 because it lost rare negatives at the
  validation-selected operating point (`R2->R3`: `TN->FP=52`, `FP->TN=1`,
  `FN->TP=25`, `TP->FN=17`) and also had lower ranking metrics. Aggregate gate
  weights did not collapse; the result is best interpreted as gate+FiLM
  interference/calibration degradation in this frozen setup, not evidence that
  gating is globally harmful.
- R4 (regularized experimental branch) gained some negatives relative to R0
  (`FP->TN=48`) but created many false negatives (`TP->FN=88`), concentrated in
  a few positive-heavy guides. The available outputs do not prove that
  experimental epigenetic features are shortcuts; they show that this bottleneck
  + dropout regularizer was too blunt or unstable for the single-seed canonical
  setting.
- Following the overtuning/leakage discipline already documented for Sprint 8,
  a post-result HP refinement would risk optimizing to the observed validation/
  test behavior. Sprint 9 should instead predeclare fixed configs, seeds, and
  paired/guide-level uncertainty analysis.

Outcome:

- Treat R2 as the Sprint 8A candidate for robustness, not as a final superior
  model.
- Do not change labels, split, threshold policy, model code, or canonical Sprint
  8A outputs based on these diagnostics.
- Sprint 9 should test R2 against the frozen base/reference with predeclared
  multi-seed fixed-split runs and paired guide-level/bootstrap diagnostics.
- Record the parameter-count reporting caveat separately as tech debt: in
  interaction mode the nominal parameter count includes an inactive base edge
  classifier, so reported R2/R3 capacity overstates active parameters.
