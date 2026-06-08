# Execution Plan: Sprint 7 Edge-Aware GAT / GATv2 Architecture Ablation

> Status: COMPLETED. Sprint 7 code, config, runner-only Colab notebook,
> contract tests, Colab headline run, returned outputs, and final report are in
> place. Sprint 7 preserved the Sprint 6 weighted-BCE Graph A contract and
> completed the predeclared Graph A GCN/GAT/GATv2 architecture comparison.

## 1. Goal

Sprint 7 tests whether routing the Sprint 5 best candidate-edge feature family
(`S5F2_energy`) into attention/message passing improves the fixed Graph A GCN
operating point under the unchanged guide-level evaluation contract.

Scientific hypothesis:

> Under the frozen Scheme A, guide-disjoint, measured-only protocol, an
> edge-aware Graph A GAT/GATv2 model that consumes `S5F2_energy` candidate-edge
> features inside attention/message passing may improve guide-level AUPRC and/or
> rare-negative recognition relative to the Sprint 6 weighted-BCE Graph A GCN
> reference.

The hypothesis is intentionally narrow. Sprint 7 is a controlled architecture
ablation, not a feature, loss, sampler, graph-schema, sequence-encoder, or data
regime sweep.

Sprint 7 will:

- Preserve Graph A topology, `S5F2_energy`, weighted BCE, split, label, and
  evaluation policy.
- Implement an edge-aware Graph A GAT model using PyG `GATConv`.
- Implement an edge-aware Graph A GATv2 model using PyG `GATv2Conv`, if the
  implementation is technically stable before headline test inspection.
- Keep `S5F2_energy` in the final edge-classifier input for parity with the
  current GCN, while additionally passing the same edge features into attention.
- Compare against the frozen `S6R0_wbce` GCN reference and the authoritative
  `xgboost_unweighted / F4` reference.
- Generate attention summaries as model-interpretation diagnostics only.

Sprint 7 will not:

- Change labels, split, measured-only universe, graph schema, feature set,
  preprocessing, threshold policy, checkpoint policy, optimizer family, loss, or
  sampler.
- Add CRISPR-Net-style CNN/RNN sequence encoders, larger sequence modules, or
  sequence-fusion architecture changes. If needed, those belong to a later
  Sprint 7B or stretch plan after the attention result is known.
- Add Graph C, Graph B, GraphSAGE, R-GCN, HGT, heterogeneous GNNs, graph
  transformers, or secondary-structure graph features.
- Use `measured=0` putative rows in headline train/validation/test.
- Run multi-seed robustness as a Sprint 7 acceptance requirement. Multi-seed,
  guide-level bootstrap CIs, and paired-difference CIs remain Sprint 8 /
  robustness work. Sprint 7 single-seed results must be framed as an architecture
  ablation, not as final statistical superiority.
- Tune architecture, heads, dropout, thresholds, epochs, or reporting choices
  from test diagnostics.

## 2. Inputs And References

Required project inputs:

- Sprint 6 handoff: `outputs/sprint6/loss_comparison/`.
- Sprint 6 completed plan:
  `docs/exec-plans/completed/006-sprint6-imbalance-loss-comparison.md`.
- Frozen split artifacts: `outputs/splits/sprint2_guides.json` and
  `outputs/splits/sprint2_split_summary.csv`.
- Sprint 5 Graph A `S5F2_energy` graph artifacts:
  `data/processed/graphs/sprint5/`.
- Current GCN model/trainer:
  `src/crispr_gnn/models/gcn.py` and `src/crispr_gnn/training/gcn.py`.
- Loss registry and sampler decisions from Sprint 6:
  `src/crispr_gnn/models/losses.py` and `src/crispr_gnn/training/samplers.py`.
- Evaluation and reporting policy:
  `docs/EVALUATION_PROTOCOL.md`, `docs/DECISIONS.md`, and
  `CRISPR_GNN_PROJECT_PLAN.md`.

Frozen reference metrics:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | --- |
| `xgboost_unweighted` | F4 tabular baseline | `0.992522` | `0.938416` | `0.345198` | `38/131/21/1512` |
| `S6R0_wbce` | Graph A + `S5F2_energy` + weighted BCE | `0.976935` | `0.819972` | `0.483719` | `49/120/6/1527` |
| `S6R7_balanced_sampling` | Graph A + `S5F2_energy` + balanced sampling | `0.976205` | `0.815167` | `0.447602` | `42/127/5/1528` |

The locked test positive prevalence is `0.900705`. AUPRC has a high prevalence
floor in this measured-only universe; specificity, TNR, MCC, macro F1, and
TN/FP/FN/TP are secondary diagnostics, not primary ranking criteria.

## 3. Sprint 6 Handoff Interpretation

Sprint 6 completed the predeclared loss/sampling comparison on fixed Graph A +
`S5F2_energy`. No objective beat weighted BCE on primary AUPRC, and weighted BCE
also had the best negative-class recognition among the headline runs.

Important Sprint 6 conclusion:

- Residual threshold collapse must not be attributed to loss alone.
- In current `GraphAEdgeGCN`, candidate-edge features, including
  `S5F2_energy`, are concatenated only at the final edge-classifier head.
- Those edge features do not participate in `GCNConv` message passing.
- The next controlled axis is therefore architecture / edge-feature flow.

This is the reason Sprint 7 focuses on edge-aware GAT/GATv2 rather than another
loss, sampler, feature, or threshold policy.

## 4. Literature And API Basis

Repository-local literature notes:

- Kipf & Welling, 2017, "Semi-Supervised Classification with Graph
  Convolutional Networks" motivates GCN-style normalized neighborhood
  aggregation. This supports the Sprint 4-6 GCN baseline lineage.
- Vinodkumar, Ozcinar & Anbarjafari, 2021, "Prediction of sgRNA Off-Target
  Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network"
  motivates CRISPR off-target graph/link-prediction framing. Its dataset,
  split, and metrics are not reproduced here.
- Jiang, Li, Xiong & Liu, 2025, "Graph-CRISPR: a gene editing efficiency
  prediction model based on graph neural network with integrated sequence and
  secondary structure feature extraction" motivates graph/attention-style CRISPR
  modeling and interpretation artifacts. It is not this project's off-target
  Scheme A guide-level task and must not be treated as a direct benchmark.

External primary references to cite in Sprint 7 reports:

- Velickovic et al., 2018, "Graph Attention Networks" (`arXiv:1710.10903`).
  GAT learns attention-weighted neighbor aggregation rather than using only
  GCN-style fixed normalized aggregation.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention Networks?"
  (`arXiv:2105.14491`). GATv2 addresses the static-attention limitation of
  original GAT by using dynamic attention.

PyTorch Geometric API facts verified for the local dependency (`torch_geometric`
2.7.0) and current official docs:

- `GATConv` and `GATv2Conv` accept `edge_dim` and `edge_attr`; edge features are
  part of attention coefficient computation when supplied.
- Both layers default to `add_self_loops=True`.
- Both layers default to `fill_value="mean"` for self-loop edge features when
  `edge_dim` is set.
- Sprint 7 must override the default self-loop `fill_value`; see Section 7.

No local dedicated GAT/GATv2 paper note was found under `docs/literature/`.
Sprint 7 may cite the canonical papers directly without claiming a local note or
paper reproduction.

## 5. Frozen Evaluation Contract

Sprint 7 preserves the Sprint 2/3/4/5/6 contract:

- Label scheme: Scheme A, `int(cleavage_freq > 1e-5)`.
- NaN `cleavage_freq` rows excluded from supervised labels.
- Negative `cleavage_freq` values remain below-threshold labels; values above 1
  remain positive labels for binary classification.
- Split: `sprint2_main_seed42`.
- Train, validation, and test guides are disjoint.
- Headline train/validation/test are measured-only.
- `experiment_id=18` excluded.
- No `measured=0` rows in headline train/validation/test.
- Graph visibility: `strict_inductive_primary`.
- Graph schema: `graph_a_minimal_physical_target`.
- Target-node representation: `zero_type_feature`; no learned per-target-ID
  embeddings and no row-varying context moved onto physical target nodes.
- Feature set: `S5F2_energy` only, 268 candidate-edge columns.
- Train-only imputation/scaling/preprocessing inherited from Sprint 5 feature
  artifacts; no val/test-fitted preprocessing.
- Loss: weighted BCE with train-derived `pos_weight = negatives / positives`
  (about `0.1267` in the Sprint 6 run).
- Optimizer/training defaults: AdamW, learning rate `1e-3`, weight decay
  `1e-4`, `ReduceLROnPlateau` on `val_auprc`, gradient clip `1.0`, dropout
  policy predeclared in config.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Test split used only for final reporting.
- Primary metric: AUPRC.
- Required references: `S6R0_wbce` and `xgboost_unweighted / F4`.

## 6. Predeclared Sprint 7 Run Matrix

Headline core runs:

| Run ID | Architecture | Edge-aware attention? | Final classifier edge features? | Role |
| --- | --- | ---: | ---: | --- |
| `S7R0_gcn_reference` | Existing `S6R0_wbce` Graph A GCN result | No | Yes | Frozen reference row, consumed from Sprint 6 artifacts; do not retrain unless a later plan explicitly requires a fresh same-contract reproduction. |
| `S7R1_gat_edge_aware` | Graph A `GATConv` | Yes | Yes | Primary Sprint 7 architecture test. |
| `S7R2_gatv2_edge_aware` | Graph A `GATv2Conv` | Yes | Yes | Dynamic-attention variant, included if implementation is technically stable before headline test inspection. |

Optional control, not required for Sprint 7 closure:

| Run ID | Architecture | Edge-aware attention? | Final classifier edge features? | Role |
| --- | --- | ---: | ---: | --- |
| `S7R3_gat_edge_blind_control` | Graph A `GATConv` | No | Yes | Tests attention over the current weak Graph A node/topology representation without edge features in message passing. Run only if time/runtime permits; failure to run does not block Sprint 7. |

Interpretation rules:

- If `S7R1` or `S7R2` improves AUPRC over `S6R0`, report a Sprint 7 GNN
  architecture improvement, while still comparing to F4 XGBoost.
- If specificity/MCC improves but AUPRC does not, report this as a
  threshold-dependent diagnostic improvement, not as a primary win.
- If GAT/GATv2 do not improve, the result is still useful: it suggests the
  remaining bottleneck is not solved by edge-aware attention under Graph A.
- If GATv2 improves over GAT, frame this as evidence that dynamic attention is a
  better candidate for this graph setting, pending Sprint 8 robustness.

## 7. Edge-Aware Architecture Contract

The edge-aware Sprint 7 architecture must satisfy all of the following:

1. **Use the same Graph A node layout as GCN.**
   - Homogeneous node tensor layout remains:
     `sgRNA [0..n_sgrna-1]`, `physical_target_site [n_sgrna..total-1]`.
   - Physical target nodes remain featureless except for the shared zero/type
     target representation policy already used by `GraphAEdgeGCN`.

2. **Preserve candidate edge direction policy from GCN.**
   - Current GCN builds a homogeneous candidate edge index by adding both
     `sgRNA -> physical_target_site` and `physical_target_site -> sgRNA`.
   - Sprint 7 must keep this bidirectional candidate-edge policy for parity.

3. **Duplicate real candidate-edge features for reverse edges.**
   - Forward real candidate edge:
     `sgRNA -> physical_target_site` receives the row's `S5F2_energy`.
   - Reverse real candidate edge:
     `physical_target_site -> sgRNA` receives the same `S5F2_energy` vector.
   - This duplication is label-free and only preserves the existing undirected
     message-passing view.

4. **Keep self-loops but explicitly zero-fill self-loop edge attributes.**
   - Use `add_self_loops=True`.
   - Use `fill_value=0.0` / all-zero edge-feature vector for synthetic
     self-loop edges.
   - Do not use PyG default `fill_value="mean"`.
   - Reason: `S5F2_energy` is a candidate-pair feature. A self-loop has no real
     sgRNA-target binding-energy meaning. Mean-filled self-loops would inject
     synthetic node-level aggregate energy summaries and confound the
     architecture ablation.

5. **Keep direct edge features in the edge classifier.**
   - The final link classifier input remains comparable to the current GCN:
     source embedding, target embedding, elementwise product, absolute
     difference, and `S5F2_energy`.
   - Edge-aware attention is additive to this access path, not a replacement.

6. **Record parameter counts.**
   - GAT/GATv2 may have more parameters than GCN depending on heads/concat.
   - Result rows and reports must include parameter counts or a clear capacity
     note so gains are not over-attributed to attention alone.

7. **Attention extraction is reporting-only.**
   - If attention weights are returned, store/aggregate them for diagnostics.
   - Do not use attention summaries to select the architecture, heads, epochs,
     features, thresholds, or hyperparameters.

## 8. Architecture Hyperparameters To Predeclare

Before any headline run, the Sprint 7 implementation slice must freeze:

- `conv_type`: `gat` or `gatv2`.
- `edge_aware`: true for headline runs.
- `edge_dim`: expected `268` for `S5F2_energy`.
- `add_self_loops`: true.
- `self_loop_edge_fill`: `0.0`.
- `reverse_edge_attr_policy`: duplicate candidate-edge `S5F2_energy`.
- `hidden_dim`: initially keep `128` unless a capacity-matching reason is
  documented before test inspection.
- `num_layers`: initially keep `2` for parity with GCN.
- `heads`: predeclare a modest value that keeps output dimensionality and memory
  bounded; do not tune from test.
- `concat`: predeclare whether heads concatenate or average; if concatenating,
  choose per-head output dimensions so the post-conv embedding width remains
  comparable to GCN where feasible.
- `attention_dropout`: predeclare separately from feature dropout if exposed.
- `residual`: predeclare; do not silently turn on residual connections as a
  test-driven stabilizer.
- `share_weights` for GATv2: predeclare false unless a specific parity argument
  is documented.

If a hyperparameter must change for a technical reason, document it before any
headline test result is inspected.

## 9. Implementation Scope

Expected new files:

- `src/crispr_gnn/models/gat.py`:
  - `GraphAEdgeGAT` / `GraphAEdgeGATv2` or one shared configurable class.
  - Edge-aware homogeneous edge-index + edge-attribute construction helpers.
  - Attention-weight return path for diagnostics.
  - Contract checks for Graph A, `S5F2_energy`, edge dimensions, self-loop fill,
    and target-node representation.
- `configs/sweeps/sprint7_gat_gatv2.yaml`:
  - Frozen base config.
  - Predeclared run matrix.
  - GAT/GATv2 architecture hyperparameters.
  - Explicit self-loop and reverse-edge policies.
- `scripts/run_sprint7_gat_comparison.py`:
  - Similar to Sprint 6 runner, but controlled variable is architecture only.
  - Consumes Sprint 5 Graph A artifacts and Sprint 6 reference rows.
  - Writes per-run resolved config, runtime, training history, metrics,
    predictions, and optional checkpoint paths.
- `colab/sprint7_gat_gatv2_runner.ipynb`:
  - Runner-only notebook. No model/training/evaluation logic inside notebook.

Expected modified files:

- `src/crispr_gnn/training/gcn.py` or a new architecture-neutral trainer module:
  - Add model dispatch for GAT/GATv2 without changing the GCN behavior.
  - Keep weighted BCE, validation checkpoint, validation threshold, and result
    schema stable.
- `src/crispr_gnn/evaluation/diagnostics.py` and/or `plots.py`:
  - Add Sprint 7 attention summaries and comparison figures as additive
    functions; do not alter Sprint 4-6 outputs.
- Tests under `tests/`:
  - Architecture contract, edge-attribute alignment, self-loop policy, config
    gating, output contract, and weighted-BCE/validation-policy preservation.

Do not modify raw data, graph artifacts, Sprint 6 outputs, or completed Sprint
4-6 reports as part of Sprint 7 implementation.

## 10. Output Contract

Canonical Sprint 7 outputs:

```text
outputs/sprint7/gat_comparison.csv
outputs/sprint7/gat_report.md
outputs/sprint7/gat_run_manifest.json
outputs/sprint7/graph_artifact_provenance.json
outputs/sprint7/diagnostics/
outputs/sprint7/figures/
outputs/sprint7/runs/<run_id>/{resolved_config.yaml,runtime.json,training_history.csv,metrics.csv,model.pt}
```

Required figures:

```text
outputs/sprint7/figures/gat_model_auprc_comparison.png
outputs/sprint7/figures/gat_pr_curves.png
outputs/sprint7/figures/gat_roc_curves.png
outputs/sprint7/figures/gat_training_curves.png
outputs/sprint7/figures/gat_threshold_metrics.png
outputs/sprint7/figures/gat_score_distributions.png
outputs/sprint7/figures/gat_per_guide_metric_distribution.png
outputs/sprint7/figures/attention_weight_summary.png
```

Required diagnostic tables:

- Consolidated result table with `S6R0_wbce` and F4 references.
- Per-run predictions for validation and test.
- Training history.
- Fixed-threshold metrics using validation-selected thresholds.
- Per-guide and per-genome metrics.
- Score deciles or score-direction diagnostics.
- Attention summaries separated by:
  - architecture,
  - layer,
  - head,
  - split,
  - candidate forward edges,
  - candidate reverse edges,
  - self-loop edges.

Reports and figures must show the positive prevalence context (`0.900705`) and
must keep AUPRC-first interpretation.

## 11. Tests And Contract Guards

Model/edge tests:

- Edge-aware GAT/GATv2 receives `edge_attr` with one row per homogeneous edge.
- Forward and reverse candidate edges receive duplicated `S5F2_energy`.
- Self-loop edge attributes are all-zero vectors.
- PyG default `fill_value="mean"` is not used in configured edge-aware models.
- Edge-aware attention has `edge_dim == 268` for `S5F2_energy`.
- Graph A physical target nodes remain featureless; no per-target ID embeddings.
- Final classifier still receives `S5F2_energy` directly.

Trainer/config tests:

- GCN weighted-BCE path remains unchanged.
- Sprint 7 configs reject non-Graph-A schemas, non-`S5F2_energy` feature sets,
  non-weighted-BCE losses, changed split IDs, changed label schemes, and
  measured-zero headline regimes.
- Checkpoint selection remains validation AUPRC.
- Threshold selection remains validation max-F1.
- Optional edge-blind control cannot replace edge-aware headline runs.

Output tests:

- Required CSV/report/manifest/provenance/diagnostics/figures are produced.
- Result table includes `baseline_reference = xgboost_unweighted / F4`.
- Result table includes the frozen `S6R0_wbce` reference or explicitly records
  its source path.
- Attention summaries separate real candidate edges from self-loops.

Run before any headline Colab run:

```bash
uv run pytest -q tests/test_sprint7_gat_model.py tests/test_sprint7_gat_runner.py tests/test_gcn_training_smoke.py tests/test_gcn_evaluation_contract.py
uv run ruff check scripts src tests
```

Then run one CPU smoke/debug Sprint 7 run with `--max-epochs 1` into a
non-canonical output directory. Do not report smoke/debug metrics as Sprint 7
evidence.

## 12. Colab Runner Workflow

Colab remains a runner only:

1. Clone or checkout the approved commit.
2. `uv sync`.
3. Copy existing Sprint 5 Graph A artifacts into `data/processed/graphs/sprint5/`.
4. Validate graph artifact provenance before training.
5. Run the Sprint 7 comparison command.
6. Copy `outputs/sprint7/` back to durable Drive storage.
7. Return outputs for local validation.

Expected command shape:

```bash
uv run python scripts/run_sprint7_gat_comparison.py \
  --config configs/sweeps/sprint7_gat_gatv2.yaml \
  --run-id sprint7_gat_gatv2_seed42_<timestamp>
```

The notebook must not implement losses, samplers, model classes, attention
aggregation, evaluation, plotting, or artifact validation logic.

## 13. Interpretation Rules

Use these rules in the Sprint 7 report:

- AUPRC is the primary metric and determines headline ranking.
- MCC, specificity/TNR, macro F1, and TN/FP/FN/TP diagnose rare-negative
  recognition but do not override AUPRC.
- Any GAT/GATv2 improvement that remains below F4 XGBoost should be reported as
  "best GNN improved, but strongest tabular F4 baseline remains ahead."
- Small single-seed deltas must be described as tentative pending Sprint 8
  robustness. Sprint 7 does not establish final statistical superiority.
- Attention weights are model signals only. They must not be described as
  biological causal evidence for cleavage, chromatin, or binding mechanisms.
- Do not claim reproduction of Kipf & Welling, Vinodkumar et al.,
  Velickovic et al., Brody et al., Jiang et al., Mak et al., Gao, or Guan unless
  dataset, target, split, metrics, and architecture match their setup.

## 14. Risks

1. **Edge-aware wiring risk:** if `S5F2_energy` is not actually passed to
   `GATConv`/`GATv2Conv` as `edge_attr`, Sprint 7 will not test the Sprint 6
   architecture hypothesis.
2. **Self-loop confound:** PyG default `fill_value="mean"` would synthesize
   node-level aggregate energy features. Sprint 7 must use zero-filled
   self-loop edge attributes.
3. **Reverse-edge misalignment:** homogeneous reverse edges must have duplicated
   edge features in the same order as the reverse edge index.
4. **Capacity confound:** multi-head GAT/GATv2 can be larger than GCN. Parameter
   counts and hyperparameters must be reported.
5. **Single-seed fragility:** Sprint 7 core is single-seed by scope decision.
   Small deltas require Sprint 8 robustness before strong claims.
6. **High-prevalence metric interpretation:** at test prevalence `0.900705`,
   AUPRC has a high floor and threshold metrics can swing with few negatives.
7. **Attention overclaim:** attention summaries may be visually compelling but
   remain interpretation-only.
8. **Scope creep:** adding sequence encoders, new losses, Graph C, or
   measured-zero screening would make the architecture effect unidentifiable.

## 15. Acceptance Criteria

Sprint 7 planning acceptance:

- This plan is reviewed and frozen before implementation begins.
- The predeclared run matrix, self-loop policy, reverse-edge policy, and
  architecture hyperparameters are recorded before headline training.
- Any intentional deviation from the older project-plan multi-seed line is
  documented as a Sprint 7 scope decision: single-seed architecture ablation now,
  Sprint 8 robustness later.

Sprint 7 implementation acceptance:

- Edge-aware GAT and, if technically stable, edge-aware GATv2 are implemented and
  tested under the frozen contract.
- `S5F2_energy` enters both attention/message passing and the final classifier
  for headline edge-aware runs.
- No test-driven tuning is performed.
- Required diagnostics, figures, report, manifest, and provenance are present.
- Report compares against `S6R0_wbce` and `xgboost_unweighted / F4`.
- Report preserves AUPRC-first interpretation and attention interpretation
  boundaries.

## 16. Implementation Slices

### Slice 0 - Planning freeze

Status: complete.

Review and freeze this execution plan. Record any final design decisions for:
run matrix, edge-aware policy, self-loop fill, reverse-edge duplication,
heads/concat/capacity, and Sprint 8 boundary.

Exit: plan frozen; no code changed yet.

### Slice 1 - Edge-aware GAT model foundation

Status: complete.

Add Graph A GAT/GATv2 model classes and edge-index/edge-attr construction
helpers. Add focused tests for Graph A contract, edge_attr alignment,
self-loop zero-fill, reverse-edge duplication, final classifier parity, and
attention-return shapes.

Exit: model tests pass; no canonical training run.

### Slice 2 - Trainer/config dispatch

Status: complete.

Wire GAT/GATv2 into the existing graph trainer or a small architecture-neutral
trainer while preserving weighted BCE, validation checkpointing, validation
thresholding, and GCN behavior. Add Sprint 7 config and config-gating tests.

Exit: CPU smoke run can train one tiny/debug run; GCN regression tests still
pass; no headline claim.

### Slice 3 - Reporting and attention diagnostics

Status: complete.

Add Sprint 7 comparison report, figures, diagnostics, attention summaries, and
output-contract tests. Ensure attention summaries separate candidate forward,
candidate reverse, and self-loop edges.

Exit: mocked/smoke outputs satisfy the reporting contract; no headline claim.

### Slice 4 - Colab runner preparation

Status: complete.

Add a runner-only Colab notebook and documented command path. Validate graph
artifact provenance before any full training.

Exit: runner and local tests pass; no full GPU claim yet.

### Slice 5 - Full headline run and local validation

Status: complete.

Run the predeclared headline architecture comparison on Colab GPU, copy outputs
back, and validate locally. No reruns or hyperparameter changes from test
diagnostics.

Exit: all required Sprint 7 outputs exist or any technical omission is
documented before interpreting results.

### Slice 6 - Sprint closure

Status: complete.

Freeze report/results/status docs and move this plan to
`docs/exec-plans/completed/`.

Exit: Sprint 7 conclusion is documented as one of:

- edge-aware attention improves the best GNN setting;
- edge-aware attention improves only secondary threshold metrics;
- edge-aware attention does not improve the GCN reference;
- GATv2 is the preferred attention candidate but requires Sprint 8 robustness.
