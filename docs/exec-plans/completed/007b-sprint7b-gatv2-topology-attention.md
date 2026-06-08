# Execution Plan: Sprint 7B GATv2 Topology-Attention Interaction

> Status: COMPLETED. Sprint 7B code, Graph B S5F2 artifact builder, configs,
> runner-only Colab notebook, contract tests, Colab headline run, returned
> outputs, and final report are in place. Sprint 7B completed the predeclared
> Graph B / Graph C GATv2 topology-attention comparison.

## 1. Goal

Sprint 7B tests whether the stronger Sprint 7 attention candidate, GATv2,
becomes useful when applied to richer pre-existing graph formulations rather
than the sparse/minimal Graph A physical-target graph.

Scientific hypothesis:

> Under the frozen Scheme A, guide-disjoint, measured-only protocol, GATv2 may
> improve matched-schema GCN references when the graph contains non-candidate
> relational structure: guide-similarity edges in Graph B or context-observation
> / context-similarity structure in Graph C.

Sprint 7 already answered the narrow Graph A question: edge-aware Graph A
GAT/GATv2 with `S5F2_energy` in message passing did not beat the Sprint 6
weighted-BCE Graph A GCN reference. Sprint 7B does not reinterpret that result.
It asks a separate topology-attention interaction question.

Sprint 7B will:

- Preserve the frozen label, split, measured-only universe, loss, checkpoint,
  threshold, and AUPRC-first evaluation contract.
- Use GATv2 only in the core run matrix. Do not rerun original GAT.
- Build or attach a new Graph B `S5F2_energy` candidate-edge feature artifact so
  Graph B GCN and Graph B GATv2 use the same feature family.
- Train a matched Graph B GCN `S5F2_energy` reference.
- Train Graph B GATv2 `S5F2_energy` against that matched reference.
- Carry forward the existing Sprint 5B Graph C `GraphCContext+S5F2_energy` GCN
  reference unless a technical compatibility issue requires a documented rerun.
- Train Graph C GATv2 on `GraphCContext+S5F2_energy`.
- Compare all graph results to the authoritative `xgboost_unweighted / F4`
  baseline.
- Produce report-ready metrics, diagnostics, figures, attention summaries, and
  graph artifact provenance.

Sprint 7B will not:

- Tune Sprint 7 attention hyperparameters from the failed Graph A test result.
- Add GAT, GraphSAGE, R-GCN, HGT, graph transformers, or HeteroConv in the core.
- Add new losses, samplers, measured-zero screening, sequence encoders, or new
  feature families.
- Add continuous similarity-edge feature engineering in the core.
- Describe Graph C as topology-only.
- Treat attention weights as biological causal evidence.
- Claim final statistical superiority from one seed.

## 2. Frozen Evaluation Contract

Sprint 7B inherits the Sprint 2/3/4/5/6/7 headline contract:

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`.
- Split ID: `sprint2_main_seed42`.
- Guide-disjoint split.
- `experiment_id=18` excluded.
- Train/validation/test headline universe: measured-only.
- Test rows: measured-only.
- Feature preprocessing: train-only fit for imputation/scaling where applicable.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- No test tuning.
- Primary metric: AUPRC.
- Secondary diagnostics: AUROC, F1, macro F1, MCC, specificity/TNR,
  sensitivity, TN/FP/FN/TP, score deciles, per-guide behavior, per-genome
  behavior.
- Test positive prevalence context: `0.900705`; negatives are the rare class.
- Authoritative external baseline: `xgboost_unweighted / F4`, test AUPRC
  `0.992522`, AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

Sprint 7B keeps the Sprint 6 best loss:

- Loss: `weighted_bce`.
- `pos_weight`: `auto` from the train labels.
- Optimizer family: AdamW.
- Scheduler: ReduceLROnPlateau on validation AUPRC.
- Gradient clipping: `1.0`.
- Seed: `42` for core single-seed exploration.

Sprint 8 remains the robustness / uncertainty layer: multi-seed, guide-level
bootstrap CIs, paired-difference intervals, and no best-seed selection.

## 3. Prior Result Context

Frozen references:

| Reference | Setting | Test AUPRC | Test AUROC | Test MCC | Specificity | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `xgboost_unweighted` | F4 tabular baseline | `0.992522` | `0.938416` | `0.345198` | n/a | `38/131/21/1512` |
| `S6R0_wbce` / `S7R0_gcn_reference` | Graph A + `S5F2_energy` + weighted-BCE GCN | `0.976935` | `0.819972` | `0.483719` | `0.289941` | `49/120/6/1527` |
| `S7R2_gatv2_edge_aware` | Graph A + `S5F2_energy` + edge-aware GATv2 | `0.965449` | `0.818282` | `0.291367` | `0.218935` | `37/132/35/1498` |
| Sprint 5B Graph C GCN | `GraphCContext+S5F2_energy` | `0.972481` | `0.836219` | `0.274287` | n/a | `14/155/0/1533` |
| Sprint 4 Graph B GCN | Graph B + `S1_pair+F1` | `0.966570` | `0.743586` | `0.126559` | n/a | `3/166/0/1533` |

Sprint 7 result:

- Graph A GAT degraded strongly.
- Graph A GATv2 was stronger than GAT but below the weighted-BCE Graph A GCN.
- Technical sanity checks passed: correct graph, split, `S5F2_energy` feature
  table, weighted BCE, validation-only checkpoint/threshold, and active
  edge-aware PyG parameters.

Sprint 7B therefore asks whether attention needs richer topology/semantics than
Graph A provides.

## 4. Graph Definitions And Scope Boundaries

### Graph A

`graph_a_minimal_physical_target` is the minimal bipartite graph:

- Node types: `sgRNA`, `physical_target_site`.
- Candidate-pair edges connect sgRNA nodes to physical target nodes.
- Physical target nodes are featureless in the model and use a shared learned
  type vector.
- Candidate-pair features live on edges.

Graph A is not retrained in Sprint 7B. Sprint 6/Sprint 7 carry-forward rows
provide context only.

### Graph B

`graph_b_guide_similarity_control` is a bounded secondary control derived from
Graph A:

- Keeps featureless `physical_target_site` nodes.
- Keeps candidate-pair edges.
- Adds label-free sgRNA-to-sgRNA `sequence_similar_to` edges based on guide
  sequence similarity.
- Does not change target semantics.

Sprint 7B must distinguish two Graph B identities:

- Historical Sprint 4 Graph B: `S1_pair+F1`; context-only reference.
- New Sprint 7B Graph B: same Graph B topology, but candidate-pair edge feature
  table `S5F2_energy` with 268 columns.

Do not compare historical Sprint 4 Graph B `S1_pair+F1` directly against Sprint
7B Graph B GATv2 `S5F2_energy` as an architecture result.

### Graph C

`graph_c_context_observation` is a context-observation graph:

- Node types: `sgRNA`, `target_observation`.
- One `target_observation` node exists per candidate observation/source row.
- `target_observation` nodes carry context features.
- Candidate-pair edges connect sgRNA nodes to target observations.
- `context_similar_to` edges connect target observations using label-free
  context similarity topology.

Graph C changes both topology and target semantics. It must not be described as
topology-only.

## 5. Literature And API Basis

Repository-local literature notes:

- Kipf & Welling, 2017, "Semi-Supervised Classification with Graph Convolutional
  Networks": supports the Sprint 4-7 GCN baseline lineage through normalized
  neighborhood aggregation.
- Vinodkumar, Ozcinar & Anbarjafari, 2021, "Prediction of sgRNA Off-Target
  Activity in CRISPR/Cas9 Gene Editing Using Graph Convolution Network":
  motivates graph/link-prediction framing for CRISPR off-target modeling. This
  project does not reproduce its dataset, split, target, architecture, or
  metrics.
- Jiang, Li, Xiong & Liu, 2025, "Graph-CRISPR: a gene editing efficiency
  prediction model based on graph neural network with integrated sequence and
  secondary structure feature extraction": motivates graph/attention-style
  CRISPR modeling and attention diagnostics, but it is an on-target efficiency
  setting, not this off-target Scheme A guide-level measured-only task.

Canonical GAT/GATv2 references:

- Velickovic et al., 2018, "Graph Attention Networks": motivates learned
  attention-weighted neighbor aggregation.
- Brody, Alon & Yahav, 2021/2022, "How Attentive are Graph Attention Networks?":
  motivates GATv2 as a dynamic-attention variant of GAT.

No dedicated local paper note for Velickovic et al. or Brody et al. is present
under `docs/literature/`; Sprint 7B should cite them as canonical external
references, not as repo-local notes.

PyTorch Geometric API basis inherited from Sprint 7:

- `GATv2Conv` supports `edge_dim` and `edge_attr`.
- When `edge_dim` is set, self-loop edge features must not use PyG's default
  `fill_value="mean"`.
- Sprint 7B core must use zero-filled self-loop edge attributes.

## 6. Core Run Matrix

Carry-forward context rows:

| ID | Graph | Architecture | Feature setting | Role |
| --- | --- | --- | --- | --- |
| `S7B_REF_XGB_F4` | tabular | XGBoost | F4 | Authoritative external AUPRC bar |
| `S7B_REF_GA_GCN` | Graph A | GCN | `S5F2_energy` | Best current GNN reference from Sprint 6 |
| `S7B_REF_GA_GATV2` | Graph A | GATv2 | `S5F2_energy` | Failed Graph A attention reference from Sprint 7 |
| `S7B_REF_GC_GCN` | Graph C | GCN | `GraphCContext+S5F2_energy` | Existing Sprint 5B Graph C energy reference |

New canonical training runs:

| ID | Graph | Architecture | Feature setting | Role |
| --- | --- | --- | --- | --- |
| `S7B_R1_graph_b_gcn_s5f2` | Graph B | GCN | `S5F2_energy` | Matched Graph B feature/reference run |
| `S7B_R2_graph_b_gatv2_s5f2` | Graph B | GATv2 | `S5F2_energy` | Tests guide-similarity topology x GATv2 |
| `S7B_R3_graph_c_gatv2_s5f2` | Graph C | GATv2 | `GraphCContext+S5F2_energy` | Tests context-observation topology/semantics x GATv2 |

Do not include GAT in the Sprint 7B core. Sprint 7 already showed GAT was much
weaker than GATv2 on Graph A. Sprint 7B's core question is whether the stronger
attention candidate interacts usefully with richer graph structure.

Graph C GCN rerun policy:

- Prefer carry-forward Sprint 5B Graph C GCN reference.
- Rerun Graph C GCN only if code/provenance compatibility requires it.
- If rerun, predeclare it as `S7B_R0_graph_c_gcn_s5f2_rerun` and report both
  the carried-forward and rerun rows; do not use the rerun to tune GATv2.

## 7. Graph B `S5F2_energy` Artifact Policy

Sprint 7B must create or attach a new Graph B candidate-edge feature artifact:

```text
data/processed/graphs/sprint7b/graph_b_guide_similarity_control/
```

Required provenance:

- `graph_schema = graph_b_guide_similarity_control`.
- Candidate edge feature set: `S5F2_energy`.
- Candidate edge feature columns: `268`.
- `sequence_similar_to` topology unchanged from validated Graph B.
- Split: `sprint2_main_seed42`.
- Label scheme: `scheme_a`.
- Visibility policy: `strict_inductive_primary`.
- Physical target node policy: featureless `zero_type_feature`.
- Strict-inductive auxiliary-edge visibility preserved.

This artifact must not overwrite or be confused with:

- Sprint 4 Graph B `S1_pair+F1` artifacts.
- Sprint 5 Graph A `S5F2_energy` artifacts.

Implementation implication:

- `graph_b_edge_feature_attrs()` currently supports only `s1_pair` and
  `f1`-`f4`. Sprint 7B must add `s5f2_energy` support for Graph B only after the
  Sprint 7B plan is frozen.
- Graph B GCN `S5F2_energy` is required before Graph B GATv2 can be interpreted
  as a matched architecture comparison.

## 8. GATv2 Edge And Relation Policy

Core architecture policy:

- Use relation-aware collapsed/homogeneous GATv2, preserving the existing GCN
  graph materialization philosophy.
- Do not use HeteroConv in the core.
- Do not add edge-type embeddings in the core unless the plan is amended before
  any canonical run.
- Report parameter counts for every trained run.

Graph B edge policy:

- Candidate-pair forward edges: `S5F2_energy` edge_attr.
- Candidate-pair reverse edges: duplicate the same `S5F2_energy` row.
- `sequence_similar_to` edges: topology only in core; zero edge_attr vector of
  the same dimension as candidate edges.
- Reverse `sequence_similar_to` edges: preserve the existing bidirectional
  homogeneous construction policy; zero edge_attr.
- Self-loops: present; zero edge_attr; no PyG mean fill.

Graph C edge policy:

- Candidate-pair forward edges: `S5F2_energy` edge_attr.
- Candidate-pair reverse edges: duplicate `S5F2_energy`.
- `context_similar_to` edges: topology only in core; zero edge_attr vector.
- Reverse `context_similar_to` edges: zero edge_attr.
- Self-loops: present; zero edge_attr.
- `target_observation` node context features remain the established Graph C
  node features from Sprint 4/5B.
- Final edge classifier still receives source embedding, target embedding,
  pair interaction terms, and candidate-pair `S5F2_energy` for parity.

HeteroConv / typed GATv2 stretch:

- HeteroConv is not Sprint 7B core.
- If added later, it must be a separate stretch with explicit interpretation
  caveat: typed relation handling adds relation-specific parameters/capacity and
  is not directly equivalent to the matched GCN reference.

## 9. Legitimate Comparisons

Clean matched-schema comparisons:

- Graph B GCN `S5F2_energy` vs Graph B GATv2 `S5F2_energy`.
- Graph C GCN `GraphCContext+S5F2_energy` vs Graph C GATv2
  `GraphCContext+S5F2_energy`.
- XGBoost F4 vs all graph models as the required external bar.

Context-only comparisons:

- Graph A GCN vs Graph B GATv2: architecture + topology, not architecture only.
- Graph A GCN vs Graph C GATv2: architecture + topology + target semantics, not
  architecture only.
- Graph B GATv2 vs Graph C GATv2: different graph semantics; report cautiously.

Invalid comparisons:

- Historical Sprint 4 Graph B `S1_pair+F1` GCN vs Sprint 7B Graph B
  `S5F2_energy` GATv2 as an architecture comparison.
- Graph C improvement over Graph A as "topology improvement".
- Choosing Graph B or Graph C variants based on test AUPRC and rerunning
  adapted variants.

## 10. Output Contract

Canonical Sprint 7B outputs:

```text
outputs/sprint7b/gatv2_topology_comparison.csv
outputs/sprint7b/gatv2_topology_report.md
outputs/sprint7b/gatv2_topology_run_manifest.json
outputs/sprint7b/graph_artifact_provenance.json
outputs/sprint7b/diagnostics/
outputs/sprint7b/figures/
outputs/sprint7b/runs/<run_id>/{resolved_config.yaml,runtime.json,training_history.csv,metrics.csv,attention_summary.csv,model.pt}
```

Graph B artifact report:

```text
outputs/sprint7b/graph_b_s5f2_artifact_report.md
```

Required diagnostic tables:

- Consolidated result table including carry-forward references and new runs.
- Per-run validation/test predictions.
- Training history.
- Fixed-threshold metrics using validation-selected thresholds.
- Score deciles / lift diagnostics.
- Per-guide metrics.
- Per-genome metrics.
- Graph artifact provenance with node/relation counts and feature dimensions.
- Edge-attribute alignment checks:
  - candidate forward counts,
  - candidate reverse counts,
  - similarity/context edge counts,
  - self-loop counts,
  - zero-fill confirmation,
  - candidate-edge `S5F2_energy` dimension.
- Attention summaries by:
  - graph schema,
  - architecture,
  - layer,
  - head,
  - split,
  - candidate_forward,
  - candidate_reverse,
  - sequence_similar_to,
  - context_similar_to,
  - self_loop.

Required figures:

```text
outputs/sprint7b/figures/gatv2_topology_auprc_comparison.png
outputs/sprint7b/figures/gatv2_topology_pr_curves.png
outputs/sprint7b/figures/gatv2_topology_roc_curves.png
outputs/sprint7b/figures/gatv2_topology_training_curves.png
outputs/sprint7b/figures/gatv2_topology_threshold_metrics.png
outputs/sprint7b/figures/gatv2_topology_score_distributions.png
outputs/sprint7b/figures/gatv2_topology_per_guide_metric_distribution.png
outputs/sprint7b/figures/gatv2_topology_attention_by_edge_kind.png
```

Figures must include positive prevalence context and the XGBoost F4 reference
where appropriate. Attention figures must separate edge kinds; do not mix
candidate-pair attention with guide-similarity or context-similarity attention.

## 11. Colab Runner Contract

Colab remains a runner only.

Expected notebook:

```text
colab/sprint7b_gatv2_topology_runner.ipynb
```

The notebook must:

1. Mount Drive.
2. Clone or update the approved branch.
3. `uv sync`.
4. Copy required graph artifacts from Drive:
   - Sprint 7B Graph B `S5F2_energy` artifacts.
   - Sprint 5B Graph C `S5F2_energy` artifacts.
   - Carry-forward output references if needed.
5. Validate graph artifact provenance before training.
6. Run the repository Sprint 7B runner command.
7. Copy `outputs/sprint7b/` back to durable Drive storage.
8. Refuse to overwrite an existing Drive returned-output folder.

The notebook must not implement model classes, losses, samplers, graph
materialization logic, evaluation, plotting, or attention-summary logic.

Expected command shape:

```bash
uv run python scripts/run_sprint7b_gatv2_topology.py \
  --config configs/sweeps/sprint7b_gatv2_topology.yaml \
  --run-id sprint7b_gatv2_topology_seed42_<timestamp>
```

## 12. Tests And Contract Guards

Graph artifact tests:

- Graph B `S5F2_energy` artifacts exist under `data/processed/graphs/sprint7b/`.
- Graph B topology counts match validated Graph B topology where expected:
  `sgRNA=150`, `physical_target_site=9880`, `candidate_pair=11446`,
  `sequence_similar_to=1208`.
- Graph B candidate-edge feature table has `268` columns.
- Graph B metadata distinguishes Sprint 7B `S5F2_energy` from Sprint 4
  `S1_pair+F1`.
- Graph C artifact provenance matches Sprint 5B `GraphCContext+S5F2_energy`.

Model/edge tests:

- Graph B GATv2 receives candidate-edge `S5F2_energy` via `edge_attr`.
- Graph B sequence-similarity edges receive zero edge_attr in core.
- Graph C GATv2 receives candidate-edge `S5F2_energy` via `edge_attr`.
- Graph C context-similarity edges receive zero edge_attr in core.
- Candidate reverse edges duplicate candidate-edge `S5F2_energy`.
- Self-loop edge attributes are zero-filled.
- PyG `fill_value="mean"` is not used.
- Final classifier still receives candidate-edge `S5F2_energy`.
- Physical target nodes in Graph B remain featureless.
- Graph C uses `target_observation` node context encoder; no
  `physical_target_site` nodes.

Trainer/config tests:

- Sprint 7B configs reject non-frozen label/split/evaluation settings.
- Sprint 7B configs reject losses other than weighted BCE.
- Sprint 7B configs reject Graph B GATv2 without matched Graph B GCN
  `S5F2_energy`.
- Sprint 7B configs reject Graph C topology-only wording/metadata.
- Optional/stretch typed HeteroConv cannot execute by default.
- Checkpoint selection remains validation AUPRC.
- Threshold selection remains validation max-F1.

Output tests:

- Required CSV/report/manifest/provenance/diagnostic/figure artifacts are
  produced.
- Result table includes XGBoost F4, Graph A GCN, Graph A GATv2, Graph B GCN,
  Graph B GATv2, Graph C GCN, and Graph C GATv2 rows.
- Attention summaries separate edge kinds.
- `model.pt` checkpoints are written as run artifacts but remain untracked.

Run before any canonical Colab run:

```bash
uv run pytest -q tests/test_sprint7b_gatv2_model.py tests/test_sprint7b_gatv2_runner.py tests/test_gcn_training_smoke.py tests/test_config_loads.py
uv run ruff check scripts src tests
```

CPU smoke/debug runs must not be reported as Sprint 7B evidence.

## 13. Interpretation Rules

Use these rules in the Sprint 7B report:

- AUPRC is the primary metric.
- MCC, specificity/TNR, macro F1, and TN/FP/FN/TP are secondary diagnostics.
- A GATv2 run is promising only if it improves matched-schema AUPRC:
  - Graph B GATv2 > Graph B GCN `S5F2_energy`, and/or
  - Graph C GATv2 > Graph C GCN `GraphCContext+S5F2_energy`.
- A practical exploratory gain is `>= 0.005` AUPRC over matched GCN.
- A strong exploratory gain is `>= 0.010` AUPRC over matched GCN.
- Even strong single-seed gains justify Sprint 8 robustness; they do not prove
  final superiority.
- If GATv2 improves only MCC/specificity while lowering AUPRC, do not describe
  it as a headline architecture win.
- If Graph B and Graph C GATv2 both fail to beat matched GCN references on
  AUPRC, stop expanding attention and move to robustness or another modeling
  direction.
- Attention summaries are interpretation-only and must not be claimed as
  biological causal evidence.
- Do not claim reproduction of Kipf & Welling, Velickovic et al., Brody et al.,
  Vinodkumar et al., Jiang et al., or Mak et al.; dataset, target, split,
  metrics, and architecture differ.

## 14. Risks

1. **Attribution ambiguity:** Graph B requires a matched GCN `S5F2_energy`
   baseline; otherwise feature and architecture changes are confounded.
2. **Graph C semantics confound:** Graph C changes target semantics and
   topology. It cannot support topology-only claims.
3. **Relation-type capacity confound:** HeteroConv, edge-type embeddings, or
   relation-specific layers add capacity and are out of core scope.
4. **Edge-attribute confound:** adding similarity-distance or context-distance
   edge features creates a new feature sprint.
5. **Self-loop confound:** PyG mean-filled self-loop edge_attr would synthesize
   aggregate candidate/context features.
6. **Single-seed fragility:** Sprint 7B is exploratory; promising results need
   Sprint 8 robustness.
7. **High-prevalence metric behavior:** AUPRC has a high floor and threshold
   metrics swing over only 169 test negatives.
8. **Attention overclaim:** attention weights are model diagnostics only.
9. **XGBoost gap:** even improved GATv2 may remain below the F4 tabular
   baseline.
10. **Scope creep:** adding sequence encoders, losses, samplers, or measured-zero
    rows would make the topology-attention conclusion uninterpretable.

## 15. Acceptance Criteria

Sprint 7B planning acceptance:

- This plan records the Graph B `S5F2_energy` artifact identity and keeps it
  separate from historical Sprint 4 Graph B.
- The run matrix is frozen before implementation.
- The homogeneous/relation-aware GATv2 core choice is frozen before
  implementation.
- HeteroConv is explicitly out of core scope.
- The interpretation rules and no-test-tuning boundary are recorded before
  canonical training.

Sprint 7B implementation acceptance:

- Graph B `S5F2_energy` artifacts are created or attached with provenance.
- Graph B matched GCN and GATv2 runs complete under the frozen contract.
- Graph C GATv2 completes under the frozen contract.
- Carry-forward references are included without being retrained unless
  predeclared.
- Required reports, diagnostics, figures, manifests, and provenance files exist.
- Report preserves AUPRC-first interpretation and graph-specific comparison
  boundaries.

## 16. Implementation Slices

### Slice 0 - Planning freeze

Review and freeze this plan. Record final decisions for run matrix, Graph B
artifact identity, Graph C reference policy, homogeneous GATv2 policy,
self-loop/edge-attr policy, and output contract.

Exit: plan frozen; no code changed yet.

### Slice 1 - Graph B `S5F2_energy` artifacts

Add repository code/config to create or attach Graph B `S5F2_energy` candidate
edge features without changing Graph B topology. Write provenance and artifact
tests.

Exit: `data/processed/graphs/sprint7b/graph_b_guide_similarity_control/`
contains validated Graph B `S5F2_energy` artifacts; no training yet.

### Slice 2 - GATv2 model support for Graph B and Graph C

Add relation-aware collapsed GATv2 support for Graph B and Graph C. Keep
similarity/context edges topology-only with zero edge_attr in core. Add edge
alignment, self-loop, target-representation, and attention-return tests.

Exit: model tests pass; no canonical training.

### Slice 3 - Trainer/config dispatch

Wire Sprint 7B GATv2 configs into the existing trainer or a small
architecture-neutral extension while preserving weighted BCE, validation
checkpointing, validation thresholding, and GCN behavior.

Exit: tiny CPU smoke can train Graph B GCN/GATv2 and Graph C GATv2; GCN
regression tests still pass.

### Slice 4 - Runner, reporting, diagnostics, figures

Add `configs/sweeps/sprint7b_gatv2_topology.yaml`,
`scripts/run_sprint7b_gatv2_topology.py`, Sprint 7B diagnostics, figures,
manifest, provenance, and output-contract tests.

Exit: mocked/smoke outputs satisfy the reporting contract; no headline claim.

### Slice 5 - Colab runner preparation

Add a runner-only Colab notebook and documented command path. Validate Drive
artifact copy paths and returned-output checks.

Exit: notebook contract tests pass; no full GPU claim yet.

### Slice 6 - Full headline run and local validation

Run predeclared Sprint 7B on Colab GPU, copy outputs back, validate locally, and
do not rerun or tune from test diagnostics.

Exit: all required Sprint 7B outputs exist or any technical omission is
documented before interpreting results.

### Slice 7 - Sprint closure

Freeze report/results/status docs, record the decision in `docs/DECISIONS.md`,
and move this plan to `docs/exec-plans/completed/`.

Exit: Sprint 7B conclusion is documented as one of:

- Graph B GATv2 improves matched Graph B GCN and justifies Sprint 8 robustness.
- Graph C GATv2 improves matched Graph C GCN and justifies Sprint 8 robustness.
- GATv2 improves only secondary threshold metrics; continue only with caution.
- GATv2 does not improve matched Graph B or Graph C GCN; stop expanding
  attention and move to robustness or another modeling direction.
