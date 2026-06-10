# Execution Plan: Sprint 8A Target-Context & Context-Edge Interaction

> Status: FROZEN — Slice 0 planning freeze complete (2026-06-10). All predeclared
> defaults are pinned in §17 Frozen Specification; Slices 1–8 may proceed with no
> remaining design decision. This plan covers Sprint 8A, the model-improvement
> core of Sprint 8. Sprint 8B
> (sequence-context encoder) is a separate companion plan,
> `008b-sprint8b-sequence-context-encoder.md`. Robustness (multi-seed,
> guide-level bootstrap CIs, paired-difference) is re-scoped to Sprint 9.

## 1. Goal

Sprint 7F showed that a **family-aware target-context encoder** is the strongest
Graph C GATv2 representation found so far, beating both a larger unified-deep
encoder and the non-family references on validation and test AUPRC while
improving rare-negative threshold behaviour. Sprint 7D/7E localized the signal:
direct `target_observation` features (especially the six experimental epigenetic
features) drive the rare-negative operating point, not `context_similar_to`
topology.

Sprint 8A asks the next, mechanism-driven question:

> Can the best Sprint 7F family-aware Graph C GATv2 model be improved by
> (a) learning how to fuse the target-context families instead of fixed
> concatenation, (b) explicitly interacting the target-context embedding with the
> candidate `S5F2_energy` edge features before classification, and (c) regularizing
> the strong-but-brittle experimental epigenetic branch — while keeping the frozen
> evaluation contract and the frozen GATv2 attention/message-passing path
> unchanged?

### AGENTS.md workflow summary

- **Task goal:** add 3 predeclared, mechanism-driven architecture deltas on top of
  the Sprint 7F R3 base, run exactly 5 canonical Graph C GATv2 runs, and select by
  validation AUPRC only.
- **Expected file changes (implementation phase, NOT this plan):**
  `src/crispr_gnn/models/target_context_encoder.py` (family gate v2, regularized
  experimental branch), `src/crispr_gnn/models/gat.py` (context-edge interaction
  head in `GraphCEdgeGATv2`), `src/crispr_gnn/training/gcn.py` (config dispatch
  for the new options), `configs/sweeps/sprint8a_target_context_interaction.yaml`,
  `scripts/run_sprint8a_target_context_interaction.py`,
  `colab/sprint8a_target_context_interaction_runner.ipynb`,
  `tests/test_sprint8a_target_context_interaction.py`.
- **Risks:** capacity confound (a gain might come from extra parameters, not
  structure); validation overfitting from too many knobs; FiLM/MLP head adding
  parameters; single-seed variance; accidental change to the frozen GATv2 message
  passing.
- **Acceptance criteria:** exactly 5 canonical trained runs, frozen contract
  preserved, only the predeclared architecture deltas vary, audits prove
  message-passing is unchanged, output contract complete, local tests + lint pass,
  Colab is runner-only. (Full list in §13.)

Sprint 8A will **not**: add a sequence encoder (that is Sprint 8B), change the
split/labels/loss/threshold/checkpoint policy, change GATv2 attention/message
passing, restore `context_similar_to` edges, run open-ended hyperparameter
search, or select any run/architecture/threshold from test metrics.

## 2. Frozen Evaluation Contract (inherited verbatim from Sprint 7F)

- Label scheme: `scheme_a` = `int(cleavage_freq > 1e-5)`; NaN `cleavage_freq`
  excluded from supervised labels.
- Split ID: `sprint2_main_seed42`; guide-disjoint.
- Headline universe: measured-only (train/validation/test); `experiment_id=18`
  excluded.
- Graph visibility: `strict_inductive_primary`; train-only preprocessing.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- No test-driven selection of architecture, threshold, feature, or hyperparameter.
- Primary metric: **AUPRC**. Secondary (threshold) metrics: AUROC, F1, macro F1,
  MCC, specificity/TNR, sensitivity, TN/FP/FN/TP.
- Test positive prevalence: `0.900705` (negatives are the rare class).
- Authoritative AUPRC bar: `xgboost_unweighted / F4`, test AUPRC `0.992522`,
  AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

Canonical base setting (frozen):

- Graph schema `graph_c_context_observation`; architecture family Graph C GATv2.
- Candidate edge features `S5F2_energy` (268 columns); active in GATv2 attention
  (`edge_aware_attention=true`) and in the final edge classifier.
- Target node features `target_observation_features` (212 columns), family split
  `115 / 6 / 78 / 13` (`target_sequence_one_hot` / `experimental_epigenetic` /
  `computed_nucleosome_aggregates` / `computed_nucleosome_missingness`).
- Loss `weighted_bce`, `pos_weight: auto` (negatives/positives ≈ 0.1267).
- GATv2: 2 layers, hidden 128, 4 heads, concat true, dropout 0.2, attention
  dropout 0.2, `share_weights=false`, self-loop edge fill 0.0.
- `context_similar_to` edges DROPPED for every Sprint 8A run.
- Optimizer AdamW, `ReduceLROnPlateau(mode="max")` on `val_auprc`, grad clip 1.0,
  LR 1e-3, weight decay 1e-4; seed 42.
- Colab is runner-only; final logic lives in `src/`, `configs/`, `scripts/`.

MCC/specificity/TN movement is reported as rare-negative operating-point evidence
but must never override AUPRC ranking.

## 3. Prior Evidence Entering Sprint 8A

Carry-forward reference rows (no retrain; exact numbers from
`outputs/sprint7f/target_context_encoder_comparison.csv`):

| Run ID | Setting | Test AUPRC | Test AUROC | Test Macro F1 | Test MCC | Specificity | TN/FP/FN/TP | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `S8A_REF_XGB_F4` | XGBoost F4 | `0.992522` | `0.938416` | `0.6427` | `0.345198` | n/a | `38/131/21/1512` | — |
| `S8A_REF_GRAPH_A_GCN` | Graph A GCN + S5F2 | `0.976935` | `0.819972` | `0.698939` | `0.483719` | `0.289941` | `49/120/6/1527` | — |
| `S8A_REF_GRAPH_C_GCN` | Graph C GCN + S5F2 | `0.972481` | `0.836219` | `0.552442` | `0.274287` | `0.082840` | `14/155/0/1533` | — |
| `S8A_REF_FULL_GRAPH_C_GATV2` | Full Graph C GATv2 | `0.969078` | `0.849705` | `0.739526` | `0.531774` | `0.372781` | `63/106/12/1521` | — |
| `S8A_REF_NO_CTX_EDGE_GATV2` | Graph C GATv2 no-context-edge (unified shallow) | `0.965598` | `0.850137` | `0.733910` | `0.517970` | `0.366864` | `62/107/14/1519` | — |
| `S8A_REF_S7F_R2` | S7F R2 balanced family-aware | `0.982062` | `0.906557` | `0.801716` | `0.603489` | `0.650888` | `110/59/63/1470` | `275601` |
| `S8A_REF_S7F_R3` | S7F R3 experimental-emphasis (**base**) | `0.984945` | `0.926551` | `0.777185` | `0.568108` | `0.497041` | `84/85/31/1502` | `274153` |

Selection-relevant validation AUPRC (the legitimate selection metric): S7F R1
unified-deep `0.976594`, S7F R2 `0.977541`, **S7F R3 `0.987522` (highest)**.

Sprint 7D/7E mechanism evidence (carried forward, not re-run): dropping
`context_similar_to` edges preserved the gain; edge-blind attention hurt; masking
direct `target_observation` (or the experimental epigenetic subgroup) collapsed
negative recognition.

## 4. Literature Framing

Use these only to justify the design axis and to set claim boundaries; do **not**
claim reproduction. All are in `docs/literature/axes/axis_4_model_architecture_components/`.

- **SENet** (Hu et al.; 4A): squeeze→excitation channel recalibration via a small
  FC bottleneck + sigmoid gate, "computationally lightweight". Justifies a cheap
  **learned group gate** over the four target-context family branches. Caveat:
  vision/channel setting; cite as the gating precedent, not genomics evidence.
- **FiBiNET** (Huang et al.; 4B): applies SENET to **field (feature-group)
  embeddings**, then a bilinear interaction — i.e. "reweight, then interact".
  Direct architectural analogue of S8A R1→R3. Caveat: large-scale CTR; borrow the
  pattern, not the scale.
- **FiLM** (Perez et al.; 4B): feature-wise affine modulation `γ·F + β` from a
  conditioning input, "two parameters per modulated feature map", generalizes
  from little data. Primary justification for the **context→S5F2 interaction
  head**. Caveat: visual-reasoning origin; mechanistic, not domain-specific.
- **GNN-FiLM** (Brockschmidt; 4B): conditions messages on the target-node
  representation; also reports that a plain **GNN-MLP on `[source||target]`** is a
  strong baseline. Justifies our interaction principle AND our predeclared
  interaction-MLP fallback. Caveat: GNN-FiLM modulates **inside** message passing;
  our adaptation is **head-only** to preserve the frozen GATv2 path — state this.
- **Lengerich** (Dropout as a regularizer of interaction effects; 4C): input
  dropout shrinks order-`k` interactions by `(1-p)^k`; "higher dropout rates for
  stronger regularization against spurious high-order interactions". Justifies
  **branch/feature-dropout** on the experimental branch. Caveat: input-dropout
  networks are biased estimators — interpretation-only.
- **Geirhos** (Shortcut Learning; 4C): models exploit easy proxies (e.g. a
  hospital token instead of pneumonia). Justifies **why** the 6 experimental
  epigenetic features may be a brittle source/cell-line/batch shortcut and should
  be regularizable. Caveat: general perspective, not a CRISPR result.
- **Overtuning** (Schneider, Bischl, Feurer; 4D): excessive validation
  optimization overfits at the HPO level (~10% of cases worse than default),
  "particularly in the small-data regime". Bounds the §6 axis-4 HP refinement.
- **Kapoor & Narayanan** (Leakage; 4D): test-set reuse / leakage causes
  overoptimism (294 papers). Backs the no-test-tuning contract and selection
  discipline.
- **Dwivedi** (Benchmarking GNNs; 4E): "fair model comparison with the same
  parameter budget". Backs the §7 capacity-confound control.
- **GATv2 / Brody** (4E): GAT computes only static attention; GATv2 is dynamic and
  "matches their parametric cost". Canonical backbone citation; also supports the
  structure-vs-capacity argument.

## 5. Primary Design Decision

Canonical base = **S7F R3 `family_aware_experimental_emphasis`** (branch dims
`24 / 48 / 40 / 16`), justified by the highest **validation** AUPRC (`0.987522`)
— so the base choice never touches the test set. S7F R2 is kept ONLY as a
carry-forward reference row for rare-negative comparison; it is not the base and
we did not switch to it.

All Sprint 8A deltas are applied to the target-context encoder and the classifier
head. The **frozen GATv2 attention/message passing must not change**: candidate
`S5F2_energy` still enters GATv2 attention exactly as in Sprint 7F, and
`context_similar_to` edges remain dropped. Edge-conditioning inside message
passing (GNN-FiLM/ECC style) is explicitly out of scope; our interaction is
head-only.

Canonical model family pipeline (additions in **bold**):

```text
target_observation (212)
  → family branches (115/6/78/13 → 24/48/40/16)      [base R3]
  → [R1/R3] learned SENET-style group gate over branches
  → concat (128) → fusion MLP → hidden_dim (128)
  → GATv2 message passing + S5F2 edge-aware attention  [FROZEN]
        ↓ post-GATv2 node embeddings
candidate edge classifier head:
  source, target(=context embedding), source*target, |source-target|,
  + [R2/R3] FiLM-modulated S5F2 edge embedding  (else raw S5F2 as in 7F)
  → edge classifier → logit
```

## 6. Architecture Deltas (exact, per run)

All new options are config-driven and default to OFF (so the base R3 config
reproduces Sprint 7F exactly). Exact dims below are **predeclared defaults**;
only the §6.4 axis-4 step may refine them, validation-only.

### 6.1 Learned family gate v2 (R1, R3) — `target_context_encoder.py`

Add a SENET-style group gate inside `FamilyAwareTargetContextEncoder`, applied to
the four branch outputs before concatenation:

- Squeeze: take each branch output `b_f ∈ R^{d_f}` (already per-node); form the
  concatenated branch vector `b = [b_1..b_4] ∈ R^{128}`.
- Excitation: `s = σ(W2 · ReLU(W1 · b))`, with `W1 ∈ R^{(128/r)×128}`,
  `W2 ∈ R^{4×(128/r)}`, reduction ratio `r = 4`, producing **one scalar gate per
  family** (4 gates).
- Re-weight: scale each branch by its gate, `b_f ← s_f · b_f`, then concat (128) →
  existing fusion block (`LayerNorm → ReLU → Dropout(0.2) → Linear(128→128) →
  ReLU`).
- New: `FamilyAwareTargetContextEncoder(..., family_gate: bool = False,
  gate_reduction: int = 4)`. New encoder type alias
  `family_aware_experimental_emphasis_gated` (or a `family_gate` flag on the
  existing type) selected via config `model.target_context_encoder.family_gate`.
- Parameter cost is tiny (≈ `128·32 + 32·4` ≈ 4.2k), supporting the
  structure-not-capacity argument; report it in the audit.

### 6.2 Context-edge interaction head (R2, R3) — `gat.py`

Add a **new, Graph-C-specific interaction head path** inside `GraphCEdgeGATv2`,
guarded by the `context_edge_interaction` flag. The shared module-level helpers
`_classify_candidate_edges` and `_edge_classifier` (also used by Graph B) MUST stay
unchanged and continue to serve the interaction-OFF / default path. Do not touch
`_apply_attention_layers` or `graph_c_attention_edge_tensors`.

- Embed the candidate edge features: `edge_embed = Linear(268 → d_e)(S5F2)`,
  `d_e = 64` (predeclared).
- Context embedding for each candidate edge = the post-GATv2 target-observation
  node embedding `target = x[target_index] ∈ R^{128}` (already computed).
- **FiLM (primary):** FiLM generator `g(target) → (γ, β)`, each `∈ R^{d_e}`, via a
  shared `Linear(128 → 2·d_e)`; modulate `edge_film = γ ⊙ edge_embed + β`.
- Classifier input becomes
  `[source, target, source*target, |source-target|, edge_film]`
  (dim `128·4 + d_e`), replacing the raw 268-col `candidate_edge_attr`
  concatenation. The 268-col S5F2 still feeds GATv2 attention unchanged.
- **Interaction-MLP (predeclared fallback):** if FiLM underperforms on validation,
  swap the head for an MLP over `[edge_embed, target, edge_embed * proj(target)]`
  (with `proj: Linear(128→d_e)`), `→ Linear → ReLU → d_e`. Genuinely competitive
  per GNN-FiLM's GNN-MLP result. Full bilinear is OUT (O(d²) params; overfit risk
  with ~900 train negatives, single seed).
- New: `GraphCEdgeGATv2(..., context_edge_interaction: str = "none",
  interaction_edge_dim: int = 64)`, values `none | film | mlp`, set via config
  `model.context_edge_interaction`.

### 6.3 Regularized experimental-epigenetic branch (R4) — `target_context_encoder.py`

Applied to the 6-feature `experimental_epigenetic` branch only:

- Feature-dropout: drop whole input features with prob `p_feat = 0.3` during
  training (Bernoulli mask over the 6 columns), per Lengerich's input-dropout
  shrinkage of spurious interactions.
- Bottleneck: `Linear(6 → bottleneck_dim=4) → LayerNorm → ReLU →
  Linear(4 → 48) → LayerNorm → ReLU` (replaces the direct `6 → 48` branch),
  forcing the brittle branch through a narrow channel.
- New: `experimental_branch_bottleneck: int | None = None`,
  `experimental_branch_feature_dropout: float = 0.0` on the encoder, set via
  config `model.target_context_encoder.experimental_branch.*`.

### 6.4 Module ownership

- `target_context_encoder.py`: family gate (6.1), regularized experimental branch
  (6.3). No change to the base family/branch dims when flags are OFF.
- `gat.py`: context-edge interaction head (6.2) as a new Graph-C-specific path
  inside `GraphCEdgeGATv2`, plus new audit hooks (γ/β summaries, interaction
  nonzero). The shared `_classify_candidate_edges` / `_edge_classifier` helpers,
  `_apply_attention_layers`, `graph_c_attention_edge_tensors`, and the GATv2 conv
  stack are untouched (so Graph A/B and the interaction-OFF path are unaffected).
- `training/gcn.py`: extend `GCNRunConfig` + `gcn_run_config_from_mapping` +
  `_build_model` to pass the new flags through. No change to the training loop,
  loss, scheduler, checkpoint, or threshold logic.

## 7. Predeclared Run Matrix

Exactly **5 canonical trained runs** (all Graph C GATv2, frozen contract, R3
base, seed 42). Do not add canonical runs after seeing test results.

| Run ID | Encoder delta | Head delta | Exp-branch reg | Axis | Lit |
| --- | --- | --- | --- | --- | --- |
| `S8A_R0_base_reference` | none (R3 exact) | none (raw S5F2) | none | — (anchor) | reproduces S7F R3 in S8 harness |
| `S8A_R1_family_gated_v2` | family gate v2 | none | none | fusion/gating | SENet, FiBiNET |
| `S8A_R2_context_edge_film` | none (R3) | FiLM head | none | interaction | FiLM, GNN-FiLM |
| `S8A_R3_gated_plus_film` | family gate v2 | FiLM head | none | both | FiBiNET |
| `S8A_R4_regularized_exp_branch` | none (R3) | none | bottleneck + feat-dropout | regularization | Lengerich, Geirhos |

Controlled variable per run is exactly one (or, for R3, the predeclared
combination of R1+R2). Carry-forward reference rows (no retrain): the seven rows
in §3.

Optional / approval-gated only (NOT canonical, do not run without a plan
amendment before test inspection):

| Run ID | Setting | Reason to defer |
| --- | --- | --- |
| `S8A_OPT_mlp_head` | R2/R3 head = interaction-MLP instead of FiLM | Only if FiLM underperforms on validation; predeclared fallback, reported separately. |
| `S8A_OPT_hp_refine` | §6.4 axis-4 HP refinement on the single validation winner | Bounded, last, validation-only (see below). |

### Axis-4 HP refinement (bounded, last)

Applied **only at the very end**, **only to the single validation-AUPRC-winning
architecture**, with knobs and a small grid predeclared **before** running:

- target-context branch dims (±1 predeclared alt allocation),
- fusion dropout `{0.2, 0.3}`,
- GATv2 heads `{4, 8}`,
- hidden dim `{128, 192}`,
- learning rate `{1e-3, 5e-4}`,
- weight decay `{1e-4, 1e-3}`.

Use a small predeclared random sample of ≤ 8 configurations (not full grid).
Selection validation-only. Cite Overtuning (2025) and Kapoor (2023) for the
small-data overtuning/leakage risk. Report the refinement as a separate,
clearly-labelled block; it must not redefine the 5 canonical rows.

This optional step is the **sole sanctioned, predeclared exception** to the
otherwise-frozen GATv2 head count, hidden dim, learning rate, and weight decay of
§2: those inherited defaults stay fixed for all 5 canonical runs (R0–R4) and may
move only here, only on the single validation-AUPRC winner, and only within the
predeclared grid above.

## 8. Implementation Scope (planned; not written in this task)

- Model: `src/crispr_gnn/models/target_context_encoder.py` (family gate,
  regularized experimental branch); `src/crispr_gnn/models/gat.py`
  (`GraphCEdgeGATv2` interaction head + audit hooks).
- Training dispatch: `src/crispr_gnn/training/gcn.py` (`GCNRunConfig` new fields,
  `gcn_run_config_from_mapping`, `_build_model`). No training-loop changes.
- Config: `configs/sweeps/sprint8a_target_context_interaction.yaml`, mirroring
  `sprint7f_target_context_encoder.yaml`, with the 5 runs and new keys:
  `model.target_context_encoder.family_gate`,
  `model.target_context_encoder.gate_reduction`,
  `model.target_context_encoder.experimental_branch.{bottleneck,feature_dropout}`,
  `model.context_edge_interaction` (`none|film|mlp`),
  `model.interaction_edge_dim`.
- Runner: `scripts/run_sprint8a_target_context_interaction.py`, mirroring the 7F
  runner (`REFERENCE_RUN_IDS`, `HEADLINE_RUN_IDS`,
  `run_sprint8a_target_context_interaction(config_path, batch_id, max_epochs,
  selected_run_ids)`), reusing `train_graph_c_gcn`,
  `collect_graph_attention_summary`, `collect_target_context_encoder_summary`.
- Colab runner: `colab/sprint8a_target_context_interaction_runner.ipynb`
  (runner-only: mount/clone/sync, artifact provenance, call runner, copy outputs;
  no model or metric logic).
- Tests: `tests/test_sprint8a_target_context_interaction.py`.

## 9. Audit Requirements (interpretation-only)

Every trained Sprint 8A run writes
`runs/<run_id>/target_context_interaction_audit.csv` proving the controlled
variable and the frozen message passing:

- encoder type + flags (family_gate, gate_reduction, experimental bottleneck /
  feature-dropout);
- family branch names, input column counts (resolved as `115/6/78/13 = 212`),
  branch output dims;
- per-family learned gate weights (mean/std by split) when gating is on;
- context-edge interaction type (`none|film|mlp`), `interaction_edge_dim`, FiLM
  γ/β mean/std/L2 by split (interaction-only);
- target-context encoder parameter count and total model parameter count;
- `context_edges_used = 0` (every canonical run);
- candidate `S5F2_energy` attention edge-attr abs-sum > 0 (attention still uses
  S5F2);
- candidate `S5F2_energy` classifier edge-attr abs-sum > 0 (head still uses S5F2,
  raw or FiLM-modulated);
- branch-output and final target-embedding mean/std/L2 by split.

The audit must demonstrate that GATv2 attention/message passing is byte-for-byte
the Sprint 7F path and that the only changes are encoder fusion + classifier head.

## 10. Output Contract (Sprint 8A pattern, mirrors 7F)

Consolidated:

```text
outputs/sprint8a/target_context_interaction_comparison.csv
outputs/sprint8a/target_context_interaction_report.md
outputs/sprint8a/target_context_interaction_run_manifest.json
outputs/sprint8a/graph_artifact_provenance.json
```

Diagnostics:

```text
outputs/sprint8a/diagnostics/target_context_interaction_threshold_metrics.csv
outputs/sprint8a/diagnostics/target_context_interaction_deltas.csv
outputs/sprint8a/diagnostics/target_context_interaction_training_history.csv
outputs/sprint8a/diagnostics/target_context_interaction_predictions.csv
outputs/sprint8a/diagnostics/target_context_interaction_score_deciles.csv
outputs/sprint8a/diagnostics/target_context_interaction_per_guide_score_summary.csv
outputs/sprint8a/diagnostics/target_context_interaction_attention_summary.csv
outputs/sprint8a/diagnostics/target_context_interaction_audit.csv
outputs/sprint8a/diagnostics/target_context_interaction_branch_gate_summary.csv
outputs/sprint8a/diagnostics/target_context_interaction_film_summary.csv
outputs/sprint8a/diagnostics/target_context_interaction_parameter_counts.csv
```

Figures:

```text
outputs/sprint8a/figures/target_context_interaction_auprc_comparison.png
outputs/sprint8a/figures/target_context_interaction_threshold_metrics.png
outputs/sprint8a/figures/target_context_interaction_pr_curves.png
outputs/sprint8a/figures/target_context_interaction_roc_curves.png
outputs/sprint8a/figures/target_context_interaction_score_distributions.png
outputs/sprint8a/figures/target_context_interaction_training_curves.png
outputs/sprint8a/figures/target_context_interaction_gate_weights.png
outputs/sprint8a/figures/target_context_interaction_parameter_counts.png
```

Per-run directory:

```text
resolved_config.yaml
runtime.json
training_history.csv
metrics.csv
attention_summary.csv
target_context_interaction_audit.csv
model.pt   # Drive-held / untracked
```

## 11. Selection & Reporting Rules

- Primary selection metric: **validation AUPRC**. Tie-break: validation MCC, then
  validation macro F1.
- Test AUPRC is the primary reported test metric; test MCC/specificity/macro F1
  are secondary threshold-dependent diagnostics.
- Report **every** predeclared run (winners and losers). Parameter counts must be
  reported next to performance changes.
- Allowed conclusion shapes:
  - "If R1/R3 gating improves validation AUPRC while keeping/improving
    MCC/specificity at comparable parameter count, learned family fusion is a
    promising improvement over fixed concatenation."
  - "If R2/R3 FiLM interaction improves validation AUPRC, context-conditioned
    energy modulation helps under the frozen single-seed contract."
  - "If R4 regularization preserves AUPRC while reducing reliance on the
    experimental branch, the signal is usable without over-trusting a brittle
    branch."
  - "If a gated/interaction run's improvement tracks its added parameter count
    (judged against the reported per-run `parameter_count` and the Sprint 7F
    precedent that family-aware already beat the larger unified-deep encoder with
    fewer parameters), interpret the gain cautiously as possibly capacity-related
    rather than structural." (No parameter-matched control run is in the canonical
    matrix; this is a reporting caution, not a tested claim.)
  - "If none improve over the R0/R3 base, target-context architecture is not the
    current bottleneck; proceed to Sprint 8B sequence-context or Sprint 9
    robustness, not more target-context tuning."
- Disallowed claims: biological causality from gates/attention/FiLM; robustness
  from a single seed; treating an MCC/specificity gain as an AUPRC gain; "best
  seed"/"best rerun" selection; reproduction of SENet/FiBiNET/FiLM/GNN-FiLM/GATv2.

## 12. Tests & Validation

Wiring/contract tests in `tests/test_sprint8a_target_context_interaction.py`:

- family gate forward + 4 learned gate weights; gate OFF reproduces the base
  encoder output exactly.
- FiLM head forward: classifier input dim `128·4 + d_e`; FiLM OFF reproduces the
  raw-S5F2 classifier path; γ/β shapes `= d_e`.
- interaction-MLP fallback forward (shape + nonzero).
- regularized experimental branch: bottleneck dim wiring; feature-dropout active
  only in train mode.
- **frozen message-passing assertion:** with all new flags OFF, `GraphCEdgeGATv2`
  produces logits identical (within tolerance) to the Sprint 7F path; attention
  records and `graph_c_attention_edge_tensors` output unchanged.
- family-index resolution `115/6/78/13 = 212` via existing
  `validate_target_context_feature_names`.
- config parse: new keys map to `GCNRunConfig`; `drop_context_similarity_edges`
  stays true; `edge_aware_attention` stays true.
- runner writes the full §10 output contract (monkeypatched training, as 7F).

Commands (per `docs/COMMANDS.md`):

```bash
uv run pytest tests/test_sprint8a_target_context_interaction.py -q
uv run pytest tests/test_sprint7f_target_context_encoder.py tests/test_sprint7b_gatv2_model.py -q
uv run ruff check src/crispr_gnn/models/gat.py src/crispr_gnn/models/target_context_encoder.py src/crispr_gnn/training/gcn.py scripts/run_sprint8a_target_context_interaction.py tests/test_sprint8a_target_context_interaction.py
git diff --check
```

Do not start full training locally; the runner smoke test monkeypatches
training, as Sprint 7B/7D/7E/7F runner tests do.

## 13. Acceptance Criteria

Sprint 8A is ready to run when:

- The run matrix has exactly the 5 canonical trained rows (R0–R4); optional rows
  (`S8A_OPT_*`) are not run without a plan amendment before any test inspection.
- All canonical runs keep Graph C GATv2, S5F2 attention, weighted BCE, split,
  seed, checkpoint policy, threshold policy, and context-edge drop fixed.
- The only canonical controlled variables are the §6 deltas (gate / FiLM head /
  exp-branch regularization), with R3 = R1+R2 combination.
- Audits prove the frozen GATv2 message passing is unchanged, family indexes
  resolve to `115/6/78/13`, context edges are dropped, and S5F2 is active in both
  attention and the head.
- Output contract (§10) is complete; parameter counts reported per run.
- Local tests and lint pass; Colab notebook is runner-only; returned outputs
  exclude committed `model.pt`/`.DS_Store`.

Sprint 8A is complete when:

- Returned Colab outputs are copied under `outputs/sprint8a/`.
- The report interprets results under AUPRC-primary, validation-only-selection,
  no-test-tuning, single-seed, capacity-aware boundaries.
- The plan can move to `docs/exec-plans/completed/` only after validated outputs
  and the final report are present.

## 14. Deferred Work

- Axis-4 HP refinement is bounded and runs only as the optional final step on the
  validation winner (§7). Not a canonical row.
- Sprint 8B (sequence-context encoder adapted from CRISPR-Net) is a separate plan
  and runs only after 8A, to avoid confounding 8A's context gains.
- Robustness (multi-seed fixed-split, guide-level bootstrap CIs, paired-difference
  bootstrap) is re-scoped to **Sprint 9**.
- Source/cell-line/assay metadata confound modelling and external validation
  remain out of scope.

## 15. Required Docs Updates On Completion

To be made when 8A outputs are validated (not in this planning task):

- `docs/DECISIONS.md`: add an entry re-scoping Sprint 8 as a model-improvement
  sprint split into 8A (target-context + interaction) and 8B (sequence-context),
  with robustness moved to Sprint 9 — superseding the 2026-06-06 "Open optional
  Sprint 8 (Robustness)" entry. Record the locked base (R3), the 5-run matrix,
  the validation-AUPRC selection rule, and the head-only-interaction /
  frozen-message-passing decision.
- `README.md` and `CRISPR_GNN_PROJECT_PLAN.md`: update the roadmap to Sprint 6
  complete, Sprint 7/7B–7F complete, **Sprint 8 = model-improvement (8A core, 8B
  sequence-context)**, Sprint 9 = robustness.

## 16. Implementation Slices

Incremental, test-gated slices. Each slice keeps the frozen contract and the
frozen GATv2 message passing; no headline claim until Slice 6.

### Slice 0 - Planning freeze

Freeze this plan: run matrix (R0–R4), canonical base (S7F R3), interaction
mechanism (FiLM primary, interaction-MLP fallback), predeclared gate/FiLM/
exp-branch dims (§6), selection rule (validation AUPRC), and the output contract.

Exit: plan frozen; no code changed yet.

### Slice 1 - Encoder deltas (`target_context_encoder.py`) — Status: COMPLETE (2026-06-10)

Add the SENET-style learned family gate (§6.1) and the regularized
experimental-epigenetic branch (§6.3), both config-flagged and default OFF. Add
unit tests: gate forward + four learned gate weights; **gate-OFF reproduces the
base R3 encoder output exactly**; bottleneck wiring; feature-dropout active in
train mode only.

Exit: encoder unit tests pass; no training; base R3 path byte-for-byte unchanged
when flags are OFF.

Done: `FamilyAwareTargetContextEncoder` extended with `family_gate`,
`gate_reduction`, `experimental_branch_bottleneck`,
`experimental_branch_feature_dropout` (all keyword-only, default OFF; per §17.2/
§17.4); `build_target_context_encoder` forwards them and rejects the new options
on unified encoders; `activation_summary` adds `family_gate_enabled` /
`family_gate_weight_mean` columns while keeping the 4-row family structure.
Tests in `tests/test_sprint8a_target_context_interaction.py` (6) plus Sprint 7F/7B/
7E regression (15) pass; ruff + `git diff --check` clean. No `gat.py` /
`training/gcn.py` / config / runner change (Slices 2–3). No tech debt added.

### Slice 2 - Context-edge interaction head (`gat.py`)

Add the new Graph-C-specific FiLM head and the interaction-MLP fallback (§6.2)
inside `GraphCEdgeGATv2`, plus γ/β and interaction audit hooks. Leave the shared
`_classify_candidate_edges` / `_edge_classifier`, `_apply_attention_layers`, and
`graph_c_attention_edge_tensors` untouched. Add the **frozen-message-passing
assertion** test (all interaction flags OFF ⇒ logits identical to the Sprint 7F
path within tolerance) and FiLM/MLP shape tests.

Exit: model tests pass (incl. frozen-message-passing assertion); no canonical
training.

### Slice 3 - Trainer/config dispatch (`training/gcn.py`)

Extend `GCNRunConfig`, `gcn_run_config_from_mapping`, and `_build_model` to pass
the new flags to `GraphCEdgeGATv2`. No change to the training loop, loss,
scheduler, checkpoint, or threshold logic. Confirm Sprint 7F/7B regression tests
still pass and a tiny CPU smoke can build/forward R0–R4.

Exit: dispatch tests pass; Sprint 4–7F GCN/GATv2 regression tests stay green.

### Slice 4 - Runner, config, reporting, diagnostics, figures

Add `configs/sweeps/sprint8a_target_context_interaction.yaml` (5 runs + new keys),
`scripts/run_sprint8a_target_context_interaction.py` (mirror the 7F runner;
`REFERENCE_RUN_IDS` / `HEADLINE_RUN_IDS`), the §9 audits, the §10 diagnostics and
figures, manifest, provenance, and an output-contract test (monkeypatched
training, as 7F).

Exit: mocked/smoke outputs satisfy the §10 contract; no headline claim.

### Slice 5 - Colab runner preparation

Add `colab/sprint8a_target_context_interaction_runner.ipynb` (runner-only) and the
documented command path. Validate Drive artifact copy-in/out and returned-output
checks.

Exit: notebook contract checks pass; no full GPU claim yet.

### Slice 6 - Full canonical run and local validation

Run the 5 predeclared canonical runs (R0–R4, seed 42) on Colab GPU, copy outputs
back under `outputs/sprint8a/`, validate locally, and do not rerun or tune from
test diagnostics.

Exit: all §10 outputs exist (or any technical omission is documented before
interpreting results).

### Slice 7 - Optional axis-4 HP refinement (approval-gated)

Only after the validation-AUPRC winner among R0–R4 is identified: run the bounded,
predeclared ≤ 8-config refinement (§7) on that single architecture, validation-
only, reported as a separate labelled block.

Exit: refinement reported separately, or explicitly skipped. Canonical rows
unchanged.

### Slice 8 - Sprint closure

Freeze report/results/status, add the `docs/DECISIONS.md` re-scope entry (§15),
update `README.md` / `CRISPR_GNN_PROJECT_PLAN.md` roadmap, and move this plan to
`docs/exec-plans/completed/`.

Exit: the Sprint 8A conclusion is documented as one of the §11 allowed shapes; if
target-context architecture is not the bottleneck, proceed to Sprint 8B (sequence
context) or Sprint 9 (robustness) per a fresh plan.

## 17. Frozen Specification (Slice 0)

Slice 0 planning freeze (2026-06-10). Consistency re-audit against source and
`outputs/sprint7f/target_context_encoder_comparison.csv`: **PASS** (no drift). All
predeclared defaults below are pinned; Slices 1–8 implement against this section
with zero remaining design decision. No `src/`, `configs/`, `scripts/`, `colab/`,
or `tests/` file was changed in Slice 0.

### 17.1 Pinned base & frozen contract (recap)

- Base encoder = `family_aware_experimental_emphasis`, branch dims
  `target_sequence_one_hot=24, experimental_epigenetic=48,
  computed_nucleosome_aggregates=40, computed_nucleosome_missingness=16`
  (sum 128 = `hidden_dim`). Matches `EXPERIMENTAL_EMPHASIS_BRANCH_DIMS`.
- Target-observation input = 212 cols, family split `115/6/78/13`, resolved by
  `validate_target_context_feature_names` / `target_context_family_indices`.
- Frozen GATv2: 2 layers, hidden 128, 4 heads, concat true, dropout 0.2, attn
  dropout 0.2, `share_weights=false`, `self_loop_edge_fill=0.0`,
  `drop_context_similarity_edges=true`, `edge_aware_attention=true`; candidate
  `S5F2_energy` = 268 cols. Loss `weighted_bce` (`pos_weight=auto`); AdamW,
  `ReduceLROnPlateau(mode="max")` on `val_auprc`, grad clip 1.0, LR 1e-3, weight
  decay 1e-4, seed 42. These are unchanged for all canonical runs; only the §6
  deltas vary.

### 17.2 Pinned family gate v2 (R1, R3)

- Variant = **(b) excitation over the full 128-dim branch concat → 4 per-family
  scalar gates**. Rationale: with only 4 families a per-family-descriptor
  bottleneck (`4/r`) is degenerate; conditioning the gates on the full 128-dim
  branch concat keeps the SENET reduction meaningful and parameter-light.
- Squeeze: branch outputs `b = [b_seq(24), b_epi(48), b_agg(40), b_miss(16)]`
  (concat = 128, per node).
- Excitation: `s = σ(W2 · ReLU(W1 · b))`, `W1 ∈ R^{32×128}`, `W2 ∈ R^{4×32}`,
  `gate_reduction = 4` (128 → 32 → 4). Output = 4 per-family scalars.
- Re-weight: `b_f ← s_f · b_f` per family, then the existing fusion block
  (`LayerNorm(128) → ReLU → Dropout(0.2) → Linear(128→128) → ReLU`) unchanged.
- Parameter cost ≈ `128·32 + 32 + 32·4 + 4 ≈ 4.26k`. Reported in the audit.
- Gate-OFF (`family_gate=false`) reproduces the base R3 encoder output exactly
  (Slice 1 test).

### 17.3 Pinned context-edge interaction head (R2, R3; gat.py, Graph-C-specific)

- `interaction_edge_dim` (`d_e`) = **64**.
- Edge embedding: `edge_embed = Linear(268 → 64)(candidate_S5F2)`.
- Context embedding = post-GATv2 target-observation node embedding
  `target = x[target_index] ∈ R^{128}` (already computed in the head).
- **FiLM (primary, `context_edge_interaction=film`):** FiLM generator =
  `Linear(128 → 2·64)(target)` → split into `γ, β ∈ R^{64}`; modulate
  `edge_film = γ ⊙ edge_embed + β`. Classifier input =
  `[source(128), target(128), source*target(128), |source-target|(128),
  edge_film(64)]`, dim `128·4 + 64 = 576`.
- **Interaction-MLP (fallback, `context_edge_interaction=mlp`):**
  `proj = Linear(128 → 64)(target)`; `z = [edge_embed(64), target(128),
  edge_embed * proj(128→64)(64)]` (dim 256) → `Linear(256 → 64) → ReLU` →
  `interaction_vector(64)`; classifier input dim also `128·4 + 64 = 576`.
- The shared `_classify_candidate_edges` / `_edge_classifier` helpers,
  `_apply_attention_layers`, and `graph_c_attention_edge_tensors` are NOT modified
  (Graph A/B and the `none` path keep the raw-`S5F2` classifier, dim
  `128·4 + 268`). Full bilinear excluded.

### 17.4 Pinned regularized experimental branch (R4)

- Applied to the `experimental_epigenetic` (6-col) branch only.
- `experimental_branch_feature_dropout` = **0.3** (Bernoulli over the 6 input
  columns, train mode only).
- `experimental_branch_bottleneck` = **4**. Branch shape becomes
  `feature_dropout(6) → Linear(6→4) → LayerNorm(4) → ReLU → Linear(4→48) →
  LayerNorm(48) → ReLU` (replaces the base `Linear(6→48) → LayerNorm → ReLU`).
- OFF defaults (`bottleneck=None`, `feature_dropout=0.0`) reproduce the base
  branch exactly.

### 17.5 Pinned new config keys (defaults reproduce Sprint 7F R3 when OFF)

| Key | Type | Default | Carried via |
| --- | --- | --- | --- |
| `model.target_context_encoder.type` | str | `family_aware_experimental_emphasis` | existing |
| `model.target_context_encoder.family_gate` | bool | `false` | `GCNRunConfig.family_gate` |
| `model.target_context_encoder.gate_reduction` | int | `4` | `GCNRunConfig.gate_reduction` |
| `model.target_context_encoder.experimental_branch.bottleneck` | int \| null | `null` | `GCNRunConfig.experimental_branch_bottleneck` |
| `model.target_context_encoder.experimental_branch.feature_dropout` | float | `0.0` | `GCNRunConfig.experimental_branch_feature_dropout` |
| `model.context_edge_interaction` | str (`none\|film\|mlp`) | `none` | `GCNRunConfig.context_edge_interaction` |
| `model.interaction_edge_dim` | int | `64` | `GCNRunConfig.interaction_edge_dim` |

Selection mechanism is **flag-based** (a `family_gate` flag + branch/interaction
fields), NOT new encoder-type aliases — keeps the encoder-type namespace small.
Flow: `gcn_run_config_from_mapping` → `GCNRunConfig` → `_build_model` →
`build_target_context_encoder` (gate + exp-branch flags) and `GraphCEdgeGATv2`
(`context_edge_interaction`, `interaction_edge_dim`).

### 17.6 Pinned 5-run canonical matrix (flag table)

| Run ID | `type` | `family_gate` | `experimental_branch.bottleneck` | `experimental_branch.feature_dropout` | `context_edge_interaction` | `interaction_edge_dim` | Controlled variable |
| --- | --- | :--: | :--: | :--: | :--: | :--: | --- |
| `S8A_R0_base_reference` | exp_emphasis | false | null | 0.0 | none | (64, unused) | none (reproduces S7F R3) |
| `S8A_R1_family_gated_v2` | exp_emphasis | **true** | null | 0.0 | none | (unused) | family gate |
| `S8A_R2_context_edge_film` | exp_emphasis | false | null | 0.0 | **film** | 64 | FiLM head |
| `S8A_R3_gated_plus_film` | exp_emphasis | **true** | null | 0.0 | **film** | 64 | gate + FiLM (R1+R2) |
| `S8A_R4_regularized_exp_branch` | exp_emphasis | false | **4** | **0.3** | none | (unused) | exp-branch regularization |

All canonical runs keep `drop_context_similarity_edges=true`,
`edge_aware_attention=true`, and the frozen GATv2 config. No run sets
`context_edge_interaction=mlp` (that is the optional `S8A_OPT_mlp_head` fallback,
validation-gated only). Carry-forward reference IDs (no retrain):
`S8A_REF_XGB_F4`, `S8A_REF_GRAPH_A_GCN`, `S8A_REF_GRAPH_C_GCN`,
`S8A_REF_FULL_GRAPH_C_GATV2`, `S8A_REF_NO_CTX_EDGE_GATV2`, `S8A_REF_S7F_R2`,
`S8A_REF_S7F_R3`.

### 17.7 New modules/classes/functions to add (Slices 1–2)

- `target_context_encoder.py`: extend `FamilyAwareTargetContextEncoder.__init__`
  with `family_gate: bool=False`, `gate_reduction: int=4`,
  `experimental_branch_bottleneck: int|None=None`,
  `experimental_branch_feature_dropout: float=0.0`; add the gate submodule
  (SE excitation per 17.2) and the regularized experimental branch (17.4); extend
  `build_target_context_encoder(...)` signature to forward these; extend
  `activation_summary` to emit per-family gate weights.
- `gat.py`: extend `GraphCEdgeGATv2.__init__` with `context_edge_interaction:
  str="none"`, `interaction_edge_dim: int=64`; add a Graph-C-specific interaction
  head (FiLM + MLP per 17.3) and a `context_edge_interaction_summary(...)` audit
  method (γ/β mean/std/L2 + interaction nonzero by split). Shared helpers untouched.
- `training/gcn.py`: add the six `GCNRunConfig` fields (17.5); read them in
  `gcn_run_config_from_mapping`; pass them in `_build_model`; add
  `collect_context_edge_interaction_summary(...)` analogous to
  `collect_target_context_encoder_summary`.

### 17.8 Output contract & tests (already specified)

- Output contract: see §10 (consolidated CSV/report/manifest/provenance,
  diagnostics incl. `branch_gate_summary.csv` + `film_summary.csv`, figures,
  per-run dirs).
- Tests: see §12 (gate forward + gate-OFF base reproduction; FiLM/MLP shape;
  exp-branch wiring + train-only feature-dropout; **frozen-message-passing
  assertion**; family-index `115/6/78/13`; config parse; output-contract).

### 17.9 Slice 0 exit

Sprint 8A plan frozen; Slices 1–8 may proceed against §17 Frozen Specification;
no code changed.
