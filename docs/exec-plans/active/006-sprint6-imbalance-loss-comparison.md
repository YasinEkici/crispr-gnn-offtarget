# Execution Plan: Sprint 6 Imbalance / Loss Comparison

> Status: active. Run list (Section 6) and hyperparameters FROZEN and approved
> 2026-06-06 — Slice 0 (planning/decision freeze) exit gate met. Implementation
> (Slice 1+) may proceed. Predeclared losses/γ/α/β must not change from test
> diagnostics.

## 1. Goal

Sprint 6 isolates the effect of the **training objective** (loss function and/or
training-time sampling) on the Sprint 5 best GCN setting, under the frozen
guide-level evaluation contract.

Sprint 6 will:

- Hold graph schema (`graph_a_minimal_physical_target`), feature set
  (`S5F2_energy`, 268 edge-feature columns), model architecture
  (`GraphAEdgeGCN`), optimizer (AdamW), scheduler (`ReduceLROnPlateau` on
  `val_auprc`), seed (42), and split (`sprint2_main_seed42`) **fixed**.
- Vary **only** the loss function and/or training-time sampling strategy.
- Predeclare every run (loss × hyperparameter) before any training.
- Produce a consolidated result table, diagnostics, figures, provenance, and a
  Markdown report under the same reporting contract as Sprint 4/Sprint 5.
- Add positive-retrieval, **negative-retrieval**, and across-guide variability
  diagnostics required by the imbalance literature.
- Use Colab only as a runner.

Sprint 6 will not:

- Change graph schema, feature set, target-node semantics, architecture,
  optimizer, scheduler, seed, or split.
- Tune any loss/sampling hyperparameter from test diagnostics.
- Introduce GAT/GATv2, GraphSAGE, R-GCN, HGT, or edge-aware message passing
  (Sprint 7 scope).
- Move `measured=0` putative rows into the headline train/validation/test
  universe.
- Claim reproduction of Mak et al. 2022, Gao 2020, or Guan 2024 setups.

## 2. Sprint 5 Inheritance (Findings And Risks Carried In)

This section reproduces the Sprint 5 → Sprint 6 handoff conclusions so the plan
is self-contained. All numbers verified against the on-disk Sprint 5/5B
artifacts.

### 2.1 Sprint 5 findings entering Sprint 6

- `S5F2_energy` (binding energy `energy_1`–`energy_5`) is the strongest GCN
  feature-family result: test AUPRC `0.976585`, AUROC `0.817765`, macro F1
  `0.695284`, MCC `0.477933`, TN/FP/FN/TP `48/121/6/1527`. It beats Sprint 4
  Graph A (`0.966287`) and is the canonical frozen forward setting.
- Experimental epigenetic scalars (`S5F3_experimental_epi`) do **not** improve
  over `S5F2_energy`.
- Computed nucleosome features (`S5F4_computed_agg` AUPRC `0.910866`,
  `S5F5_computed_pos` AUPRC `0.909031`) collapse to zero true negatives and drop
  below the Sprint 4 Graph A AUPRC.
- Sprint 5B (`GraphCContext+S5F2_energy`): AUPRC `0.972481`, MCC `0.274287`,
  TN/FP/FN/TP `14/155/0/1533`. Energy features help Graph C's AUPRC vs Sprint 4
  Graph C (`0.961586`) but do not beat Graph A `S5F2_energy`.
- **Threshold collapse** (MCC≈0, TN≈0, specificity 0) recurs in `S5F1_mismatch`,
  `S5F4_computed_agg`, `S5F5_computed_pos`, and Sprint 5B Graph C. This is the
  central motivation for Sprint 6.
- **No GCN result beats XGBoost F4** on primary AUPRC (`0.992522`). F4 remains
  the authoritative bar.
- **AUPRC prevalence floor:** the Sprint 2 `dummy_prior` baseline scores test
  AUPRC `0.900705` = positive prevalence. At ~90% positives, AUPRC has a high,
  "easy" floor; loss/sampling changes will move negative-class threshold metrics
  (specificity, true-negative rate, MCC, macro F1) far more visibly than AUPRC.

### 2.2 Architecture reality inherited from Sprint 4/5

In `GraphAEdgeGCN` (`src/crispr_gnn/models/gcn.py`), candidate-edge features
(including `S5F2_energy`) **do not enter GCN message passing**. They are
concatenated only at the final edge classifier head (`gcn.py:80`); the
convolution sees guide encodings + a shared featureless target type vector.
Consequence: the threshold-collapse pattern may be partly architectural, not
purely a loss artifact. Sprint 6 must produce evidence to distinguish the two and
must not attribute collapse to "loss alone."

### 2.3 Risks carried into Sprint 6 (from the handoff)

1. **Post-hoc loss design (primary no-test-tuning risk):** loss comparisons must
   be predeclared, not chosen after inspecting Sprint 5 test diagnostics.
2. **Focal γ / Dice (Tversky) α,β not predeclared:** risk of tuning from test.
3. **Sampling silently changes the train pos:neg ratio** relative to the locked
   evaluation population — must be documented as a training-only audit.
4. **Threshold-collapse misattribution** to loss without architecture/feature
   evidence (see 2.2).
5. **Metric confusion:** improved MCC/macro F1 must not be reported as AUPRC
   improvement; check the primary metric first.
6. **Prevalence inversion:** the measured-only universe is ~90% positive / ~10%
   negative. The scarce class is **negatives**, the opposite of Gao 2020 /
   Guan 2024 (~99% negative). Their positive-oversampling / SMOTE recipes do not
   transfer un-inverted.
7. **Holding `S5F2_energy` fixed:** Sprint 6 must not re-tune the feature set
   from its own loss-experiment diagnostics.
8. **Single-seed fragility:** all prior results are single-seed/single-split
   (DECISIONS 2026-06-01). AUPRC ranking is threshold-free and relatively
   stable; MCC/TN advantages are threshold-sensitive and fragile.
9. **Colab notebook logic escaping version control** — keep Sprint 6 logic under
   `src/` and `scripts/`.
10. **Diagnostic completeness:** positive/negative-retrieval and per-guide
    variability figures do not yet exist; Sprint 6 must build them.

## 3. Frozen Evaluation Contract

Sprint 6 preserves the Sprint 2/3/4/5 contract:

- Scheme A label exactly: `int(cleavage_freq > 1e-5)`.
- NaN `cleavage_freq` excluded from supervised labels; negative and `>1` values
  per Sprint 1 outlier policy.
- Split `sprint2_main_seed42`; guide-disjoint train/validation/test.
- Headline train/validation/test are measured-only; `experiment_id=18` excluded.
- No `measured=0` rows in the headline universe.
- Train-only imputation/scaling (reuse the existing Sprint 5 Graph A
  `S5F2_energy` feature table; do not refit on val/test).
- Validation-only checkpoint selection (`val_auprc`).
- Validation-only threshold selection (`select_threshold_by_f1`,
  `validation_max_f1`). **This stays fixed** so all Sprint 6 rows are
  same-contract comparable to Sprint 2–5 and to XGBoost F4. An alternative
  threshold-policy sweep, if produced, is interpretation-only and must not
  replace the headline policy.
- Test used only for final reporting; no test-driven tuning.
- AUPRC primary, with positive prevalence (`0.900705`) reported on every figure.
- Required comparison row: `xgboost_unweighted / F4` — test AUPRC `0.992522`,
  AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.
- Required prior-context rows: Sprint 4 Graph A (`0.966287` / MCC `0.300781`),
  Sprint 5 Graph A `S5F2_energy` (`0.976585` / MCC `0.477933`).

## 4. Frozen Forward Setting (Held Constant In Every Run)

| Axis | Value | Source |
| --- | --- | --- |
| Graph schema | `graph_a_minimal_physical_target` | Sprint 4/5 |
| Feature set | `S5F2_energy` (268 cols) | Sprint 5 best |
| Graph artifacts | `data/processed/graphs/sprint5/` (existing `s5f2_energy` edge table) | Sprint 5 build |
| Model | `GraphAEdgeGCN`, 2-layer `GCNConv`, hidden 128, LayerNorm+ReLU, dropout 0.2 | `models/gcn.py` |
| Optimizer | AdamW, lr 1e-3, weight_decay 1e-4 | `training/gcn.py` |
| Scheduler | `ReduceLROnPlateau` on `val_auprc`, factor 0.5, patience 5, min_lr 1e-5 | `training/gcn.py` |
| Grad clip | max_norm 1.0 | `training/gcn.py` |
| Epochs | max 300, min 5, patience 15 | Sprint 5 config |
| Seed / split | 42 / `sprint2_main_seed42` | frozen |
| AMP / compile | bfloat16 AMP + `torch.compile` on CUDA only | DECISIONS 2026-05-30 |

**Controlled variable: loss function and/or training-time sampling only.**

## 5. Primary vs Secondary Regime Decision

- **Primary (headline):** strictly **measured-only**. Loss comparison plus, if
  used, measured-only sampling. Same contract as Sprints 2–5; directly
  comparable to XGBoost F4.
- **Secondary (optional, approval-gated):** a separately named
  `putative_augmented_screening` / `genome_wide_candidate_filtering` track that
  uses `measured=0` rows as **training-only noisy negatives** (never in
  validation/test). It is reported separately, must not be mixed into the
  headline loss table, and follows the Sprint 5B "secondary sensitivity"
  precedent. It is **not** part of the default Sprint 6 scope and requires
  explicit predeclaration before any run. `measured=0` rows are putative/
  unmeasured candidates and must never be labeled true negatives.

`measured=0` is the switch into the low-prevalence genome-wide screening regime
described by Gao 2020 and Guan 2024. The measured-only universe is ~90% positive
(negatives rare); adding the ~284K putative `measured=0` candidates flips the
dataset to ~7% positive (negatives dominant, ≈1:13). The literature's
positive-oversampling / SMOTE recommendations apply **only** in that second
regime, where the *positive* class is rare — they do not transfer to the
measured-only headline, where the *negative* class is rare.

Rationale for keeping Sprint 6 measured-only: (1) mixing `measured=0` would change
two axes at once (loss and data regime), making the loss effect unidentifiable;
(2) it would break same-contract comparability with XGBoost F4 and Sprints 2–5;
(3) the measured-only train set has only 901 negatives across 98 guides, so
balanced sampling can only upsample those few negatives (overfitting risk) and
hard-negative mining has almost nothing to mine without crossing into the
screening regime. Keeping the regimes separate preserves comparability and keeps
the loss experiment clean.

## 6. Predeclared Run List (Freeze Before Training)

All runs use the Section 4 frozen setting. Hyperparameters below are predeclared
and must not be changed from test diagnostics. Direction note: because negatives
are the minority here, class-weighting is set to protect the **negative** class
(inverse of Gao/Guan).

| Run ID | Objective | Predeclared parameters | Role |
| --- | --- | --- | --- |
| `S6R0_wbce` | Weighted BCE (Sprint 5 baseline, re-run in Sprint 6 harness) | `pos_weight = negatives/positives` (data-derived, ≈0.1267) | Baseline / continuity check vs Sprint 5 `S5F2_energy` |
| `S6R1_bce_unw` | Unweighted BCE | `pos_weight = 1.0` | Control: isolates the effect of weighting |
| `S6R2_focal_g2_a25` | Focal loss | γ=2.0, α=0.25 | Lin 2017 / Guan 2024 strong setting (small α up-weights rare negatives) |
| `S6R3_focal_g1_a25` | Focal loss | γ=1.0, α=0.25 | Milder focusing |
| `S6R4_focal_g2_a50` | Focal loss | γ=2.0, α=0.50 | Pure focusing, no class α |
| `S6R5_dice` | Soft / generalized Dice | smoothing ε=1.0, class weights from train frequency | Sudre 2017 |
| `S6R6_tversky_a70_b30` | Tversky (Dice generalization) | α_FP=0.70, β_FN=0.30, ε=1.0 | Penalize FP more → recover negatives |
| `S6R7_balanced_sampling` | Unweighted BCE + balanced supervised-edge subsampling | per-epoch majority(positive)-class subsampling to 1:1 train ratio | Sampling axis (measured-only) |

Optional / approval-gated (declare before running, report separately):

| Run ID | Objective | Notes |
| --- | --- | --- |
| `S6R8_classbalanced_cui` | Class-balanced reweighted BCE (Cui 2019, effective number, β=0.999) | Refinement of inverse-frequency weighting; full-batch friendly |
| `S6R9_hard_neg_mining` | Hard-negative mining | Likely deferred — little to mine in measured-only (see Section 5) |
| `S6S1_putative_augmented` | Best headline loss + `measured=0` training-only negatives | Secondary `putative_augmented_screening` regime only |

Total predeclared headline runs: **8** (`S6R0`–`S6R7`). Each is one Colab GPU run.

### 6.1 Literature basis and formula references (predeclared, not tuned)

These values come from a Sprint-6 literature review (axis_2 notes + primary
sources); they are frozen on review and cannot be changed from test diagnostics.

These losses are **adapted** to this project's inverted class structure (rare
class = negative), not reproduced from their source experiments. The direction of
every class-asymmetric parameter is validated below and must be enforced by a
unit test (Section 9), because the effect depends on the implementation's
foreground/weight convention.

**Focal loss (Lin 2017; Guan 2024 eq. 3) — NOT inverted:**
`FL = -α(1-p)^γ log(p)` if `y=1`; `-(1-α)p^γ log(1-p)` if `y=0`.
- **γ is class-agnostic:** `(1-p)^γ` down-weights confident-correct examples and
  up-weights hard/misclassified ones regardless of class. Our hard cases are the
  rare negatives (the FP=121), so γ>0 helps in the right direction **without any
  inversion**. γ=2 is the validated default (robust over γ∈[0.5,5]; too large
  destabilizes training).
- **α is kept, not flipped:** in this formula α weights the positive (y=1) term
  and `(1-α)` the negative term, so α=0.25 already places 0.75 on the rare
  negative class (correct direction). α=0.25 is a *directionally-correct
  transferred* value, not an optimum re-validated for the inverted class
  structure (Lin's α=0.25 was jointly tuned with γ=2 for a rare-*positive*
  structure; the α–γ interaction optimum may differ here). `S6R4` (α=0.5, pure
  focusing) hedges this α-sensitivity. α>0.5 is excluded (down-weights negatives).
- Guan 2024 found focal loss the best/most stable cost-sensitive method across
  CRISPR off-target models — hence the primary candidate.

**Dice / generalized Dice (Sudre 2017):** overlap-based, robust to learning rate.
- **Must be generalized Dice (inverse-volume class weights), not single-class
  Dice on the positive class.** At ~90% positive prevalence, plain Dice on the
  majority foreground has a documented degenerate-solution / region-size bias and
  yields high precision / low recall on the rare class. Generalized Dice weights
  each class by `1/(Σlabel)²` (from train frequencies, not tuned), giving the
  rare negative proper weight; ε=1.0 smoothing. Optionally combined with BCE.
- This high-precision/low-recall-on-the-rare-class failure is exactly why Tversky
  is included as the targeted generalization.

**Tversky (Salehi 2017) — α>β is the documented specificity setting, validated:**
`TI = TP/(TP + α·FP + β·FN)`, loss = 1-TI (`α+β=1`; Dice = α=β=0.5).
- The parameters exist to choose the FP/FN trade-off: **α>β reduces false
  positives / raises specificity; β>α reduces false negatives / raises recall**
  (documented, intended use).
- Literature standard α=0.3, β=0.7 targets the rare *positive/foreground* class.
  This project is inverted: rare class = negative, failure = excess FP
  (S5F2_energy: FP=121 vs FN=6; specificity ≈0). We set **α=0.70, β=0.30** to
  penalize FP more.
- This is not an ad-hoc inversion: it is **mathematically equivalent** to
  applying Salehi's standard α=0.3/β=0.7 with the negative (minority) class as
  foreground (both penalize the same 121 negatives-called-positive). Predeclared
  from the confusion profile, not chosen from test results.

**Class-balanced reweighting (Cui 2019, optional `S6R8`):** effective number
`(1-βⁿ)/(1-β)`, β=0.999 — a smoother, more stable alternative to naive
inverse-frequency weighting at this moderate (~8:1) imbalance.

**Cost-sensitive vs sampling (Gao 2020; Guan 2024; general imbalance reviews):**
for deep nets with a small minority pool, loss-based cost-sensitive methods are
generally preferred over resampling (which risks overfitting on the 901 unique
train negatives). Sampling (`S6R7`) is therefore secondary, not headline.

## 7. Required Repository Changes

New (do not exist yet):

- `src/crispr_gnn/models/losses.py` — loss registry: `weighted_bce`,
  `bce_unweighted`, `focal`, `dice`, `tversky`, and a `class_balanced` (Cui)
  variant; each returning a callable consuming logits + labels (+ optional
  per-sample weights). Numerically stable; AMP-safe (cast loss inputs to float32
  as the existing trainer does).
- `src/crispr_gnn/training/samplers.py` — measured-only balanced supervised-edge
  selection for `S6R7` (per-epoch deterministic-by-seed positive subsampling;
  all negatives retained).
- `scripts/run_sprint6_loss_comparison.py` — sweep runner mirroring
  `scripts/run_sprint5_feature_ablation.py`: iterate the predeclared run list,
  write per-run `resolved_config.yaml` / `runtime.json` / `training_history.csv`
  / `metrics.csv` / `model.pt`, then consolidated results/diagnostics/figures/
  report + provenance.
- `configs/sweeps/sprint6_loss_comparison.yaml` — declares the frozen Section 4
  setting and the Section 6 predeclared run list (loss name + params per run).
- `colab/sprint6_loss_comparison_runner.ipynb` — runner-only notebook.
- `tests/test_sprint6_losses.py` — see Section 9.

Modified:

- `src/crispr_gnn/training/gcn.py` — replace the hard-locked
  `loss == "weighted_bce"` check (`gcn.py:127-129`) and the inline
  `BCEWithLogitsLoss` construction with a call into the loss registry, gated to
  the predeclared Sprint 6 loss set. Keep weighted BCE the default so Sprint 4/5
  configs are unaffected. No change to label/split/visibility/threshold/
  checkpoint logic.
- `src/crispr_gnn/evaluation/diagnostics.py` and `plots.py` — add
  positive-retrieval, **negative-retrieval (per-guide TNR)**, per-guide metric
  distribution, and threshold-metric diagnostics/figures.
- `docs/DECISIONS.md` — record the predeclared loss set, γ/α,β values, the
  measured-only headline vs `putative_augmented_screening` boundary, and the
  no-test-tuning freeze.
- `docs/COMMANDS.md`, `docs/PROJECT_CONTEXT.md` (after completion), `README.md`
  (after completion), `colab/README.md`.

## 8. Output Contract

```text
outputs/sprint6/loss_comparison/
  sprint6_loss_comparison_results.csv
  sprint6_loss_comparison_report.md
  sprint6_loss_comparison_run_manifest.json
  graph_artifact_provenance.json
  diagnostics_sprint6/
  figures_sprint6/
  runs/<run_id>/{resolved_config.yaml,runtime.json,training_history.csv,metrics.csv,model.pt}
```

Required figures (aligning project-plan §Sprint 6 names; prevalence + F4 + Sprint
5 `S5F2_energy` reference shown where applicable):

```text
figures_sprint6/imbalance_auprc_comparison.png
figures_sprint6/imbalance_pr_curves.png
figures_sprint6/imbalance_threshold_metrics.png        # specificity/MCC/TNR vs loss
figures_sprint6/imbalance_score_distributions.png
figures_sprint6/imbalance_per_guide_metric_distribution.png
figures_sprint6/imbalance_positive_retrieval_summary.png
figures_sprint6/imbalance_negative_retrieval_summary.png   # inverted-prevalence diagnostic
```

Required diagnostic tables: consolidated metrics by run, per-run confusion
(TN/FP/FN/TP at the validation-selected threshold), per-guide metrics, per-genome
metrics, score deciles, positive- and negative-retrieval summaries, and a
training-distribution audit for any sampling run (`S6R7`).

Report must be structured to mirror the Sprint 5 handoff format and state, for
every run: AUPRC (primary) with prevalence, then AUROC, macro F1, MCC,
specificity, TN/FP/FN/TP; the delta vs `S6R0_wbce`; and an explicit
"AUPRC-first" interpretation that does not present MCC/macro-F1 gains as AUPRC
gains.

`model.pt`, copied graph artifacts, raw data, caches, and Drive-local folders
remain untracked.

## 9. Tests

Loss-registry tests (`tests/test_sprint6_losses.py`):

- Each loss returns a finite scalar on a tiny logits/labels fixture.
- Weighted BCE reproduces the existing `BCEWithLogitsLoss(pos_weight=...)` value
  (regression guard so the refactor does not change Sprint 4/5 behavior).
- Focal reduces to (weighted) BCE at γ=0; γ>0 down-weights confident-correct
  examples relative to hard ones.
- Tversky reduces to Dice at α=β=0.5.
- **Direction guard (validates the inversion/orientation):** on a fixture with
  excess false positives (negatives predicted positive), increasing Tversky α
  must increase the FP penalty term; increasing the negative-class weight in
  focal (lower α) and using generalized Dice (inverse-volume) must raise the loss
  contribution of misclassified negatives. This enforces that the implemented
  foreground/weight convention actually targets the FP-type errors, not the
  FN-type — i.e., the loss protects the rare negative class as intended.
- Predeclared params are read from config, not hard-coded in the loss call site.

Sampling tests:

- Balanced sampler keeps all negatives and subsamples positives to the declared
  ratio; selection is deterministic under the fixed seed; it never touches
  validation/test masks.

Contract/leakage tests (extend existing patterns):

- Reject non-predeclared loss names.
- Reject any change to split ID, label scheme, visibility, feature set, or
  threshold/checkpoint selection split.
- Reject `measured=0` rows in headline val/test.
- Assert the frozen feature set is `S5F2_energy` for every headline run.

Output tests: required CSV/report/diagnostics/figures/provenance/per-run files
exist for completed runs.

Run `uv run pytest -q` and one CPU smoke run (`--max-epochs 1`) before any
headline Colab run.

## 10. Colab Runner Workflow

Colab is a runner only (clone/checkout approved commit, `uv sync`, copy Sprint 5
Graph A artifacts from Drive, validate provenance, run the sweep, copy outputs
back, verify). Notebook cells must not implement losses, samplers, training,
evaluation, or plotting.

```bash
uv run python scripts/run_sprint6_loss_comparison.py \
  --config configs/sweeps/sprint6_loss_comparison.yaml \
  --run-id sprint6_loss_comparison_seed42_<timestamp>
```

## 11. Acceptance Criteria

- This plan is reviewed and the predeclared run list (Section 6) is frozen
  before implementation.
- `docs/DECISIONS.md` records the predeclared losses, γ/α,β, and the
  measured-only vs `putative_augmented_screening` boundary.
- Loss registry + sampler implemented and tested; weighted-BCE regression guard
  passes (Sprint 4/5 behavior unchanged).
- All predeclared headline runs (`S6R0`–`S6R7`) completed, or any
  skipped/failed run documented with a technical reason.
- Frozen Section 4 setting held constant across all runs (verified in
  provenance: feature set `S5F2_energy`, schema, seed, split, architecture).
- Validation-only checkpoint and threshold selection; no test-driven tuning.
- AUPRC primary with prevalence on every figure; F4 and Sprint 5 `S5F2_energy`
  shown as references.
- Positive-retrieval, negative-retrieval, per-guide variability, and
  threshold-metric figures present.
- Report states the threshold-collapse interpretation **with** the architecture
  caveat from Section 2.2 (does not attribute collapse to loss alone).
- Report does not claim Gao/Guan/Mak reproduction; states prevalence-inversion
  context explicitly.
- `uv run pytest -q` passes; one smoke run validated before headline runs.

## 12. Implementation Slices

### Slice 0 — Planning and decision freeze
Finalize this plan; record decisions; freeze the run list. Exit: Sections 3–6
documented before code.

### Slice 1 — Loss registry and sampler  — Status: COMPLETE (2026-06-06)
Implement `models/losses.py` + `training/samplers.py`; refactor
`training/gcn.py` to dispatch losses; add tests incl. weighted-BCE regression
guard. Exit: loss/sampler tests pass; no training yet.

Done: `src/crispr_gnn/models/losses.py` (registry: weighted_bce, bce_unweighted,
focal, generalized_dice, tversky, class_balanced_bce + `build_loss`),
`src/crispr_gnn/training/samplers.py` (measured-only balanced subsample),
`training/gcn.py` dispatches via `build_loss` and wires the sampler (weighted_bce
default preserved, regression-guarded). Tests: `tests/test_sprint6_losses.py`
(10) + trainer-level Sprint 6 tests in `tests/test_gcn_training_smoke.py`;
`tests/test_gcn_evaluation_contract.py` updated for the new permitted-loss
contract. Full suite 121 passed; ruff clean. No GPU/Colab training run.

### Slice 2 — Sweep runner, config, reporting extensions — Status: COMPLETE (2026-06-06)
Implement `scripts/run_sprint6_loss_comparison.py` +
`configs/sweeps/sprint6_loss_comparison.yaml`; extend diagnostics/plots with
positive/negative-retrieval and per-guide variability. Exit: CPU smoke run
produces the required output directory structure.

Done: `scripts/run_sprint6_loss_comparison.py` consumes the frozen
`configs/sweeps/sprint6_loss_comparison.yaml`, defaults to headline `S6R0`-
`S6R7`, requires explicit opt-in for optional runs, records per-run `run_id`,
`loss_params`, and `sampling` provenance, validates Graph A `S5F2_energy`
artifacts before training, and writes the Slice 2 output contract under
`outputs/sprint6/loss_comparison/`. Additive Sprint 6 diagnostics/plots now
produce positive-retrieval, negative-retrieval/TNR, threshold-metric, score,
and per-guide distribution outputs without changing Sprint 4/5 reporting call
sites. Tests cover the output contract, optional-run gating, `run_id`
uniqueness, and resolved-config loss/sampling provenance. Full suite passed:
`uv run pytest -q` (124 passed); ruff clean.

### Slice 3 — Colab runner preparation
Runner-only notebook; provenance gate; returned-artifact checklist. Exit: runner
documented; no headline result claimed.

### Slice 4 — Full predeclared loss comparison (Colab GPU)
Run `S6R0`–`S6R7`; copy outputs to Drive; validate locally. Exit: all runs
present or documented; output validation passes; no test-driven reruns.

### Slice 5 — Optional secondary track (approval-gated)
Only if approved: run `S6S1_putative_augmented` (and/or `S6R8`/`S6R9`) as a
separately named, separately reported regime. Exit: secondary report explicitly
states it is not headline and that `measured=0` are not true negatives.

### Slice 6 — Final Sprint 6 report freeze
Finalize report/CSV/diagnostics/figures/status docs. Move this plan to
`docs/exec-plans/completed/`.

Final interpretation must state: under the locked Scheme A, guide-level,
measured-only, `experiment_id=18`-excluded protocol and the fixed Graph A
`S5F2_energy` setting, objective X changed AUPRC by Y and negative-class
recognition (specificity/TNR/MCC) by Z relative to weighted BCE; whether any
objective improves negative-class recognition without sacrificing AUPRC; and
whether residual collapse implicates architecture/feature distribution
(→ Sprint 7) rather than loss alone. This is not Gao/Guan/Mak reproduction and
used no test-driven selection.
