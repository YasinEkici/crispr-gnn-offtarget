# Execution Plan: Sprint 9 Robustness — Uncertainty and Variance Quantification

> Status: ACTIVE — Slice 0 planning freeze (2026-06-13). This plan freezes the
> Sprint 9 robustness scope; Slices 1–6 implement against §15 Frozen Specification
> with no remaining design decision. Sprint 9 is **interpretation-only**: it
> changes no labels, splits, thresholds, architectures, or any frozen Sprint 8
> result. It quantifies the uncertainty/variance behind the already-frozen
> Sprint 7F/8A/8B candidates. Companion completed plans:
> `../completed/008-sprint8a-target-context-interaction.md`,
> `../completed/008b-sprint8b-sequence-context-encoder.md`.

## 1. Goal

Sprints 8A and 8B each selected a candidate by **validation** AUPRC
(`S8A_R2_context_edge_film`, `S8B_R2_sequence_plus_context`) but explicitly
deferred any **superiority/robustness** claim to Sprint 9. Sprint 9 answers the
single question the handoff poses:

> Are the Sprint 8A / 8B candidate improvements robust enough to support a
> stronger claim, or are they within single-seed / guide-composition uncertainty?

It does this with three **non-interchangeable** uncertainty summaries (per
`docs/literature/sprint9-deep-research.pdf` §1), which must never be merged into a
single generic interval because they target different sources of variation:

1. **Guide-cluster bootstrap CIs** — test-set sampling uncertainty under
   guide-level resampling of the one fixed held-out split.
2. **Paired guide-cluster bootstrap** — model-to-model differences on **common**
   guide resamples from that same split.
3. **Predeclared multi-seed retraining** — sensitivity of the training pipeline to
   stochastic optimization/initialization on the same fixed split.

### AGENTS.md workflow summary

- **Task goal:** build a tested robustness layer that (a) replays saved per-row
  predictions into exact full-split metrics, (b) computes guide-cluster bootstrap
  CIs, (c) computes paired-difference bootstrap for a predeclared comparison
  matrix, (d) regenerates XGBoost F4 per-row predictions under the Sprint 2
  contract, and (e) runs predeclared multi-seed retraining for the headline
  configs — then consolidates a robustness report with explicit claim boundaries.
- **Expected file changes (implementation phase, NOT this plan):**
  `src/crispr_gnn/evaluation/robustness.py` (new; registry + replay + bootstrap +
  paired-difference; generalizes `scripts/compute_sprint6_bootstrap_ci.py` — named
  `robustness.py` rather than the working title `bootstrap.py` since it spans more
  than resampling), `scripts/run_sprint9_robustness.py`,
  `scripts/regenerate_f4_predictions.py`, a `--seed` override on the existing
  `scripts/run_sprint7f/8a/8b` runners (seed value only; no training-loop change),
  `configs/sweeps/sprint9_multiseed.yaml`, a runner-only
  `colab/sprint9_multiseed_runner.ipynb`, and
  `tests/test_sprint9_robustness.py`.
- **Risks:** treating row-bootstrap as valid; over-reading threshold-metric CIs in
  a 29-cluster / 9-negative-guide regime; BCa instability with few clusters;
  overlapping-CI fallacy; F4 reproduction failure; multi-seed misread as
  generalization uncertainty; cross-file row misalignment when pairing.
- **Acceptance criteria:** guide-level (not row-level) bootstrap; predeclared
  paired matrix and seed list; thresholds read from prior outputs (never
  recomputed from test); replayed metrics match source reports within tolerance;
  no model/threshold/feature/label/split change; every Sprint 8 number preserved
  as a historical output; claim boundaries stated. (Full list in §13.)

Sprint 9 will **not**: change labels/splits/thresholds/architectures; run new
architecture variants or hyperparameter search; tune from test metrics; add
RNA-FM/DNABERT-2 transfer; select a best seed or best rerun; use MCC/specificity to
override AUPRC; infer biological causality from gates/FiLM/attention/sequence
embeddings; or claim model equivalence (no prespecified TOST margin — §11).

## 2. Frozen Evaluation Contract (inherited verbatim)

- Label scheme `scheme_a` = `int(cleavage_freq > 1e-5)`; NaN `cleavage_freq`
  excluded from supervised labels.
- Split ID `sprint2_main_seed42`; guide-disjoint.
- Headline universe measured-only (train/validation/test); `experiment_id=18`
  excluded.
- Graph visibility `strict_inductive_primary`; train-only preprocessing.
- Checkpoint selection: validation AUPRC only. Threshold selection: validation
  max-F1 only. **Sprint 9 reads these frozen thresholds; it never reselects them.**
- No test-driven selection of architecture, threshold, feature, or hyperparameter.
- Primary metric **AUPRC**. Secondary threshold metrics: MCC, specificity/TNR,
  macro F1, AUROC, TN/FP/FN/TP.
- Authoritative non-GNN bar `xgboost_unweighted / F4`, test AUPRC `0.992522`,
  AUROC `0.938416`, MCC `0.345198`, TN/FP/FN/TP `38/131/21/1512`.

Sprint 9 may recompute metrics from saved predictions but must not alter the
original predictions, thresholds, labels, splits, or selected candidates.

### 2.1 Fixed test-set geometry (verified 2026-06-13)

Confirmed empirically from
`outputs/sprint8b/diagnostics/sequence_context_predictions.csv` and matching
`docs/literature/sprint9-deep-research.pdf`:

- Test = **1702 rows from 29 guide clusters** (key column `grna_target_id`).
- **1533 positives / 169 negatives**; positive prevalence `1533/1702 = 0.9007`.
- Negatives occur in **only 9 of 29 guides**; one guide (`9251`) holds
  **80/169 = 47.3%** of all negatives.
- Under guide-cluster bootstrap the dominant guide `9251` is omitted in
  `(28/29)^29 ≈ 36.1%` of replicates, so thresholded negative-class metrics
  (specificity/MCC) are inherently lumpy and **fragile by construction**.

This geometry is the central limitation: cluster-aware information is governed by
the **number of guides (29), not the row count (1702)**. Intervals are reported as
**finite-sample, guide-cluster-aware compatibility summaries**, not exact 95%
coverage guarantees (PDF §1, §3.2).

## 3. Models / Run IDs In Scope

### 3.1 Bootstrap & paired-difference (from saved per-row predictions — no retrain)

Predeclared registry, all keyed by `predeclared_run_id` within each sprint's
authoritative consolidated batch:

| Registry ID | Source predictions file | Authoritative batch |
| --- | --- | --- |
| `S7F_R1`, `S7F_R2`, `S7F_R3` | `outputs/sprint7f/diagnostics/target_context_encoder_predictions.csv` | `sprint7f_..._20260608_051125` |
| `S8A_R0`..`S8A_R4` | `outputs/sprint8a/diagnostics/target_context_interaction_predictions.csv` | `sprint8a_..._20260611_011416` |
| `S8B_R1`, `S8B_R2` | `outputs/sprint8b/diagnostics/sequence_context_predictions.csv` | `sprint8b_..._20260612_220002` |
| `XGB_F4` | regenerated in Slice 2 (Sprint 2 did not save per-row predictions) | Sprint 2 `xgboost_unweighted / F4` contract |

Carry-forward GCN reference rows (Graph A/B/C, Sprint 5/5B/6) have **no saved
per-row predictions in these files** and are therefore **out of Sprint 9 CI
scope**; they remain covered only by the Sprint 6 prototype. This keeps Sprint 9
bounded to the candidates and their direct references (handoff §1).

### 3.2 Thresholds (read, never recomputed)

Each model's frozen validation-selected threshold is read from the `threshold`
column of its comparison CSV, keyed by `predeclared_run_id`:
`outputs/sprint7f/target_context_encoder_comparison.csv`,
`outputs/sprint8a/target_context_interaction_comparison.csv`,
`outputs/sprint8b/sequence_context_comparison.csv`. The `XGB_F4` threshold is the
**regenerated** model's own `validation_max_f1` threshold (`0.604756`,
validation-only), carried in `outputs/sprint9/diagnostics/f4_predictions.csv`, so
the F4 bar is one self-consistent model (Slice 2 / 2026-06-13 Option C; historical
value `0.605734`).

### 3.3 Multi-seed retraining (GPU/CPU — predeclared configs only)

Approved scope (this conversation, 2026-06-13): **Core + S7F R2 + F4**.

| Multi-seed config | Runner | Device |
| --- | --- | --- |
| `S8A_R2_context_edge_film` | `scripts/run_sprint8a_target_context_interaction.py --run S8A_R2_context_edge_film` | Colab GPU |
| `S8B_R2_sequence_plus_context` | `scripts/run_sprint8b_sequence_context.py --run S8B_R2_sequence_plus_context` | Colab GPU |
| `S7F_R3_family_aware_experimental_emphasis` | `scripts/run_sprint7f_target_context_encoder.py --run S7F_R3...` | Colab GPU |
| `S7F_R2_family_aware_context_encoder` | `scripts/run_sprint7f_target_context_encoder.py --run S7F_R2...` | Colab GPU |
| `XGB_F4` (XGBoost training stochasticity) | `scripts/regenerate_f4_predictions.py --seed ...` | CPU |

Excluded from multi-seed (discipline, not budget): all closed ablation cells
(Graph A/B/C GCN, Sprint 5/5B, Sprint 6 losses, Sprint 7/7B/7D/7E mechanism runs)
per DECISIONS 2026-06-06 ("multi-seed is NOT applied retroactively to locked
ablation cells"); and the **rejected** Sprint 8 arms (`S8A_R0/R1/R3/R4`,
`S8B_R1`) — multi-seeding losers invites best-of optics (Schneider overtuning;
PDF §2.3 "do not best-seed select").

## 4. Bootstrap Method (PDF §2.1, §3.3)

- **Resampling unit = the guide cluster** (`grna_target_id`), never the row. Each
  replicate samples **29 guides with replacement** from the 29 test guides and
  includes **all rows** of each drawn guide (Field & Welsh 2007; Cameron & Miller
  2015). Row-level resampling is invalid (rows within a guide are correlated).
- **B = 5000** replicates; bootstrap RNG seed fixed (`12345`) for reproducibility.
  Floor of ≥2000 for 95% percentile CIs (Carpenter & Bithell 2000; Efron &
  Tibshirani 1993); 5000 for stable lumpy-tail quantiles at negligible CPU cost.
- **Interval method: percentile is PRIMARY** (2.5th/97.5th). **BCa is a
  sensitivity check only**, trusted only if the leave-one-guide jackknife
  acceleration is finite, smooth, and not dominated by 1–2 guides; if percentile
  and BCa disagree materially, report both and flag interval construction as
  unstable (PDF §3.3). **This supersedes the "prefer BCa" wording** in DECISIONS
  2026-06-06 and `CRISPR_GNN_PROJECT_PLAN.md` §12 — see §14.
- **AUPRC definition is fixed** to the same `average_precision_score` used in the
  source reports (`src/crispr_gnn/evaluation/metrics.py`); never silently switch to
  trapezoidal PR-AUC (Boyd 2013; Davis & Goadrich 2006).
- **Thresholded metrics** (specificity, MCC) apply the frozen validation threshold
  in every replicate; never retuned inside the bootstrap.
- **Degenerate replicates** are tracked, not silenced: AUPRC undefined if a
  replicate has no positives; specificity undefined if no negatives; MCC undefined
  if a confusion denominator collapses. Report per-metric **undefined/failure
  rates**.
- **AUPRC framing:** `0.9007` is the **no-skill baseline, not a floor** (an
  adversarial ranking can fall below it); always report AUPRC point + interval
  beside the baseline and inspect for pile-up near 1.0 / discreteness (PDF §3.1,
  §6).

### 4.1 Required bootstrap diagnostics (PDF §2.1 step 7, §3.2)

Per model: undefined-replicate rate per metric; unique guides per replicate;
negative-bearing guides per replicate; inclusion frequency of the dominant guide
`9251`; histogram/quantile shape flags (bounded/lumpy/multimodal/endpoint
pile-up); and a **leave-one-guide-out influence table** (largest LOGO metric
change, especially for negative-bearing guides). Negative-class CIs are reported
**with an explicit fragility warning** whenever they are dominated by few guides
or have high undefined rates.

## 5. Paired-Difference Bootstrap (PDF §2.2, §3.4)

For every replicate, draw **one** guide resample and evaluate **all** compared
models on exactly that resample; per metric compute `Δ_b = metric(A) − metric(B)`;
build the interval from the empirical `Δ_b` distribution. **Overlap of two marginal
CIs is never used** to judge a difference (Schenker & Gentleman 2001). If a
thresholded metric is undefined for either model in a replicate, `Δ_b` is undefined
for that replicate and the **undefined-Δ rate** is reported.

### 5.1 Predeclared comparison matrix

Primary (AUPRC; ranking question):

| # | A − B | Question |
| --- | --- | --- |
| P1 | `S8B_R2` − `S8A_R2` | Does sequence-context add over target-context interaction? |
| P2 | `S8B_R2` − `S7F_R3` | Does the 8B candidate beat the strongest carry-forward GNN? |
| P3 | `S8A_R2` − `S7F_R3` | Did 8A interaction beat its own base lineage? |
| P4 | `S8B_R2` − `XGB_F4` | Is XGBoost's lead over the 8B candidate robust? |
| P5 | `S8A_R2` − `XGB_F4` | Is XGBoost's lead over the 8A candidate robust? |
| P6 | `S7F_R3` − `XGB_F4` | Is XGBoost's lead over the strongest GNN robust? |

Secondary (operating point; MCC + specificity at frozen thresholds, fragility-
caveated):

| # | A − B | Question |
| --- | --- | --- |
| P7 | `S8B_R2` − `S7F_R2` | Operating-point change vs the rare-negative reference |
| P8 | `S8A_R2` − `S7F_R2` | Operating-point change vs the rare-negative reference |

All P1–P8 also report AUROC and macro F1 as secondary ranking/operating-point
context. No comparison is added after seeing results.

## 6. F4 Per-Row Prediction Recovery (Slice 2)

Sprint 2 saved **no** XGBoost per-row predictions and **no** `model.pt` — only
aggregate diagnostics. F4 paired comparison and an F4 CI therefore require
regenerating per-row test (and validation) scores under the **exact Sprint 2
contract**: Scheme A labels, `sprint2_main_seed42` split, measured-only val/test,
`experiment_id=18` excluded, F4 feature ladder, train-only imputation, predeclared
XGBoost seed/params.

- **Reproduction gate (reframed by the 2026-06-13 Option C decision).** XGBoost is
  not bit-reproducible across library versions, so the *blocking* gate is a faithful
  reproduction under the current pinned environment: **test geometry exact**
  (1702/29/169) **and test AUPRC within `2e-3`** of the historical `0.992522` (this
  still catches genuine pipeline/leakage breakage). The strict `±1e-4` historical
  match is kept as a **non-blocking diagnostic**. Outcome: AUPRC `0.992338`
  (Δ `1.84e-4`, version drift) — blocking gate **passed**, F4 accepted; F4 uses the
  regenerated model's own validation-selected threshold `0.604756`. If the blocking
  gate had failed, F4 paired comparison (P4–P6) and the F4 CI would be dropped and
  the failure reported — no tuning to force the number (Kapoor & Narayanan
  leakage/no-test-tuning discipline). This is replay, not modelling.

## 7. Multi-Seed Retraining (Slice 5; PDF §2.3, §3.5)

- **Predeclared seed set (fixed before running): `{42, 7, 13, 123, 2024}`** — 5
  seeds. Seed 42 is the canonical existing run; re-running it under the Sprint 9
  harness may not bit-reproduce the original number (documented harness drift, as
  `S8A_R0 ≠ S7F_R3` already showed) — that drift is itself reportable.
- **Configs:** §3.3 (S8A R2, S8B R2, S7F R3, S7F R2 on GPU; F4 on CPU). Same
  split, labels, threshold-selection protocol, model family; only the seed varies
  (no training-loop/architecture/threshold change). Implemented via a `--seed`
  override on the existing runners.
- **Report every seed** for AUPRC/MCC/specificity/AUROC; summarise descriptively
  (mean, std, min, max, all values). **No best-seed selection; no hidden seeds.**
- **Framing:** training-stochasticity sensitivity **conditional on the fixed
  split**. This is **not** a generalization CI and **not** a substitute for the
  guide-bootstrap (which captures test-sampling uncertainty); the two spreads are
  reported **side by side** (Bengio & Grandvalet 2004: no unbiased estimator of
  resampling variance → descriptive only).

## 8. No seed count or B is literature-derived — predeclaration basis

No source prescribes a seed count or B value (PDF protocol refs Bethard 2022;
Bengio & Grandvalet 2004; Schneider 2025 fix *how*, not *how many*). `N=5` and
`B=5000` are predeclared pragmatic budgets, labelled as such — not literature-
derived constants. The contestable methodological choices (guide-cluster bootstrap,
percentile-over-BCa, paired-not-overlap, compatibility language) **are** cited
(§4, §5, §11).

## 9. Output Contract

Consolidated:

```text
outputs/sprint9/robustness_report.md
outputs/sprint9/robustness_bootstrap_cis.csv
outputs/sprint9/robustness_paired_differences.csv
outputs/sprint9/robustness_model_seed_summary.csv
```

Diagnostics:

```text
outputs/sprint9/diagnostics/metric_replay_check.csv
outputs/sprint9/diagnostics/f4_reproduction_check.csv
outputs/sprint9/diagnostics/per_guide_metric_contributions.csv
outputs/sprint9/diagnostics/bootstrap_replicate_diagnostics.csv
outputs/sprint9/diagnostics/leave_one_guide_influence.csv
```

Figures:

```text
outputs/sprint9/figures/robustness_auprc_cis.png
outputs/sprint9/figures/paired_difference_distributions.png
outputs/sprint9/figures/per_seed_metric_variance.png
outputs/sprint9/figures/bootstrap_distribution_diagnostics.png
```

Multi-seed per-run directories follow the existing sprint runner layout
(`resolved_config.yaml`, `runtime.json`, `training_history.csv`, `metrics.csv`,
`model.pt` untracked). No committed `model.pt`/`.DS_Store`; no stale smoke/partial
runs in the consolidated manifest.

## 10. Tests & Validation

`tests/test_sprint9_robustness.py` must cover (handoff §7 + PDF §2):

- **Guide bootstrap resamples guides, not rows:** a drawn guide contributes all
  its rows; a row-level resample path is absent/rejected.
- **Paired bootstrap uses a common resample:** both models scored on the identical
  drawn guide set per replicate.
- **Metric replay matches source:** full non-resampled test metrics for every
  registry model match the source comparison CSVs within tolerance.
- **Thresholds read from prior outputs**, not recomputed from test.
- **F4 reproduction gate:** regenerated F4 test AUPRC within `±1e-4` of
  `0.992522`; on failure, F4 pairs/CI are excluded with a recorded reason.
- **Degenerate-replicate handling:** undefined specificity/MCC (no negatives /
  collapsed denominator) are counted, not crashed; undefined rates surfaced.
- **Output contract complete** (monkeypatched bootstrap/training where needed).

```bash
uv run pytest tests/test_sprint9_robustness.py -q
uv run ruff check src/crispr_gnn/evaluation/robustness.py scripts/run_sprint9_robustness.py scripts/regenerate_f4_predictions.py tests/test_sprint9_robustness.py
git diff --check
```

Bootstrap/replay slices run locally (CPU). Multi-seed full training runs only on
Colab (runner-only), as in Sprints 7F/8A/8B.

## 11. Claim Boundaries (PDF §4, §6)

Licensed claims:

- **Paired Δ-CI excludes zero** → "directional evidence of a model difference for
  guide-level resampling of this fixed held-out split, conditional on the trained
  models and frozen threshold protocol" — not automatically generalized to future
  retrainings, new guide populations, or different splits.
- **Paired Δ-CI includes zero** → "the fixed-split guide-bootstrap did not provide
  clear evidence of superiority; the data are compatible with no difference and
  with effect sizes inside the interval." **May not** claim the models are
  equivalent/identical/non-inferior — equivalence needs a prespecified margin +
  TOST (Schuirmann 1987; Lakens 2017), which Sprint 9 **defers** (no defensible
  AUPRC margin exists; setting one from observed variance is circular).

Red-flag statements to avoid (verbatim discipline, PDF §6): "AUPRC has a floor of
0.9007" (→ "no-skill PR baseline is 0.9007"); "the 95% CI has exact 95% coverage"
(→ "finite-sample guide-cluster compatibility interval"); "1702 rows make the CI
reliable" (must note the 29-guide / 9-negative-guide bottleneck); "overlapping CIs
show no difference" / "non-overlap proves a difference" (→ paired Δ-CI); "the
models are equivalent" with Δ-CI∋0; "multi-seed variance estimates generalization
uncertainty"; "BCa is automatically better here". Also retained: no biological
causality from gates/FiLM/attention/sequence embeddings; MCC/specificity never
override AUPRC ranking.

## 12. Implementation Slices

Incremental, test-gated; frozen contract preserved throughout; no headline claim
until Slice 6.

### Slice 0 — Planning freeze (this document)

Freeze: registry (§3.1), thresholds-read rule (§3.2), multi-seed scope/seeds
(§3.3, §7), bootstrap method (§4: guide-cluster, B=5000, percentile-primary/
BCa-sensitivity), paired matrix (§5.1), F4 reproduction gate (§6), output contract
(§9), tests (§10), claim boundaries (§11), and the BCa supersession (§14). No code.

Exit: plan frozen; no code changed.

Done (2026-06-13): plan frozen. Consistency re-audit against the repo — **PASS, no
drift**: all four runners (`run_sprint7f/8a/8b`, `compute_sprint6_bootstrap_ci.py`)
exist; the F4 validation-threshold artifact
(`outputs/sprint2/diagnostics/xgboost_unweighted_fixed_threshold_metrics.csv`)
exists; every registry `predeclared_run_id` resolves in the source prediction files
with both `val` and `test` splits (`S7F_R1/R2/R3`, `S8A_R0..R4`, `S8B_R1/R2`); the
fixed test geometry (1702 rows / 29 guides / 169 negatives in 9 guides / guide
`9251` = 47.3%) is confirmed empirically. No `src/`, `configs/`, `scripts/`,
`colab/`, or `tests/` file changed in Slice 0.

### Slice 1 — Prediction registry & metric replay

Build `src/crispr_gnn/evaluation/robustness.py` + a registry loader for the §3.1
GNN predictions; replay full-split metrics using `evaluation/metrics.py`.

Exit: `metric_replay_check.csv` shows replayed full-test metrics match the source
comparison CSVs within tolerance for `S7F_R1..R3`, `S8A_R0..R4`, `S8B_R1/R2`.

Done (2026-06-13): added `src/crispr_gnn/evaluation/robustness.py` (`RegistryEntry`,
`GNN_REGISTRY` per §3.1, `ModelScores`, `load_model_scores`/`load_registry`,
`replay_split_metrics`, `replay_check_records`) reusing the canonical
`binary_classification_metrics`; thresholds are read from each run's comparison CSV
(`threshold`, validation-selected) and never recomputed from test; the guide key
`grna_target_id` is loaded but not resampled (Slice 3). Added
`scripts/run_sprint9_robustness.py` (Slice 1 replay stage) writing
`outputs/sprint9/diagnostics/metric_replay_check.csv`, and
`tests/test_sprint9_robustness.py` (registry completeness/single-batch, threshold
read + not-recomputed-from-test, guide-cluster geometry 29/169/9/80, replay match).
Validation: replay reproduces all 10 GNN models' test AUPRC/AUROC/MCC/specificity/
macro-F1/F1/sensitivity and TN/FP/FN/TP with max abs diff `1.11e-16` (atol `1e-9`,
integer cells exact); `uv run pytest tests/test_sprint9_robustness.py -q` (5 passed);
ruff + `git diff --check` clean. `XGB_F4` deferred to Slice 2. Module named
`robustness.py` (not the §1 working title `bootstrap.py`); §1/§10 updated. No
frozen artifact, threshold, label, split, or model changed; no DECISIONS/tech-debt
entry needed (additive replay foundation, no methodological or result change).

### Slice 2 — F4 per-row prediction recovery

`scripts/regenerate_f4_predictions.py` retrains XGBoost under the Sprint 2 F4
contract; reproduce test AUPRC `0.992522` (`±1e-4`); save per-row val/test scores +
the validation-selected threshold; add `XGB_F4` to the registry.

Exit: `f4_reproduction_check.csv` passes the gate (or documents failure and removes
F4 from P4–P6 and the F4 CI). No hyperparameter or threshold change.

Done (2026-06-13): added `scripts/regenerate_f4_predictions.py` (reuses
`run_xgboost_baselines` for F4 unweighted only, joins `grna_target_id`, leakage
audit, no frozen Sprint 2 artifact touched) and `load_f4_model_scores` /
`load_full_registry` in `robustness.py`; F4 tests added. **The strict `±1e-4`
historical gate was missed (test AUPRC `0.992338`, Δ `1.84e-4`) due to XGBoost
version drift** (env `3.2.0` vs the earlier Sprint 2 build; geometry 1702/29/169
exact). Per the **2026-06-13 "Option C" decision** (DECISIONS.md + tech-debt.md),
the blocking gate is reframed to a version-drift-tolerant reproduction (geometry
exact + AUPRC within `2e-3`), the strict `±1e-4` match is reported as a non-blocking
diagnostic, and F4 uses the regenerated model's **own** `validation_max_f1`
threshold `0.604756` (validation-only). `f4_predictions.csv` written;
`load_full_registry()` now returns 11 models; F4 test guides == GNN test guides
(P4-P6 alignment). Validation: regeneration blocking gate PASS; `uv run pytest
tests/test_sprint9_robustness.py -q` (8 passed); ruff + `git diff --check` clean.
Slice 3/4 must cite F4 test AUPRC `0.992338` with the historical-`0.992522`
version-drift footnote.

### Slice 3 — Guide-cluster bootstrap CIs

Implement the §4 guide-cluster bootstrap (percentile primary, BCa sensitivity) over
all registry models; write CI tables for AUPRC/AUROC/MCC/specificity/macro F1 plus
the §4.1 fragility diagnostics (undefined rates, guides/replicate, negative-bearing
guides/replicate, dominant-guide inclusion, LOGO influence).

Exit: `robustness_bootstrap_cis.csv`, `bootstrap_replicate_diagnostics.csv`,
`leave_one_guide_influence.csv`, `robustness_auprc_cis.png`,
`bootstrap_distribution_diagnostics.png` written.

### Slice 4 — Paired-difference bootstrap

Implement §5 paired bootstrap on common guide resamples for the §5.1 matrix
(P1–P8), with undefined-Δ rates.

Exit: `robustness_paired_differences.csv`,
`paired_difference_distributions.png` written for all predeclared pairs.

### Slice 5 — Multi-seed fixed-split retraining

Add the `--seed` override + `configs/sweeps/sprint9_multiseed.yaml` + runner-only
`colab/sprint9_multiseed_runner.ipynb`; run the §3.3 configs over seeds
`{42,7,13,123,2024}` (GNN on Colab GPU; F4 on CPU); consolidate every seed.

Exit: `robustness_model_seed_summary.csv` + `per_seed_metric_variance.png` report
every seed (mean/std/min/max/all values); no best-seed selection.

### Slice 6 — Consolidated robustness report & docs

Write `robustness_report.md` separating ranking quality (AUPRC/AUROC), operating
point (MCC/specificity/TN/FP/FN/TP), and the three uncertainty summaries; state a
clear claim status (**robust improvement / inconclusive / not robust**) under §11
boundaries; update source-of-truth docs (§16) including the BCa supersession entry.

Exit: report complete; plan moved to `docs/exec-plans/completed/` after validated
outputs.

## 13. Acceptance Criteria

Sprint 9 plan/run is acceptable only if it: keeps the Sprint 2/3/8 frozen contract;
treats AUPRC as primary; uses **guide-level** (not row-level) bootstrap; predeclares
the paired matrix (§5.1) and the seed list (§7); introduces no model/threshold/
feature/hyperparameter tuning; reads thresholds from prior outputs; preserves all
Sprint 8 numbers as historical outputs; reproduces F4 (or documents failure and
drops F4 comparisons); reports fragility/undefined diagnostics; and uses
finite-sample compatibility language with the §11 claim boundaries. Local tests +
lint pass; Colab is runner-only; no committed `model.pt`.

## 14. BCa Supersession

The bootstrap interval method is changed for Sprint 9 from the earlier
"prefer BCa" guidance to **percentile-primary, BCa-sensitivity-only**, on the basis
of `docs/literature/sprint9-deep-research.pdf` §3.3: with only 29 guide clusters,
bounded/nonsmooth metrics, and a few highly influential negative-bearing guides, the
leave-one-guide jackknife acceleration is unstable and can be dominated by one or
two guides. This supersedes (interval-method wording only) DECISIONS 2026-06-06
("BCa preferred") and `CRISPR_GNN_PROJECT_PLAN.md` §12 ("Prefer BCa intervals"); a
DECISIONS entry recording this is added at Slice 6. All other robustness
methodology in those entries is unchanged.

## 15. Frozen Specification (Slice 0)

Slice 0 planning freeze (2026-06-13). All values below are pinned; Slices 1–6
implement against this section with zero remaining design decision. No `src/`,
`configs/`, `scripts/`, `colab/`, or `tests/` file changed in Slice 0.

- **Resampling unit:** guide cluster (`grna_target_id`); 29 guides; sample 29 with
  replacement; all rows of each drawn guide.
- **B:** 5000; bootstrap RNG seed `12345`.
- **Interval:** percentile primary (2.5/97.5); BCa sensitivity-only with LOGO
  jackknife trust check.
- **Metrics:** AUPRC primary (`average_precision_score`); AUROC, MCC, specificity,
  macro F1 secondary at frozen validation thresholds (read from comparison CSVs).
- **Registry:** `S7F_R1/R2/R3`, `S8A_R0..R4`, `S8B_R1/R2`, `XGB_F4` (Slice 2).
- **Paired matrix:** P1–P8 (§5.1); no additions post-result.
- **F4 gate (Option C, 2026-06-13):** blocking = geometry exact + AUPRC within
  `2e-3` of `0.992522`; strict `±1e-4` is a non-blocking version-drift diagnostic.
  Accepted at AUPRC `0.992338`; F4 threshold = regenerated `0.604756`.
- **Multi-seed:** seeds `{42,7,13,123,2024}`; configs `S8A_R2`, `S8B_R2`, `S7F_R3`,
  `S7F_R2` (GPU), `XGB_F4` (CPU); report all seeds; no best-seed.
- **TOST:** deferred; compatibility language only.
- **Claim boundaries:** §11 (PDF §4, §6).
- **Output contract & tests:** §9, §10.

Slice 0 exit: Sprint 9 plan frozen; Slices 1–6 may proceed against §15; no code
changed.

## 16. Required Docs Updates On Completion (Slice 6, not now)

- `docs/DECISIONS.md`: record the Sprint 9 outcome (robust / inconclusive / not
  robust per pair), the BCa→percentile supersession (§14), the F4 reproduction
  result, the predeclared seed set, and B=5000. Cite
  `docs/literature/sprint9-deep-research.pdf` and its underlying references.
- `docs/PROJECT_CONTEXT.md`: update the Sprint 9 status line.
- `README.md` / `CRISPR_GNN_PROJECT_PLAN.md`: mark Sprint 9 complete; reconcile the
  §12 "prefer BCa" wording with §14.
- `docs/exec-plans/tech-debt.md`: any residual Sprint 9 caveats.

## 17. Literature Anchors

Primary methodology source: `docs/literature/sprint9-deep-research.pdf` (tailored to
this fixed split). Underlying references it carries: Efron 1979; Efron & Tibshirani
1993; Davison & Hinkley 1997; Carpenter & Bithell 2000 (B for percentile/BCa CIs);
Field & Welsh 2007 and Cameron & Miller 2015 (cluster bootstrap, few-clusters
caveat); Boyd et al. 2013, Davis & Goadrich 2006, Saito & Rehmsmeier 2015, Williams
2021 (AUPRC/PR under imbalance); Schenker & Gentleman 2001 (overlapping-CI fallacy);
Bengio & Grandvalet 2004 (no unbiased CV-variance estimator); Bethard 2022 (random
seeds); Schuirmann 1987, Lakens 2017 (TOST/equivalence). Corpus support: Gao 2020,
Guan 2024 (AUPRC-primary at imbalance, complete-test-set evaluation), Kapoor &
Narayanan 2023 (leakage/no-test-tuning), Schneider 2025 (overtuning/no best-seed),
Dwivedi 2022 (parameter-budget/active-parameter caveat). All cited as
methodological adaptation, not reproduction.
