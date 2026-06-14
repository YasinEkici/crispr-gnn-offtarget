# Sprint 9 Robustness Report — Uncertainty & Variance Quantification

Interpretation-only consolidation of Sprint 9 Slices 1–5. No model, label, split,
threshold, feature, or architecture was changed or selected here; every number below
is read from the committed Sprint 9 outputs. Thresholds are the frozen
validation-selected values (never recomputed from test); all bootstrap resampling is
by **guide cluster** (`grna_target_id`), never by row.

**Governing question.** *Are the Sprint 8A/8B candidate improvements robust enough to
support a stronger claim, or are they within single-seed / guide-composition
uncertainty?*

**Headline answer.** On the primary metric (**AUPRC**) the Sprint 8 improvements are
**not robust**: no predeclared paired difference excludes zero, the single-seed
guide-cluster intervals are wide and mutually overlapping, and the seed-to-seed spread
is larger than the candidate gains. XGBoost **F4** remains the highest-AUPRC bar with
the tightest spread. The data are **compatible with no AUPRC difference** among the GNN
candidates and F4 — this is *not* a claim of equivalence (a TOST margin was deferred,
§11). A genuine, threshold-dependent operating-point effect exists (the GNNs recover
more rare negatives than F4 at their thresholds) but it is itself seed-fragile and, per
§11, does not override the AUPRC ranking.

## 1. Frozen Evaluation Contract (recap)

- Label scheme A (`cleavage_freq > 1e-5`); split `sprint2_main_seed42`; guide-disjoint;
  **measured-only** headline; `experiment_id = 18` excluded; validation-only checkpoint
  and threshold selection; **AUPRC primary**; XGBoost **F4** is the non-GNN bar.
- **Test geometry (verified):** 1702 rows / **29 guides** / 169 negatives concentrated in
  **9 negative-bearing guides**; guide `9251` alone carries 80 negatives (47.3%).
  Positive prevalence 0.900705 — the **no-skill PR baseline is 0.9007** (not a "floor").

## 2. Method (recap) and the BCa supersession

Three complementary uncertainty lenses, all on the fixed split:

1. **Guide-cluster bootstrap CIs** (single trained model per config): resample the 29
   guides with replacement (a drawn guide contributes all its rows), `B = 5000`,
   RNG seed `12345`, **percentile interval primary** (2.5/97.5).
2. **Paired-difference bootstrap**: one shared guide resample applied to *both* models
   per replicate; report `Δ = metric(A) − metric(B)` and whether its interval excludes
   zero. Marginal-CI overlap is **never** used to judge a difference (Schenker &
   Gentleman 2001).
3. **Multi-seed fixed-split retraining**: seeds `{42, 7, 13, 123, 2024}`; report every
   seed and mean/std/min/max; **no best-seed selection** (Bengio & Grandvalet 2004 —
   no unbiased variance estimator; descriptive sensitivity only).

**BCa supersession (§14).** Sprint 9 uses **percentile-primary, BCa-sensitivity-only**.
This is borne out empirically: with 29 clusters the leave-one-guide jackknife
acceleration is untrusted for nearly every metric (only `S8B_R2` AUPRC/AUROC passed the
trust gate; all other model/metric BCa intervals are flagged untrusted). This wording
supersedes the earlier "prefer BCa" guidance (DECISIONS 2026-06-06;
`CRISPR_GNN_PROJECT_PLAN.md` §12); all other robustness methodology is unchanged.

**Structural fragility (bootstrap diagnostics).** Across replicates the 29-guide draw
yields on average **18.5 unique guides** and only **5.74 negative-bearing guides**;
dominant negative guide `9251` appears in **63.7%** of replicates. The intervals are
wide because the *effective* negative-class sample is a handful of guides — that is the
honest point of the diagnostic, not a defect to be smoothed away.

**Replay integrity.** All 110 replayed metric checks across the 10 GNN registry models
match their source comparison CSVs within tolerance (`metric_replay_check.csv`).

## 3. Section A — Ranking quality (AUPRC primary, AUROC)

### A.1 Single-seed guide-cluster CIs (B = 5000, percentile)

AUPRC — point [95% finite-sample guide-cluster compatibility interval]:

| model | AUPRC | 95% interval | BCa trusted |
| --- | --- | --- | --- |
| XGB_F4 | 0.992338 | [0.950179, 0.999336] | no |
| S8B_R2 | 0.986020 | [0.929981, 0.998966] | yes |
| S7F_R3 | 0.984945 | [0.923532, 0.999913] | no |
| S8A_R2 | 0.982757 | [0.910478, 0.999892] | no |
| S7F_R2 | 0.982062 | [0.914690, 0.999242] | no |

AUROC: F4 0.9364 [0.8334, 0.9875]; S7F_R3 0.9266 [0.7765, 0.9977]; S8A_R2 0.9106
[0.7446, 0.9976]; S7F_R2 0.9066 [0.7377, 0.9877]; S8B_R2 0.9035 [0.7429, 0.9819].

Every AUPRC interval spans ≈0.05–0.09 and **all five overlap heavily**; F4's lower bound
(0.950) sits below every GNN point estimate and vice-versa. (Overlap is reported as
context only — the difference question is settled by the paired bootstrap in A.3, not by
overlap.) The no-skill PR baseline (0.9007) lies at or below the lower bound of most
intervals, so all models clear no-skill, but the headroom is small and uncertain.

### A.2 Multi-seed AUPRC (5 seeds; mean ± std [min, max])

| model | mean | std | min | max |
| --- | --- | --- | --- | --- |
| XGB_F4 | 0.990649 | 0.001944 | 0.988110 | 0.992557 |
| S8B_R2 | 0.978963 | 0.011322 | 0.958963 | 0.986479 |
| S8A_R2 | 0.975538 | 0.012187 | 0.954756 | 0.984490 |
| S7F_R2 | 0.974810 | 0.003802 | 0.969181 | 0.979555 |
| S7F_R3 | 0.968681 | 0.004643 | 0.963136 | 0.974476 |

The GNN seed-std (0.004–0.012) is **larger than the entire Sprint 8 candidate gain**
(the S8B−S8A single-seed AUPRC gain was +0.0033). F4 has both the highest mean and the
smallest spread. (These are fresh retrains in a separate Colab environment; the seed-42
reruns differ slightly from the frozen single-seed headline numbers and do **not**
replace them. Seed spread is training-stochasticity sensitivity, **not** a
generalization confidence interval.)

### A.3 Paired-difference AUPRC — `Δ = A − B` [95% interval]

| pair | A − B | Δ | 95% interval | excludes 0? | P(Δ>0) |
| --- | --- | --- | --- | --- | --- |
| P1 | S8B_R2 − S8A_R2 | +0.00326 | [−0.01484, +0.03706] | no | 0.615 |
| P2 | S8B_R2 − S7F_R3 | +0.00107 | [−0.02099, +0.02652] | no | 0.509 |
| P3 | S8A_R2 − S7F_R3 | −0.00219 | [−0.01932, +0.00124] | no | 0.157 |
| P4 | S8B_R2 − XGB_F4 | −0.00632 | [−0.03124, +0.00117] | no | 0.076 |
| P5 | S8A_R2 − XGB_F4 | −0.00958 | [−0.05788, +0.00541] | no | 0.253 |
| P6 | S7F_R3 − XGB_F4 | −0.00739 | [−0.04285, +0.00799] | no | 0.345 |
| P7 | S8B_R2 − S7F_R2 | +0.00396 | [−0.01800, +0.03707] | no | 0.634 |
| P8 | S8A_R2 − S7F_R2 | +0.00070 | [−0.01605, +0.00776] | no | 0.615 |

**No AUPRC paired difference excludes zero.** This includes the headline Sprint 8B
gain over Sprint 8A (P1) and F4's lead over each GNN candidate (P4–P6).

## 4. Section B — Operating point (MCC / specificity / macro-F1 / confusion)

These use each model's **frozen validation-selected threshold**; they describe behaviour
at a chosen operating point and, per §11, never override the AUPRC ranking.

### B.1 Single-seed CIs (point [95% interval])

| model | MCC | specificity | macro-F1 |
| --- | --- | --- | --- |
| XGB_F4 | 0.3511 [0.079, 0.699] | 0.2367 [0.047, 0.678] | 0.6485 [0.509, 0.844] |
| S8B_R2 | 0.5673 [0.312, 0.764] | 0.8639 [0.575, 0.972] | 0.7601 [0.607, 0.879] |
| S7F_R3 | 0.5681 [0.289, 0.914] | 0.4970 [0.214, 1.000] | 0.7772 [0.618, 0.956] |
| S8A_R2 | 0.5637 [0.308, 0.887] | 0.5207 [0.266, 1.000] | 0.7780 [0.638, 0.943] |
| S7F_R2 | 0.6035 [0.426, 0.883] | 0.6509 [0.496, 0.900] | 0.8017 [0.712, 0.940] |

At their thresholds the GNNs recover far more rare negatives than F4 (specificity
0.50–0.86 vs F4's 0.24; MCC 0.56–0.60 vs 0.35).

### B.2 Paired-difference operating point — intervals that **exclude zero**

| pair | metric | Δ | 95% interval |
| --- | --- | --- | --- |
| P5 (S8A_R2 − F4) | MCC | +0.2125 | [+0.0306, +0.4925] |
| P6 (S7F_R3 − F4) | MCC | +0.2170 | [+0.0488, +0.5219] |
| P4 (S8B_R2 − F4) | specificity | +0.6272 | [+0.1412, +0.8271] |
| P5 (S8A_R2 − F4) | specificity | +0.2840 | [+0.0828, +0.6549] |
| P6 (S7F_R3 − F4) | specificity | +0.2604 | [+0.0662, +0.7000] |
| P5 (S8A_R2 − F4) | macro-F1 | +0.1295 | [+0.0250, +0.2709] |
| P6 (S7F_R3 − F4) | macro-F1 | +0.1287 | [+0.0324, +0.2847] |

So *conditional on these specific trained models and their frozen thresholds*, the GNNs
provide directional evidence of better rare-negative recovery than F4. GNN-vs-GNN
operating-point differences (P1–P3, P7–P8) all include zero.

### B.3 But the operating point is seed-fragile

Multi-seed retraining shows the operating point swings widely across seeds:

| model | MCC mean ± std [min, max] | specificity mean ± std [min, max] |
| --- | --- | --- |
| XGB_F4 | 0.360 ± 0.033 [0.335, 0.415] | 0.234 ± 0.050 [0.189, 0.314] |
| S8B_R2 | 0.464 ± 0.165 [0.175, 0.569] | 0.466 ± 0.224 [0.343, 0.864] |
| S7F_R2 | 0.463 ± 0.134 [0.319, 0.594] | 0.402 ± 0.153 [0.266, 0.627] |
| S7F_R3 | 0.430 ± 0.102 [0.308, 0.571] | 0.381 ± 0.074 [0.260, 0.456] |
| S8A_R2 | 0.385 ± 0.161 [0.206, 0.563] | 0.289 ± 0.171 [0.053, 0.479] |

The GNN operating-point std (MCC 0.10–0.17; specificity up to 0.22) dwarfs F4's
(0.03/0.05), and at a bad seed the gap can vanish (e.g. S8A_R2 specificity drops to
0.053, **below** F4's mean 0.234; S8A_R2 MCC drops to 0.206). The operating-point
advantage is real for the specific frozen models but is **not stable across
retrainings** and is sensitive to the few negative-bearing guides (notably `9251`).

## 5. Section C — Synthesis of the three lenses

- **Single-seed guide-cluster CIs:** wide and overlapping on AUPRC (≈±0.05); the
  29-guide / ~5.7-negative-guide structure caps achievable precision regardless of the
  1702 rows.
- **Paired bootstrap:** removes the overlapping-CI ambiguity by sharing the guide draw —
  and still finds **no AUPRC difference excludes zero**.
- **Multi-seed:** the candidate AUPRC gains (~0.003) are smaller than the per-config
  seed std (0.004–0.012); operating-point metrics are even more seed-volatile.

All three agree: on AUPRC the candidates are inside single-seed / guide-composition /
training-stochasticity uncertainty.

## 6. Section D — Per-pair claim verdict (AUPRC, primary)

| pair | comparison | verdict (under §11) |
| --- | --- | --- |
| P1 | S8B_R2 vs S8A_R2 | inconclusive — compatible with no difference |
| P2 | S8B_R2 vs S7F_R3 | inconclusive — compatible with no difference |
| P3 | S8A_R2 vs S7F_R3 | inconclusive — compatible with no difference |
| P4 | S8B_R2 vs XGB_F4 | inconclusive — compatible with no difference |
| P5 | S8A_R2 vs XGB_F4 | inconclusive — compatible with no difference |
| P6 | S7F_R3 vs XGB_F4 | inconclusive — compatible with no difference |
| P7 | S8B_R2 vs S7F_R2 | inconclusive — compatible with no difference |
| P8 | S8A_R2 vs S7F_R2 | inconclusive — compatible with no difference |

"Inconclusive / compatible with no difference" means the fixed-split guide-bootstrap did
not provide clear evidence of superiority and the data are compatible with effect sizes
inside the interval. It does **not** mean the models are equivalent — equivalence needs a
prespecified margin + TOST, which Sprint 9 defers (no defensible AUPRC margin exists;
setting one from observed variance would be circular).

## 7. Section E — Overall verdict

- **No robust AUPRC improvement** was demonstrated by any Sprint 8 candidate over its
  lineage or over XGBoost F4. F4 remains the highest-AUPRC bar with the smallest seed
  spread.
- The Sprint 8 candidates remain **validation-selected mechanism results**, not
  statistical-superiority claims — exactly the boundary Sprint 8 reserved for Sprint 9.
- A **threshold-dependent operating-point effect** exists: at their frozen thresholds the
  GNNs recover more rare negatives than F4 (P4–P6 exclude zero on specificity; P5/P6 on
  MCC and macro-F1). This is directional evidence *conditional on these trained models
  and thresholds*; it is seed-fragile, negative-class-fragile (guide `9251`), and per §11
  does **not** override the AUPRC ranking.
- This is a **statistical-power outcome**, not a pipeline failure: ~0.003 candidate
  effects against ≈±0.05 guide-cluster uncertainty on a 29-guide ceiling benchmark are
  not separable. A robust AUPRC win would require more guides / a larger held-out guide
  population, not further tuning on this split.

## 8. Claim-boundary discipline (restated)

Avoided throughout: "AUPRC floor 0.9007" (→ no-skill PR baseline 0.9007); "exact 95%
coverage" (→ finite-sample guide-cluster compatibility interval); "1702 rows make the CI
reliable" (the 29-guide / 9-negative-guide bottleneck governs); "overlapping CIs show no
difference" / "non-overlap proves a difference" (the paired Δ-CI is authoritative); "the
models are equivalent" with Δ-CI ∋ 0; "multi-seed variance estimates generalization
uncertainty"; "BCa is automatically better here". No biological causality is attributed
to gates / FiLM / attention / sequence embeddings. MCC/specificity never override AUPRC.

## 9. Artifact index

Consolidated:
- `outputs/sprint9/robustness_report.md` (this file)
- `outputs/sprint9/robustness_bootstrap_cis.csv`
- `outputs/sprint9/robustness_paired_differences.csv`
- `outputs/sprint9/robustness_model_seed_summary.csv`

Diagnostics:
- `outputs/sprint9/diagnostics/metric_replay_check.csv`
- `outputs/sprint9/diagnostics/f4_reproduction_check.csv`
- `outputs/sprint9/diagnostics/bootstrap_replicate_diagnostics.csv`
- `outputs/sprint9/diagnostics/leave_one_guide_influence.csv`

Figures:
- `outputs/sprint9/figures/robustness_auprc_cis.png`
- `outputs/sprint9/figures/paired_difference_distributions.png`
- `outputs/sprint9/figures/per_seed_metric_variance.png`
- `outputs/sprint9/figures/bootstrap_distribution_diagnostics.png`

**Note on the §9 output contract.** The plan's §9 listed
`diagnostics/per_guide_metric_contributions.csv`. Per-guide influence is instead captured
by `leave_one_guide_influence.csv` (leave-one-guide deltas per model/metric), which is the
quantity the report relies on; the original filename is superseded to avoid generating a
second, redundant test-derived table. Recorded in `docs/exec-plans/tech-debt.md`.

## 10. F4 reproduction footnote

The F4 bar was regenerated under XGBoost `>= 3.2.0` (Option C, DECISIONS 2026-06-13):
test AUPRC `0.992338` vs the historical `0.992522` (Δ `1.84e-4`; threshold `0.604756`),
a benign library-version drift ≈50–100× below the guide-cluster CI width and unable to
change any Sprint 9 conclusion. The regenerated F4 is **not** presented as bit-identical
to the Sprint 2 publication number.
