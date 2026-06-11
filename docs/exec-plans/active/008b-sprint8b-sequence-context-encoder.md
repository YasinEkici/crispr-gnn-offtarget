# Execution Plan: Sprint 8B Sequence-Context Encoder (CRISPR-Net-adapted)

> Status: FROZEN — Slice 0 planning freeze complete (2026-06-11). Concrete design
> decisions are pinned in §15 Frozen Specification; Slices 1–7 implement against it
> with no remaining design decision. Companion to
> `../completed/008-sprint8a-target-context-interaction.md`. This is the sequence-context
> slice of Sprint 8's model-improvement scope. Robustness remains Sprint 9.

## 1. Goal

Sprint 7D/7E/7F localized the Graph C GATv2 gain to **target-context**
representation, and Sprint 8A pushes target-context modelling and context-edge
interaction. Sprint 8B asks a deliberately separate question:

> Does adding a **sequence-context encoder over the aligned sgRNA/target
> sequence pair**, adapted (not reproduced) from CRISPR-Net-style architectures,
> add value over the Sprint 8A target-context model under the frozen contract —
> when the encoder is re-implemented in `src/` and trained from scratch on our
> locked guide-disjoint split?

Sprint 8B is intentionally **after** Sprint 8A and **separately scoped** so that a
sequence encoder never confounds whether 8A's gains came from target-context,
sequence encoding, or their interaction (the handoff's explicit warning).

### AGENTS.md workflow summary

- **Task goal:** add one predeclared, from-scratch sequence-context encoder
  branch (CRISPR-Net-adapted) and test whether it improves the Sprint 8A
  best model under the frozen contract; report all runs.
- **Expected file changes (implementation phase, NOT this plan):** a sequence
  encoder module under `src/crispr_gnn/models/` (e.g.
  `sequence_context_encoder.py`), wiring in `gat.py`/`training/gcn.py`, a
  `configs/sweeps/sprint8b_sequence_context.yaml`, a
  `scripts/run_sprint8b_sequence_context.py`, a runner-only Colab notebook, and
  `tests/test_sprint8b_sequence_context.py`.
- **Risks:** leakage from any externally-pretrained CRISPR/genomic weights;
  confounding 8A's context gains; over-claiming reproduction of CRISPR-Net;
  larger scope/compute; single-seed variance.
- **Acceptance criteria:** from-scratch training on the locked split only,
  minimal predeclared run set, validation-AUPRC selection, no reproduction
  claims, leakage controls documented, output contract complete, tests + lint
  pass, Colab runner-only.

## 2. Frozen Evaluation Contract

Identical to Sprint 8A §2 (scheme_a; `sprint2_main_seed42`; guide-disjoint;
measured-only; `experiment_id=18` excluded; train-only preprocessing;
validation-only checkpoint + threshold; no test-driven selection; AUPRC primary;
XGBoost F4 bar `0.992522`; seed 42; Colab runner-only). Sprint 8B does **not**
change the split, labels, loss, or threshold policy.

## 3. Hard Rules (leakage + reproduction discipline)

- **From-scratch only for same-contract results.** The sequence encoder
  architecture is re-implemented in `src/` and trained on our locked split. We do
  **not** load externally-pretrained CRISPR weights (e.g. CRISPR-Net/DeepCRISPR
  released checkpoints) as a same-contract result, because their training data may
  overlap our held-out guides/targets. This follows Kapoor & Narayanan (leakage
  taxonomy; overoptimism when test-disjointness is not guaranteed).
- **No reproduction claims.** Our data (measured-only Mak/crisprSQL), split
  (guide-disjoint `sprint2_main_seed42`), label (scheme_a binary), and primary
  metric (AUPRC at ~90% positive prevalence) differ from CRISPR-Net / CRISPR-IP
  (CIRCLE-seq/SITE-seq, LOGOCV). Cite these as **adaptation/inspiration** only;
  document the architecture/encoding adaptation in code comments and this plan.
- **Optional transfer slice is separate and approval-gated** (§7).

## 4. Sequence Input & Encoding

- Base sequence input = Sprint 2 `S1_sequence_pair` style: aligned guide and
  target sequences as guide-base one-hot channels, target-base one-hot channels,
  and one aligned mismatch channel over 23 positions (already defined and audited
  in Sprint 2; reuse, do not redefine). It carries no energy/epigenetic/context
  scalars — keeping the sequence-only signal separate.
- Encoding-design references (cite, do not reproduce): CRISPR-IP (type channels
  for base vs base-pair + a function channel marking sequence regions) and
  CRISPR-Net (compact bulge-aware mismatch/indel encoding). Indels/bulges are NOT
  central to the current measured-only Mak contract; the canonical 8B encoder uses
  the mismatch-aware `S1` pair representation and notes bulge handling as
  out-of-scope unless a later plan adds indel rows.

## 5. Architecture (adapted, from-scratch)

- Canonical 8B sequence encoder = a CRISPR-Net-adapted recurrent-convolutional
  block over the `S1` pair input: Conv (local mismatch/identity features) +
  BiLSTM (positional/context features), producing a fixed-dim sequence embedding
  `seq_embed`. Document the adaptation vs CRISPR-Net (data/split/target/metric
  differ; architecture is inspired, not copied).
- Integration with the Sprint 8A best model is **predeclared and controlled**:
  - `S8B_R0`: Sprint 8A best model (reference row, no retrain).
  - `S8B_R1`: sequence-only — `seq_embed → classifier` (pure sequence baseline,
    re-trained on our split; comparable to Sprint 2 sequence baselines but
    re-evaluated under the 8A harness).
  - `S8B_R2`: late fusion — concatenate `seq_embed` into the Sprint 8A R2
    candidate-edge classifier input vector
    `[source, target, source*target, |source-target|, edge_film, seq_embed]`
    (see §15.3); the GATv2 message passing, target-context encoder, and FiLM head
    stay frozen; only the sequence branch and the fusion-widened classifier are new.
- The GATv2 message passing, target-context encoder, and S5F2 edge-aware
  attention from the Sprint 8A canonical model remain frozen in `S8B_R2`; only the
  added sequence branch + fused classifier head are new.

## 6. Predeclared Run Matrix (minimal)

| Run ID | Setting | Role |
| --- | --- | --- |
| `S8B_R0_reference` | **`S8A_R2_context_edge_film`** — Sprint 8A validation-AUPRC winner (no retrain; batch `sprint8a_target_context_interaction_seed42_20260611_011416`) | anchor |
| `S8B_R1_sequence_only` | CRISPR-Net-adapted `S1` encoder → classifier, from scratch | pure-sequence value test |
| `S8B_R2_sequence_plus_context` | late fusion: `seq_embed` + 8A context/edge head | does sequence add over context? |

Default canonical runs: **2 trained** (`R1`, `R2`) plus the `R0` reference. Do not
add runs after seeing test results.

## 7. Optional / Approval-Gated Transfer Slice (NOT core)

A separately-labelled transfer experiment, only with a plan amendment and
explicit leakage caveats, reported apart from the headline:

- sgRNA side: RNA-FM embeddings; DNA target side: DNABERT-2 embeddings.
- Any positive result using external pretrained weights must be reported as
  **transfer learning with possible overlap to held-out guides/targets**, never
  as a same-contract baseline. Cite Kapoor & Narayanan. Not part of the canonical
  8B claim.

## 8. Selection & Reporting Rules

- Primary selection = validation AUPRC; tie-break = validation MCC / macro F1.
- Test AUPRC reported as primary test metric; test MCC/specificity/macro F1
  secondary. Report every predeclared run. Parameter counts reported per run.
- Allowed conclusions: "sequence-context adds/does not add measurable validation
  AUPRC over the 8A target-context model under the frozen single-seed contract."
- Disallowed: reproduction of CRISPR-Net/CRISPR-IP/DeepCRISPR; same-contract
  claims from externally-pretrained weights; robustness from a single seed;
  MCC/specificity gain framed as an AUPRC gain.

## 9. Output Contract (Sprint 8B pattern)

```text
outputs/sprint8b/sequence_context_comparison.csv
outputs/sprint8b/sequence_context_report.md
outputs/sprint8b/sequence_context_run_manifest.json
outputs/sprint8b/graph_artifact_provenance.json
outputs/sprint8b/diagnostics/...   # threshold metrics, deltas, training history,
                                    # predictions, score deciles, per-guide,
                                    # sequence-input audit, parameter counts
outputs/sprint8b/figures/...        # auprc comparison, PR/ROC, score dists,
                                    # training curves, parameter counts
```

Per-run: `resolved_config.yaml`, `runtime.json`, `training_history.csv`,
`metrics.csv`, `sequence_input_audit.csv`, `model.pt` (untracked).

## 10. Tests & Validation

- sequence encoder forward (shape; `S1` channel layout 23×channels);
- sequence-input audit (no energy/epigenetic/context leakage into the sequence
  branch);
- late-fusion wiring (frozen-context isolation assertion, §15.3): with `seq_embed`
  set to zero, the seq columns contribute zero and the upstream GATv2 /
  target-context / FiLM activations and the 8A classifier-input sub-vector are
  bit-identical to the 8A R2 path — i.e. the sequence branch is purely additive
  through the classifier and never perturbs the frozen context path. (Numerical
  reproduction of the 8A *trained* classifier is NOT claimed: the fusion-widened
  classifier is retrained jointly, standard for late fusion.);
- config parse; output-contract schema; runner smoke test (monkeypatched
  training).

```bash
uv run pytest tests/test_sprint8b_sequence_context.py -q
uv run ruff check src/crispr_gnn/models/sequence_context_encoder.py scripts/run_sprint8b_sequence_context.py tests/test_sprint8b_sequence_context.py
git diff --check
```

## 11. Acceptance Criteria

- Exactly the predeclared run set (`R1`, `R2` trained; `R0` reference); transfer
  slice not run without amendment.
- From-scratch training on the locked split; no external pretrained weights in
  any headline row; sequence branch carries sequence-only signal.
- Frozen 8A context path preserved in `S8B_R2`; output contract complete; tests +
  lint pass; Colab runner-only; no committed `model.pt`.

## 12. Deferred / Out Of Scope

- Indel/bulge-augmented rows (would change the evaluation universe).
- The transfer slice (§7) unless explicitly amended.
- Robustness (multi-seed, bootstrap, paired-difference) → Sprint 9.

## 13. Required Docs Updates On Completion

- `docs/DECISIONS.md`: record the 8B outcome (sequence-context adds / does not add
  over 8A target-context under the frozen contract; adaptation-not-reproduction;
  leakage discipline).
- `README.md` / `CRISPR_GNN_PROJECT_PLAN.md`: mark Sprint 8B status in the roadmap
  (model-improvement, sequence-context slice; Sprint 9 = robustness).

## 14. Implementation Slices

Runs only AFTER Sprint 8A is complete. Incremental, test-gated; from-scratch
training on the locked split only; no headline claim until Slice 5.

### Slice 0 - Planning freeze — Status: COMPLETE (2026-06-11)

Freeze this plan after Sprint 8A closure: run set (R0 reference / R1
sequence-only / R2 late-fusion), the CRISPR-Net-adapted encoder design, the `S1`
sequence input, the leakage + no-reproduction rules (§3), and the output
contract. Confirm the Sprint 8A validation-AUPRC winner that R0/R2 build on.

Exit: plan frozen; no code changed yet.

Done: R0 anchored to `S8A_R2_context_edge_film` (authoritative batch + metrics,
§15.1); decision (a) S1-input-source and (b) late-fusion injection point pinned
(§15.2–15.3) after literature validation (CRISPR-Net/CRISPR-IP/DeepCRISPR encoding,
Kapoor leakage, Sprint 2 late-fusion precedent, Dwivedi parameter budget); encoder
architecture pinned (§15.4); frozen-context assertion redefined as a wiring
isolation test. No code/config/test added; no training; no Sprint 8A result
changed. `DECISIONS.md` records the freeze.

### Slice 1 - Sequence-context encoder (`sequence_context_encoder.py`) — Status: COMPLETE (2026-06-11)

Implement the CRISPR-Net-adapted Conv + BiLSTM encoder over the Sprint 2 `S1`
pair input, from scratch, documenting the adaptation (data/split/target/metric
differ from CRISPR-Net/CRISPR-IP). Add tests: forward + `S1` channel layout
(23 positions); **sequence-input audit proving no energy/epigenetic/context
leakage into the sequence branch**.

Exit: encoder unit tests pass; no training.

Done: added `src/crispr_gnn/models/sequence_context_encoder.py` —
`build_s1_pair_from_onehot` / `build_s1_pair_for_edges` reconstruct the 23×11 `S1`
tensor from the frozen Graph C guide/target one-hot node features (decision (a), no
raw join); `resolve_s1_onehot_indices` + `sequence_input_audit` enforce
sequence-only provenance (raise on incomplete one-hot or non-sequence columns);
`SequenceContextEncoder` is a 1D-Conv + BiLSTM → mean-pool → `seq_embed` (§15.4).
The §15.2 channel-order verification is satisfied by a test that reproduces the
Sprint 2 `build_sequence_pair_encoding` output **byte-exact** (channels + mismatch).
Tests in `tests/test_sprint8b_sequence_context.py` (7) pass; 8A/7F regression green;
ruff + `git diff --check` clean. No `gat.py` / `training/gcn.py` / config / runner /
colab change (Slices 2–4). No training. No reproduction claim. No tech debt added.

### Slice 2 - Integration and dispatch

Wire the sequence-only path (R1) and the late-fusion head (R2) into the Graph C
GATv2 / training dispatch, keeping the Sprint 8A context path frozen in R2. Add
the **frozen-context assertion** (sequence branch zeroed ⇒ R2 reproduces the 8A
head output) and config-parse tests.

Exit: model/dispatch tests pass; Sprint 8A regression tests stay green.

### Slice 3 - Runner, config, reporting, diagnostics

Add `configs/sweeps/sprint8b_sequence_context.yaml`,
`scripts/run_sprint8b_sequence_context.py`, the sequence-input audit, diagnostics,
figures, manifest, provenance, and an output-contract test (monkeypatched
training).

Exit: mocked/smoke outputs satisfy the §9 contract; no headline claim.

### Slice 4 - Colab runner preparation

Add a runner-only Colab notebook and documented command path; validate artifact
copy and returned-output checks.

Exit: notebook contract checks pass; no full GPU claim yet.

### Slice 5 - Full run and local validation

Run R1 and R2 (seed 42) on Colab GPU from scratch on the locked split, copy
outputs under `outputs/sprint8b/`, validate locally, no test-driven tuning.

Exit: all §9 outputs exist (or any technical omission is documented before
interpreting results).

### Slice 6 - Optional transfer slice (approval-gated)

Only with a plan amendment: the RNA-FM / DNABERT-2 transfer experiment (§7),
reported separately with explicit leakage caveats, never as a same-contract row.

Exit: transfer slice reported separately, or explicitly skipped.

### Slice 7 - Sprint closure

Freeze report/results/status, add the `docs/DECISIONS.md` outcome entry (§13),
update the roadmap, and move this plan to `docs/exec-plans/completed/`.

Exit: the Sprint 8B conclusion is documented (sequence-context adds / does not add
over the Sprint 8A target-context model under the frozen contract).

## 15. Frozen Specification (Slice 0)

Slice 0 planning freeze (2026-06-11). All design decisions below are pinned;
Slices 1–7 implement against this section with no remaining design choice. No
`src/`, `configs/`, `scripts/`, `colab/`, or `tests/` file was changed in Slice 0;
no training was run; no Sprint 8A output was modified.

### 15.1 Sprint 8A anchor (R0, no retrain)

- `S8B_R0_reference` = `S8A_R2_context_edge_film`, Sprint 8A validation-AUPRC
  winner. Authoritative batch:
  `sprint8a_target_context_interaction_seed42_20260611_011416`.
- Recorded metrics (carry-forward, no recomputation): validation AUPRC `0.987496`;
  test AUPRC `0.982757`; test AUROC `0.910575`; test MCC `0.563656`;
  TN/FP/FN/TP `88/81/39/1494`; nominal `parameter_count` `381866`.
- Carry-forward interpretation caveats (from the Sprint 8A forensic review):
  - R2 is a **validation-selected candidate, not a superiority claim**; no Sprint 8A
    variant beat XGBoost F4 test AUPRC `0.992522`; superiority/variance is deferred
    to Sprint 9.
  - R2's rare-negative operating-point edge is **partly threshold-landing**
    (validation-max-F1), not purely ranking — so in Sprint 8B, MCC/specificity stay
    secondary and are never used to rank runs.
  - R2's nominal `parameter_count` is inflated by the inactive base
    `edge_classifier` (~100k dead params; see `docs/exec-plans/tech-debt.md`).
    Sprint 8B must report active parameter counts and not over-read nominal counts
    (Dwivedi parameter-budget discipline).

### 15.2 Decision (a) — S1 sequence input source: reconstruct from Graph C artifacts

- The `S1` sgRNA/target pair is **reconstructed deterministically from the frozen
  Graph C artifacts**, not re-derived from raw data:
  - guide one-hot from `nodes_sgRNA.parquet` columns
    `feature__guide_pos_{00..22}_{A,C,G,T,N}` (115 cols, verified present);
  - target one-hot from `features_target_observation_features.parquet` columns
    `feature__target_pos_{00..22}_{A,C,G,T,N}` (115 cols, verified present);
  - aligned mismatch channel computed per position as
    `argmax(guide_onehot) != argmax(target_onehot)`.
  - Channel layout: 5 guide + 5 target + 1 mismatch = 11 channels × 23 positions
    (matches the Sprint 2 `S1_sequence_pair` 23×11 contract; verify exact channel
    order against the Sprint 2 builder in Slice 1).
- Rationale: lossless, deterministic, and **leakage-clean** — no error-prone
  edge-id join to raw data (Kapoor & Narayanan leakage discipline; "do not silently
  join wrong keys"). The artifacts are already edge-aligned and sha256-recorded in
  provenance.
- The sequence branch carries **sequence-only** signal: no energy/epigenetic/
  nucleosome/context scalars. A `sequence_input_audit` must prove this (§10).

### 15.3 Decision (b) — late-fusion injection point: concatenate into the R2 classifier input

- `seq_embed` (one fixed-dim vector per candidate edge = per guide/target pair) is
  **concatenated into the Sprint 8A R2 candidate-edge classifier input vector**:
  `[source, target, source*target, |source-target|, edge_film, seq_embed]`. The
  fusion-widened first classifier layer is retrained jointly in `S8B_R2`.
- Frozen: GATv2 attention/message passing, the target-context encoder, and the FiLM
  head from the Sprint 8A R2 model are unchanged; only the sequence branch and the
  fusion-widened classifier are new.
- **Frozen-context isolation assertion** (replaces "reproduces 8A head"): zeroing
  `seq_embed` makes the seq columns contribute zero and leaves the upstream
  activations and the 8A classifier-input sub-vector bit-identical to the 8A R2
  path. This is a wiring/no-confound test; numerical reproduction of the 8A
  *trained* classifier is NOT claimed (joint retrain is standard late fusion).
- Rationale: matches the project's own Sprint 2 late-fusion precedent
  (`sequence_cnn_plus_F3/F4_late_fusion`, DECISIONS 2026-05-23) and the late-dense
  combination in CRISPR-IP/DeepCRISPR; granularity-aligned (per candidate edge);
  keeps the controlled "does sequence add over context?" question clean. An
  early-fusion / seq×context FiLM interaction is explicitly **out of scope** for
  the 8B core (it would entangle and confound the frozen 8A path).

### 15.4 Encoder architecture (adapted, from-scratch)

- `seq_embed` = small **1D-Conv (local mismatch/identity features) + BiLSTM
  (positional features)** over the 23×11 `S1` tensor → fixed-dim embedding.
- Adapted from CRISPR-Net (Inception-conv + BiLSTM) and CRISPR-IP (CNN-identity +
  BiLSTM-position); attention is **deferred** (kept out of the core to limit scope
  and overtuning risk — Overtuning discipline).
- **No reproduction claim**: data (measured-only Mak/crisprSQL), split
  (guide-disjoint `sprint2_main_seed42`), label (scheme_a), and primary metric
  (AUPRC at ~90% prevalence) differ from the source papers. Document the adaptation
  in code comments and this plan.

### 15.5 Run matrix, contract, and discipline (pinned)

- Canonical runs: `S8B_R0_reference` (no retrain), `S8B_R1_sequence_only`
  (seq-encoder → classifier, from scratch), `S8B_R2_sequence_plus_context`
  (late fusion per §15.3). Transfer slice (RNA-FM / DNABERT-2) is approval-gated
  only (§7); no external pretrained weights in any headline row.
- Frozen evaluation contract (inherited): `scheme_a`, `sprint2_main_seed42`,
  guide-disjoint, measured-only headline, `experiment_id=18` excluded, train-only
  preprocessing, validation-only checkpoint (val AUPRC) and threshold
  (validation max-F1), no test-driven selection, AUPRC primary, seed 42, Colab
  runner-only, core logic in `src/`.
- Selection: validation AUPRC primary; report every predeclared run; report active
  parameter counts; MCC/specificity are secondary threshold diagnostics (never
  used to rank); single-seed → directional only, superiority deferred to Sprint 9.

### 15.6 Slice 0 exit

Sprint 8B plan frozen; Slices 1–7 may proceed against §15; no code changed.
