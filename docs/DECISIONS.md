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

Reason: Sprint 1 audit confirmed `cleavage_freq` is present, transformed `CA` is absent, and Scheme A gives 21,365 positives and 288,777 negatives on the full dataset. The threshold is aligned with the paper's assay-accuracy boundary.

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

- NaN `cleavage_freq`: label-ineligible until a documented policy is approved.
- Negative `cleavage_freq`: below-threshold for binary sensitivity counts, but flagged as raw-label quality issue.
- `cleavage_freq > 1`: positive for binary thresholds; do not clip for binary classification.

Reason: Sprint 1 audit found 78 NaN values, 685 negative values, and 298 values above 1. Silent imputation, clipping, or dropping would make later labels hard to audit.

Outcome: label generation must preserve these policies and report affected counts.

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
