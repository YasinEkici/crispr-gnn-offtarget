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
