# Exec Plan: Sprint 2 Fair Non-Graph and Sequence Baselines

## Goal

Establish fair, same-split non-graph baselines for Scheme A CRISPR-Cas9 off-target classification before any GNN novelty claims.

Sprint 2 should answer:

- How well do sequence, mismatch, binding-energy, and epigenetic/context features perform without graph structure?
- Which baseline should future Sprint 4+ GNN models compare against under the same guide-level split, label scheme, feature policy, and metrics?
- Does the dataset/feature pipeline support credible supervised learning without leakage?

Required final deliverables:

```text
outputs/results/baseline_results.csv
outputs/reports/baseline_report.md
```

Supporting deliverables required for reproducibility:

```text
outputs/splits/sprint2_guides.json
outputs/splits/sprint2_split_summary.csv
outputs/features/sprint2_feature_catalog.md
```

## Inputs

- `CRISPR_GNN_PROJECT_PLAN.md`
- `PROJECT_FOLDER_STRUCTURE.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/DATASET_AUDIT.md`
- `docs/LABEL_SCHEMES.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/FEATURE_PARSING.md`
- `docs/DECISIONS.md`
- `docs/literature/literature_index.md`
- Dataset config: `configs/data/mak2022.yaml`
- Primary raw dataset: `data/raw/260520_putative_nucleosomal.parquet`

Relevant Sprint 1 decisions:

- Primary binary label is Scheme A: `cleavage_freq > 1e-5`.
- NaN `cleavage_freq` rows are excluded from supervised label generation.
- Test rows must be `measured=1` only.
- Validation should use `measured=1`.
- Training may include `measured=0` rows only as optional noisy negatives.
- Final evaluation uses guide-level split; random edge split is debug-only.
- `experiment_id=18` is a main-evaluation risk.
- Computed nucleosome arrays must parse as exactly 23 numeric values, with missingness tracked separately.

Sprint 2 decisions locked on 2026-05-23:

- Exclude `experiment_id=18` from main Sprint 2 train/validation/test rows.
- Add XGBoost as the official boosted-tree tabular baseline dependency.
- Add PyTorch for non-graph sequence baselines only; defer PyTorch Geometric.

## Current Status

Completed:

- Sprint 2 baseline dependencies were added and validated:
  - `torch`
  - `xgboost`
- Guide-level split infrastructure was added:
  - `src/crispr_gnn/data/splits.py`
  - `scripts/build_splits.py`
  - `tests/test_splits.py`
- Locked Sprint 2 split artifacts were generated:
  - `outputs/splits/sprint2_guides.json`
  - `outputs/splits/sprint2_split_summary.csv`

Latest split verification:

- Split ID: `sprint2_main_seed42`.
- Main split excludes `experiment_id=18`.
- Train/validation/test guides are disjoint.
- Validation and test contain `measured=1` rows only.
- Every split contains both positive and negative Scheme A labels.
- Current measured-only main split sizes:
  - train: 8,010 rows, 98 guides, 7,109 positives, 901 negatives.
  - validation: 1,734 rows, 23 guides, 1,511 positives, 223 negatives.
  - test: 1,702 rows, 29 guides, 1,533 positives, 169 negatives.
- Expected guide-size caveat: the project already requires guide-level evaluation and the Sprint 1 audit documented highly uneven guide sizes. As an expected consequence, a few high-row-count guides can dominate split-level metrics. In the locked split, the largest guide accounts for 27.6% of train rows, 28.7% of validation rows, and 24.0% of test rows. This is the known tradeoff of leakage-safe guide-level evaluation on this dataset, not evidence of a split bug, and must be considered when interpreting AUPRC and secondary metrics.

Completed after split lock:

- Sprint 2 tabular feature ladder infrastructure was added:
  - `src/crispr_gnn/features/tabular.py`
  - `scripts/build_features.py`
  - `tests/test_tabular_features.py`
- Feature catalog artifacts were generated:
  - `outputs/features/sprint2_feature_catalog.md`
  - `outputs/features/sprint2_feature_summary.csv`
- Current feature-set dimensions on the locked split:
  - F1: 33 columns, 11,446 rows, no missing values.
  - F2: 38 columns, 11,446 rows, no missing values.
  - F3: 44 columns, 11,446 rows, no missing values.
  - F4: 135 columns, 11,446 rows, 789 rows with missing computed-nucleosome aggregates before train-only imputation.

Next step:

- Draft the Sprint 2 baseline report from the completed tabular and sequence baseline artifacts.

## Scope

In scope:

- Implement guide-level split generation and validation.
- Build a model-ready feature ladder from sequence, mismatch, binding-energy, experimental epigenetic, and computed nucleosome features.
- Implement train-only preprocessing for imputation, scaling, and feature transforms.
- Train and evaluate:
  - dummy/prevalence sanity baseline,
  - Logistic Regression,
  - XGBoost,
  - MLP,
  - CnnCrispr-inspired sequence CNN/BiLSTM baseline.
- Report all baselines on the same locked split and Scheme A labels.
- Add focused tests for split leakage, feature preprocessing, metrics, and smoke training.
- Update docs when dependency setup, commands, or evaluation behavior changes.

Out of scope:

- GCN, GAT, GraphSAGE, heterogeneous GNN, or graph construction.
- PyTorch Geometric.
- Scheme B / Mak CA reproduction.
- Scheme C robustness training.
- CRISPRoffT external validation.
- Full 299-dimensional computed nucleosome feature ablation unless time remains after required baselines.
- Exhaustive hyperparameter leaderboard chasing.
- Claims about epigenetic feature importance beyond Sprint 2 baseline comparisons.

## Main Data Policy

### Main Clean Universe

Sprint 2 main baselines use a `main_clean` row universe:

- Scheme A label-eligible rows only.
- Exclude rows with NaN `cleavage_freq`.
- Exclude `experiment_id=18` from train, validation, and test.
- Validation and test contain `measured=1` rows only.
- Training uses `measured=1` rows for required baselines.

`experiment_id=18` may be evaluated separately only as a no-cell-line / high-computed-missingness sensitivity subset if enough measured, label-eligible rows exist. Sensitivity results must not be mixed into the headline baseline table.

### Optional Putative Augmentation

Required Sprint 2 acceptance uses measured-only training.

Optional putative augmentation may add `measured=0` rows from training guides only:

- deterministic sampling with fixed seed,
- per-guide capped,
- default cap: at most 1:1 putative-to-measured ratio,
- never include `measured=0` in validation or test,
- report as `putative_augmented`, separate from `measured_only`.

## Split Policy

The required split is one locked guide-level split.

Hard checks:

- Train, validation, and test guide IDs are disjoint.
- Test is `measured=1` only.
- Validation is `measured=1` only for Sprint 2 main baselines.
- `experiment_id=18` is absent from main split rows.
- Each split has nonzero positives and negatives.
- Row counts, guide counts, positive/negative counts, measured composition, genome distribution, and largest-guide contribution are reported.

Expected guide-size caveat:

- Guide-level splitting prevents guide leakage, but it does not make all guides equally weighted. The Mak 2022 main-clean dataset is known to have highly uneven guide sizes from the Sprint 1 audit, so a high-row-count guide can strongly influence split-level metrics. Reports must include `largest_guide_share` and avoid overinterpreting small metric differences without considering large-guide dominance.

Repeated guide-level seeds are optional robustness work and do not block Sprint 2 acceptance.

Random row/edge split may exist only for debug smoke checks and must not appear as final model performance.

## Feature Ladder

Feature sets must be named and reproducible.

Required:

```text
F0: dummy/prevalence baseline
F1: sequence + mismatch engineered features
F2: F1 + binding-energy scalars (`energy_1` to `energy_5`)
F3: F2 + 6 experimental epigenetic scalar features
F4: F3 + aggregated computed nucleosome features + missingness indicators
```

Optional:

```text
F5: full position-resolved computed nucleosome features (13 x 23 = 299)
```

F4 policy:

- Parse computed nucleosome arrays using the Sprint 1 strict parser.
- Aggregate valid 23-position arrays using a documented recipe.
- Treat missing or invalid arrays as missing.
- Fit imputation statistics on training data only.
- Add explicit missingness indicators.
- Do not row-drop computed-feature-missing rows in main F4 comparisons.

Raw identifiers such as guide ID, target location, experiment ID, cell line ID, and genome label are used for splitting/reporting only, not as predictive features in Sprint 2 main baselines.

## Baseline Lineup

Required models:

1. Dummy/prevalence sanity baseline.
2. Logistic Regression.
3. XGBoost boosted-tree classifier.
4. MLP.
5. CnnCrispr-inspired sequence CNN/BiLSTM baseline.

Optional models:

- Random Forest fallback/sanity baseline.
- scikit-learn boosted-tree fallback if XGBoost is unavailable.
- Putative-augmented variants of required models.
- CNN + context late-fusion variant.
- F5 position-resolved feature variants.

Implementation order:

1. Dummy/prevalence baseline to validate metrics and result schema.
2. Logistic Regression on F1-F4 to validate features, preprocessing, and leakage checks.
3. XGBoost on F1-F4 as the official tabular baseline.
4. MLP after the tabular pipeline is stable.
5. CnnCrispr-inspired sequence model after split and metrics are locked.

Current status:

- Dummy/prevalence and unweighted scikit-learn Logistic Regression are implemented for F1-F4 in `scripts/train.py` via `configs/experiments/baseline_logistic_regression.yaml`.
- The run writes `outputs/results/baseline_results.csv`.
- The run writes report-ready figures under `outputs/figures/sprint2/`:
  - `logistic_regression_feature_set_auprc.png`
  - `logistic_regression_pr_curves.png`
  - `logistic_regression_roc_curves.png`
- The run writes diagnostic tables and figures under `outputs/diagnostics/sprint2/`:
  - score-direction checks
  - fixed-threshold metrics at 0.5
  - per-genome metrics
  - per-guide test metrics
  - score-decile lift tables and plots
- The initial Logistic Regression result is a wiring/debug baseline, not the expected strongest Sprint 2 model. AUPRC is interpreted against the measured-only test prevalence baseline.
- XGBoost is the next implementation step. It should upsert results into the shared `outputs/results/baseline_results.csv` without removing the dummy/Logistic Regression rows, and it should use:
  - `xgboost_unweighted` as the primary XGBoost baseline.
  - `xgboost_balanced_train_weights` as a separately labeled negative-class sensitivity.
  - validation-only early stopping/model selection.
  - the same F1-F4 feature sets and diagnostic artifact style as Logistic Regression.
- XGBoost F1-F4 has been implemented and run. The strongest current measured-only result is `xgboost_unweighted` on `F4`:
  - test AUPRC: 0.992522
  - test AUROC: 0.938416
  - test MCC: 0.345198
  - confusion matrix at validation-selected threshold: TN=38, FP=131, FN=21, TP=1512
- XGBoost diagnostics were generated under `outputs/diagnostics/sprint2/`. Results must still be interpreted with the locked split's guide/genome composition caveat and must not be used to retune the test set.
- Follow-up XGBoost audit artifacts were added:
  - `xgboost_feature_column_audit.csv` confirms no forbidden target, split, measured, genome, cell-line, experiment, guide, coordinate, or raw identifier columns are predictive features.
  - `xgboost_feature_importance.csv` records feature/family importance for every XGBoost feature set and training variant.
  - In the current unweighted F4 run, computed nucleosome missingness indicators have zero total gain; F4's added computed nucleosome aggregate features contribute a smaller lift than the F3 experimental epigenetic scalars.
- Tabular MLP has been implemented with scikit-learn `MLPClassifier` via `configs/experiments/baseline_mlp.yaml`.
  - `tabular_mlp_unweighted` runs on F1-F4.
  - `tabular_mlp_balanced_train_weights` runs as a focused F3/F4 training-weight sensitivity.
  - The run writes `tabular_mlp_training_summary.csv`, `tabular_mlp_feature_column_audit.csv`, PR/ROC/AUPRC plots, score-direction diagnostics, fixed-threshold diagnostics, per-genome/per-guide diagnostics, and score-decile diagnostics.
  - In the current measured-only run, the strongest MLP result is `tabular_mlp_unweighted` on F3:
    - test AUPRC: 0.959889
    - test AUROC: 0.735388
    - test MCC: 0.220770
    - confusion matrix at validation-selected threshold: TN=31, FP=138, FN=46, TP=1487
  - The MLP improves over Logistic Regression on context-rich features but does not surpass XGBoost F3/F4. XGBoost remains the current strongest non-graph tabular baseline.
- Sequence-only CNN/BiLSTM baselines have been implemented via `configs/experiments/sequence_cnn_bilstm.yaml`.
  - Feature set `S1` uses `S1_sequence_pair`: guide one-hot channels, target one-hot channels, and aligned mismatch channel over 23 positions.
  - The sequence input audit confirms only `grna_target_sequence` and `target_sequence` are used as source columns.
  - Pure sequence models do not receive binding energy, epigenetic/context features, genome/cell-line labels, experiment IDs, guide IDs, coordinates, measured flags, labels, or cleavage values.
  - The current run includes:
    - `sequence_cnn_unweighted`
    - `sequence_bilstm_unweighted`
    - `sequence_cnn_balanced_train_weights`
    - `sequence_bilstm_balanced_train_weights`
  - The strongest sequence-only result is `sequence_cnn_unweighted`:
    - test AUPRC: 0.920075
    - test AUROC: 0.535711
    - test MCC: -0.019749
  - Sequence-only models are weak under the locked measured-only guide-level split. The BiLSTM is directionally correct on validation but inverted on test, so it should be interpreted as poor guide-held-out sequence-only generalization rather than as a strong baseline.
  - XGBoost F3/F4 remains the strongest Sprint 2 non-graph baseline.

## Metrics And Thresholding

Primary metric:

- AUPRC.

Secondary metrics:

- AUROC.
- F1.
- MCC.
- Precision@K.
- Recall at fixed FPR.
- Confusion matrix.

Rules:

- Report test positive prevalence alongside AUPRC.
- Select classification thresholds on validation only.
- Apply the selected threshold once to test.
- Do not tune on the test set.
- Compare models only on the same split, label scheme, training regime, and feature policy.

## File And Module Boundaries

Reuse the existing repository structure.

Expected code locations:

```text
src/crispr_gnn/data/splits.py
src/crispr_gnn/features/
src/crispr_gnn/models/
src/crispr_gnn/training/
src/crispr_gnn/evaluation/
```

Expected scripts:

```text
scripts/build_splits.py
scripts/train.py
scripts/evaluate.py
```

Scripts should be thin, config-driven entrypoints. Do not create a separate `src/crispr_gnn/splits/` package. Do not add many sprint-specific scripts unless a general script would become unclear.

Expected config additions:

```text
configs/experiments/baseline_logistic_regression.yaml
configs/experiments/baseline_xgboost.yaml
configs/experiments/baseline_mlp.yaml
configs/experiments/sequence_cnn_bilstm.yaml
```

Config names may change if the implementation finds a clearer existing convention, but the experiments must remain reproducible from config files.

## Tests And Verification

Required tests:

- Label policy still excludes NaN `cleavage_freq` and preserves Scheme A counts.
- Guide-level split has disjoint train/validation/test guide IDs.
- Validation and test contain only `measured=1`.
- Main split excludes `experiment_id=18`.
- Optional putative augmentation never uses validation/test guides and never enters validation/test rows.
- Feature preprocessing fits imputation/scaling statistics on training data only.
- Computed nucleosome aggregation handles missing arrays and adds indicators.
- Raw identifiers are not included as predictive features.
- Metrics produce AUPRC, AUROC, F1, MCC, Precision@K, recall at fixed FPR, and confusion matrix.
- Threshold selection uses validation scores only.
- A small smoke run trains at least Logistic Regression end to end and writes result rows.

Required commands before marking Sprint 2 complete:

```bash
uv sync
uv run pytest -q
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml
uv run python scripts/train.py --config configs/experiments/baseline_logistic_regression.yaml
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
uv run python scripts/train.py --config configs/experiments/baseline_mlp.yaml
uv run python scripts/train.py --config configs/experiments/sequence_cnn_bilstm.yaml
uv run python scripts/evaluate.py --config configs/experiments/baseline_xgboost.yaml
```

Exact command flags may change during implementation, but all commands must use `uv run`.

## Acceptance Criteria

Sprint 2 is complete when:

- Scheme A label generation follows the documented Sprint 1 policy.
- One locked guide-level split exists and is documented.
- Train/validation/test guide sets are disjoint.
- Validation and test are `measured=1` only.
- Main split excludes `experiment_id=18`.
- `outputs/splits/sprint2_guides.json` exists.
- `outputs/splits/sprint2_split_summary.csv` exists.
- `outputs/features/sprint2_feature_catalog.md` exists.
- F1-F4 feature sets are documented and reproducible.
- F4 uses train-only imputation and missingness indicators.
- Logistic Regression, XGBoost, MLP, and one CnnCrispr-inspired sequence baseline are trained and evaluated.
- `outputs/results/baseline_results.csv` exists and includes model, feature set, split ID, seed, training regime, label scheme, and metrics.
- `outputs/reports/baseline_report.md` explains dataset, label policy, split policy, feature ladder, model lineup, results, caveats, and the recommended future GNN comparison baseline.
- Random split results are not reported as final performance.
- Relevant docs are updated for dependency and evaluation policy changes.
- `uv run pytest -q` passes.

## Open Follow-Ups

- Choose the exact guide-level split balancing heuristic after inspecting guide row/positive distributions.
- Choose the exact F4 aggregation recipe before implementation.
- Decide whether XGBoost class weighting or scale-pos-weight is part of the default baseline after split composition is known.
- Decide whether to run optional putative augmentation in Sprint 2 or defer it until after measured-only baselines are stable.
