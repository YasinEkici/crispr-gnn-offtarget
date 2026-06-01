# Sprint 2 Baseline Report

Generated: 2026-05-24

## 1. Objective

Sprint 2 established fair, same-split non-graph baselines for Scheme A CRISPR-Cas9 off-target classification before any future GNN claims.

The central question was:

> Before using graph structure, how well do sequence, mismatch, binding-energy, epigenetic/context, and sequence-neural baselines perform under the locked guide-level measured-only evaluation protocol?

This report summarizes the completed Sprint 2 baseline suite and the diagnostics used to interpret it. The final conclusion is that **XGBoost with F3/F4 context-rich tabular features is the strongest Sprint 2 non-graph baseline**. Future GNN models should be compared primarily against this baseline, not only against Logistic Regression or sequence-only neural models.

Primary artifacts:

- Results table: `outputs/sprint2/baseline_results.csv`
- Split manifest: `outputs/splits/sprint2_guides.json`
- Split summary: `outputs/splits/sprint2_split_summary.csv`
- Feature catalog: `outputs/features/sprint2_feature_catalog.md`
- Feature summary: `outputs/features/sprint2_feature_summary.csv`
- Diagnostic tables and figures: `outputs/sprint2/diagnostics/`
- Report-ready figures: `outputs/sprint2/figures/`

## 2. Dataset, Label, And Split Policy

The primary supervised label is Scheme A:

```text
positive = cleavage_freq > 1e-5
negative = cleavage_freq <= 1e-5
```

Rows with missing `cleavage_freq` are excluded from supervised label generation. Negative `cleavage_freq` values remain below-threshold labels and are documented as raw-label quality issues; values above 1 are positive and are not clipped for binary classification.

The Sprint 2 main universe is the measured-only, main-clean universe:

- Scheme A label-eligible rows only.
- Main train/validation/test exclude `experiment_id=18`.
- Validation and test contain only `measured=1`.
- Required baselines use measured-only training.
- `measured=0` rows are not used in this report.
- Random row/edge split results are not reported as final performance.

The locked split is `sprint2_main_seed42`. It is guide-level, so no guide appears in more than one of train, validation, and test.

| split | rows | guides | positives | negatives | positive_rate | largest_guide_share | genome_counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 8010 | 98 | 7109 | 901 | 0.887516 | 0.276404 | hg19=6601, hg38=1121, mm10=222, mm9=34, rn5=32 |
| val | 1734 | 23 | 1511 | 223 | 0.871396 | 0.286621 | hg19=1368, hg38=333, mm10=22, rn5=11 |
| test | 1702 | 29 | 1533 | 169 | 0.900705 | 0.239718 | hg19=1446, hg38=176, mm10=80 |

The test set is highly positive-heavy: test prevalence is `0.900705`. Therefore, AUPRC and F1 must always be interpreted against the dummy/prevalence baseline, not as standalone evidence of model quality.

The guide-level split is valid and leakage-safe, but the dataset has highly uneven guide sizes. In the test set:

- 29 guides total.
- 20 guides are positive-only.
- 1 guide is negative-only.
- 8 guides contain both classes.
- Guide `9251` contributes 80 of the 169 test negatives.

This guide/genome composition affects aggregate metrics. It is not a split bug, but it is a core limitation when interpreting per-test results.

## 3. Feature Sets

Tabular feature sets are documented in `outputs/features/sprint2_feature_catalog.md`.

| feature_set | columns | rows | rows_with_missing | columns_with_missing | total_missing_values |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 33 | 11446 | 0 | 0 | 0 |
| F2 | 38 | 11446 | 0 | 0 | 0 |
| F3 | 44 | 11446 | 0 | 0 | 0 |
| F4 | 135 | 11446 | 789 | 78 | 61542 |

Feature definitions:

- `F1`: sequence and mismatch engineered numeric features.
- `F2`: F1 plus binding-energy scalar features.
- `F3`: F2 plus experimental epigenetic scalar features.
- `F4`: F3 plus aggregated computed nucleosome features and missingness indicators.

F4 missing values are handled by train-only median imputation plus explicit missingness indicators. Rows are not dropped for F4 missingness, so F1-F4 comparisons use the same row universe.

Sequence feature sets:

- `S1`: sequence-only aligned guide/target representation.
- `S1+F3`: CNN late fusion using raw sequence representation plus train-only preprocessed F3.
- `S1+F4`: CNN late fusion using raw sequence representation plus train-only preprocessed F4.

`S1` uses guide-base one-hot channels, target-base one-hot channels, and one aligned mismatch channel over 23 positions. It uses only:

- `grna_target_sequence`
- `target_sequence`

Input audits confirmed zero forbidden predictive columns:

| audit file | rows | forbidden columns |
| --- | ---: | ---: |
| `outputs/sprint2/diagnostics/xgboost_feature_column_audit.csv` | 250 | 0 |
| `outputs/sprint2/diagnostics/tabular_mlp_feature_column_audit.csv` | 250 | 0 |
| `outputs/sprint2/diagnostics/sequence_input_audit.csv` | 2 | 0 |
| `outputs/sprint2/diagnostics/sequence_late_fusion_input_audit.csv` | 183 | 0 |

Raw identifiers, guide IDs, target coordinates, experiment IDs, genome labels, cell-line labels, measured flags, labels, scores, and cleavage values are not predictive features in the Sprint 2 main baselines.

## 4. Models Evaluated

The completed baseline suite includes 28 result rows in `outputs/sprint2/baseline_results.csv` with no duplicate model/feature keys.

Model families:

- Dummy/prevalence baseline.
- Logistic Regression on F1-F4.
- XGBoost on F1-F4, with unweighted and balanced-train-weight variants.
- Tabular MLP on F1-F4, with focused F3/F4 balanced-train-weight sensitivity.
- Pure sequence CNN/BiLSTM on S1, with unweighted and balanced-train-weight variants.
- CNN + F3/F4 late-fusion unweighted variants.

All models use:

- Scheme A labels.
- Locked split `sprint2_main_seed42`.
- Measured-only training/validation/test in this report.
- Validation-selected threshold by maximum validation F1.
- Test set scored once after model/threshold selection.

## 5. Main Results

Full result table is in `outputs/sprint2/baseline_results.csv`.

| model | feature_set | test_AUPRC | test_AUROC | test_F1 | test_MCC | test confusion `(TN, FP, FN, TP)` |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dummy_prior | F1/F2/F3/F4 | 0.900705 | 0.500000 | 0.947759 | 0.000000 | 0, 169, 0, 1533 |
| logistic_regression | F1 | 0.912826 | 0.465310 | 0.947434 | -0.008050 | 0, 169, 1, 1532 |
| logistic_regression | F2 | 0.890767 | 0.394609 | 0.942857 | -0.031308 | 0, 169, 15, 1518 |
| logistic_regression | F3 | 0.906506 | 0.473512 | 0.943841 | -0.027978 | 0, 169, 12, 1521 |
| logistic_regression | F4 | 0.917126 | 0.522015 | 0.943150 | -0.011986 | 1, 168, 15, 1518 |
| xgboost_unweighted | F1 | 0.870583 | 0.315231 | 0.900551 | -0.067603 | 5, 164, 143, 1390 |
| xgboost_unweighted | F2 | 0.884561 | 0.365075 | 0.947759 | 0.000000 | 0, 169, 0, 1533 |
| xgboost_unweighted | F3 | 0.985918 | 0.906893 | 0.950913 | 0.320396 | 35, 134, 22, 1511 |
| xgboost_unweighted | F4 | 0.992522 | 0.938416 | 0.952141 | 0.345198 | 38, 131, 21, 1512 |
| xgboost_balanced_train_weights | F3 | 0.985503 | 0.909699 | 0.951962 | 0.326182 | 33, 136, 17, 1516 |
| xgboost_balanced_train_weights | F4 | 0.986039 | 0.910305 | 0.952083 | 0.314439 | 29, 140, 13, 1520 |
| tabular_mlp_unweighted | F1 | 0.927354 | 0.554576 | 0.935797 | -0.014822 | 3, 166, 39, 1494 |
| tabular_mlp_unweighted | F2 | 0.925531 | 0.549022 | 0.942164 | -0.016579 | 1, 168, 18, 1515 |
| tabular_mlp_unweighted | F3 | 0.959889 | 0.735388 | 0.941735 | 0.220770 | 31, 138, 46, 1487 |
| tabular_mlp_unweighted | F4 | 0.945968 | 0.678354 | 0.944877 | 0.093593 | 8, 161, 16, 1517 |
| sequence_cnn_unweighted | S1 | 0.920075 | 0.535711 | 0.945804 | -0.019749 | 0, 169, 6, 1527 |
| sequence_bilstm_unweighted | S1 | 0.886870 | 0.340957 | 0.947759 | 0.000000 | 0, 169, 0, 1533 |
| sequence_cnn_plus_F3_late_fusion_unweighted | S1+F3 | 0.911630 | 0.483872 | 0.945116 | 0.000181 | 1, 168, 9, 1524 |
| sequence_cnn_plus_F4_late_fusion_unweighted | S1+F4 | 0.928996 | 0.579449 | 0.947759 | 0.000000 | 0, 169, 0, 1533 |

Primary figure references:

- Logistic Regression AUPRC/PR/ROC: `outputs/sprint2/figures/logistic_regression_feature_set_auprc.png`, `outputs/sprint2/figures/logistic_regression_pr_curves.png`, `outputs/sprint2/figures/logistic_regression_roc_curves.png`
- XGBoost AUPRC/PR/ROC: `outputs/sprint2/figures/xgboost_feature_set_auprc.png`, `outputs/sprint2/figures/xgboost_unweighted_pr_curves.png`, `outputs/sprint2/figures/xgboost_unweighted_roc_curves.png`
- MLP AUPRC/PR/ROC: `outputs/sprint2/figures/tabular_mlp_feature_set_auprc.png`, `outputs/sprint2/figures/tabular_mlp_unweighted_pr_curves.png`, `outputs/sprint2/figures/tabular_mlp_unweighted_roc_curves.png`
- Sequence-only AUPRC/PR/ROC: `outputs/sprint2/figures/sequence_feature_set_auprc.png`, `outputs/sprint2/figures/sequence_cnn_unweighted_pr_curves.png`, `outputs/sprint2/figures/sequence_bilstm_unweighted_pr_curves.png`
- Late-fusion AUPRC/PR/ROC: `outputs/sprint2/figures/sequence_late_fusion_feature_set_auprc.png`, `outputs/sprint2/figures/sequence_cnn_plus_F4_late_fusion_unweighted_pr_curves.png`, `outputs/sprint2/figures/sequence_cnn_plus_F4_late_fusion_unweighted_roc_curves.png`

## 6. Logistic Regression

Logistic Regression served as a wiring/debug baseline. It confirmed that labels, feature ladders, preprocessing, metrics, and result output were functional, but it did not produce a strong guide-held-out classifier.

Best Logistic Regression row:

- `logistic_regression / F4`
- test AUPRC: 0.917126
- test AUROC: 0.522015
- test MCC: -0.011986
- confusion matrix: TN=1, FP=168, FN=15, TP=1518

Although F4 had the highest Logistic Regression AUPRC and the only AUROC above 0.5, the effect is weak. AUPRC is only modestly above the prevalence baseline, and the validation-selected threshold recovered only 1 of 169 test negatives.

Diagnostic references:

- `outputs/sprint2/diagnostics/logistic_regression_score_direction.csv`
- `outputs/sprint2/diagnostics/logistic_regression_fixed_threshold_metrics.csv`
- `outputs/sprint2/diagnostics/logistic_regression_per_genome_metrics.csv`
- `outputs/sprint2/diagnostics/logistic_regression_score_deciles.csv`
- `outputs/sprint2/diagnostics/logistic_regression_test_decile_lift.png`

Conclusion: Logistic Regression is a weak linear baseline and should not be the primary baseline for future GNN comparison.

## 7. XGBoost

XGBoost is the strongest Sprint 2 non-graph baseline.

The important pattern is the feature ladder:

- F1/F2 XGBoost are weak or inverted on test.
- F3 creates a large jump.
- F4 adds the strongest headline result.

Best XGBoost row:

- `xgboost_unweighted / F4`
- validation AUPRC: 0.985808
- validation AUROC: 0.925341
- test AUPRC: 0.992522
- test AUROC: 0.938416
- test MCC: 0.345198
- confusion matrix: TN=38, FP=131, FN=21, TP=1512

XGBoost F3/F4 are not just high-AUPRC because of high prevalence. They also improve AUROC, MCC, score decile lift, and negative-class recovery relative to Logistic Regression, MLP, sequence-only models, and late-fusion CNN.

Feature importance diagnostics are in `outputs/sprint2/diagnostics/xgboost_feature_importance.csv`.

For `xgboost_unweighted / F3`, total-gain family share:

- sequence_summary: 46.57%
- experimental_epigenetic: 41.57%
- binding_energy: 8.09%
- mismatch_position: 3.77%

For `xgboost_unweighted / F4`, total-gain family share:

- sequence_summary: 47.50%
- experimental_epigenetic: 42.30%
- computed_nucleosome_aggregates: 5.19%
- binding_energy: 4.20%
- mismatch_position: 0.82%
- computed_nucleosome_missingness: 0.00%

Top XGBoost F4 total-gain features:

| feature | family | total_gain |
| --- | --- | ---: |
| guide_gc_fraction | sequence_summary | 11349.63 |
| MNase | experimental_epigenetic | 8583.37 |
| epigen_h3k4me3 | experimental_epigenetic | 1713.36 |
| energy_1 | binding_energy | 853.96 |
| mismatch_count | sequence_summary | 488.55 |
| epigen_drip | experimental_epigenetic | 428.13 |
| NucleotideBDM_max | computed_nucleosome_aggregates | 350.99 |
| NucleotideBDM_mean | computed_nucleosome_aggregates | 334.57 |

The F4 result should be phrased carefully. Computed nucleosome aggregates add a smaller F4 lift, but the main XGBoost jump begins at F3. The strongest conclusion is that context-rich tabular features matter, especially experimental epigenetic scalars, with computed nucleosome aggregates adding some additional model-dependent signal.

XGBoost diagnostics:

- `outputs/sprint2/diagnostics/xgboost_feature_column_audit.csv`
- `outputs/sprint2/diagnostics/xgboost_feature_importance.csv`
- `outputs/sprint2/diagnostics/xgboost_unweighted_score_direction.csv`
- `outputs/sprint2/diagnostics/xgboost_unweighted_fixed_threshold_metrics.csv`
- `outputs/sprint2/diagnostics/xgboost_unweighted_score_deciles.csv`
- `outputs/sprint2/diagnostics/xgboost_unweighted_test_decile_lift.png`
- `outputs/sprint2/diagnostics/xgboost_unweighted_test_per_genome_auroc.png`

Conclusion: XGBoost F3/F4 is the main Sprint 2 baseline that later GNNs must beat.

## 8. Tabular MLP

The tabular MLP provides a neural non-graph tabular baseline. It improves substantially over Logistic Regression on context-rich features but does not surpass XGBoost.

Best MLP row:

- `tabular_mlp_unweighted / F3`
- validation AUPRC: 0.973276
- validation AUROC: 0.857247
- test AUPRC: 0.959889
- test AUROC: 0.735388
- test MCC: 0.220770
- confusion matrix: TN=31, FP=138, FN=46, TP=1487

F3 is better than F4 for the current MLP configuration on both validation and test. This is plausible rather than suspicious: F4 expands the feature space from 44 to 135 columns, introduces imputed computed aggregates, and may add correlated/noisy dimensions that a small MLP cannot exploit as cleanly as XGBoost.

The balanced MLP sensitivity is useful diagnostically but is not the headline MLP result. It changes threshold behavior and remains below unweighted F3 under the official validation-threshold policy.

MLP diagnostics:

- `outputs/sprint2/diagnostics/tabular_mlp_feature_column_audit.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_training_summary.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_score_direction.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_fixed_threshold_metrics.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_score_deciles.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_test_decile_lift.png`

Conclusion: the MLP is a useful neural tabular baseline, but XGBoost remains the stronger non-graph reference.

## 9. Sequence-Only CNN/BiLSTM

The pure sequence baselines use feature set `S1`, which contains only aligned guide/target sequence representation. They do not use binding energy, epigenetic scalars, computed nucleosome features, genome, cell line, experiment ID, guide ID, target coordinate, measured flag, cleavage frequency, or labels.

Best pure sequence row:

- `sequence_cnn_unweighted / S1`
- validation AUPRC: 0.902992
- validation AUROC: 0.615003
- test AUPRC: 0.920075
- test AUROC: 0.535711
- test MCC: -0.019749
- confusion matrix: TN=0, FP=169, FN=6, TP=1527

The BiLSTM is directionally correct on validation but inverted on test:

- `sequence_bilstm_unweighted / S1`
- validation AUROC: 0.646197
- test AUROC: 0.340957

This indicates poor guide-held-out transfer for this sequence-only model, not a class-column wiring bug. The score-direction diagnostics show validation is directionally correct; test inversion is best interpreted as distribution shift plus weak sequence-only generalization.

Balanced sequence variants did not improve the result. They mostly behaved as all-positive thresholded classifiers.

Sequence diagnostics:

- `outputs/sprint2/diagnostics/sequence_input_audit.csv`
- `outputs/sprint2/diagnostics/sequence_training_summary.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_unweighted_score_direction.csv`
- `outputs/sprint2/diagnostics/sequence_bilstm_unweighted_score_direction.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_unweighted_score_deciles.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_unweighted_test_decile_lift.png`

Conclusion: sequence-only neural models satisfy the Sprint 2 requirement for sequence deep-learning baselines, but they are weak under the locked measured-only guide-level split. Sequence alone does not explain the strongest signal in this dataset.

## 10. CNN + F3/F4 Late Fusion

Late fusion was evaluated as an optional Sprint 2 extension after pure sequence models were stable.

Models:

- `sequence_cnn_plus_F3_late_fusion_unweighted / S1+F3`
- `sequence_cnn_plus_F4_late_fusion_unweighted / S1+F4`

These are not pure sequence baselines. They combine raw sequence representation with the same train-only preprocessed tabular F3/F4 feature sets used by tabular baselines. F3/F4 include engineered sequence/mismatch features, so late fusion duplicates some sequence-derived information. This is acceptable as a context-fusion baseline, but it must be labeled separately.

Late-fusion results:

| model | feature_set | val_AUPRC | val_AUROC | test_AUPRC | test_AUROC | test_MCC | confusion `(TN, FP, FN, TP)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| sequence_cnn_plus_F3_late_fusion_unweighted | S1+F3 | 0.943696 | 0.717961 | 0.911630 | 0.483872 | 0.000181 | 1, 168, 9, 1524 |
| sequence_cnn_plus_F4_late_fusion_unweighted | S1+F4 | 0.946493 | 0.736364 | 0.928996 | 0.579449 | 0.000000 | 0, 169, 0, 1533 |

F4 late fusion improves ranking over pure sequence CNN:

- pure `sequence_cnn_unweighted / S1` test AUROC: 0.535711
- `sequence_cnn_plus_F4_late_fusion_unweighted / S1+F4` test AUROC: 0.579449

However, the improvement is modest. The F4 late-fusion model still predicts all test rows as positive under the official validation-selected F1 threshold. It remains far below:

- `tabular_mlp_unweighted / F3`: test AUROC 0.735388
- `xgboost_unweighted / F4`: test AUROC 0.938416

Late-fusion diagnostics:

- `outputs/sprint2/diagnostics/sequence_late_fusion_input_audit.csv`
- `outputs/sprint2/diagnostics/sequence_late_fusion_training_summary.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_score_direction.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_fixed_threshold_metrics.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_score_deciles.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_test_decile_lift.png`

Conclusion: CNN + F4 late fusion mildly improves sequence ranking, but not enough to justify expanding Sprint 2 into BiLSTM fusion, balanced fusion, or architecture tuning. Sprint 2 modeling is frozen after this slice.

## 11. Decile And Ranking Diagnostics

Score deciles help distinguish real ranking from high prevalence.

Selected test decile positive rates:

| model | feature_set | decile 1 positive_rate | decile 5 positive_rate | decile 10 positive_rate |
| --- | --- | ---: | ---: | ---: |
| xgboost_unweighted | F4 | 1.000000 | 1.000000 | 0.438596 |
| tabular_mlp_unweighted | F3 | 1.000000 | 0.929825 | 0.666667 |
| sequence_cnn_unweighted | S1 | 0.947059 | 0.883041 | 0.959064 |
| sequence_cnn_plus_F4_late_fusion_unweighted | S1+F4 | 0.970588 | 0.923977 | 0.853801 |

XGBoost F4 has the clearest ranking structure: top deciles are almost entirely positive and the bottom decile is substantially enriched for negatives. MLP F3 shows meaningful lift, but less than XGBoost. Sequence CNN does not produce a coherent monotonic ranking. CNN+F4 late fusion improves over pure sequence but remains weak.

Relevant decile plots:

- `outputs/sprint2/diagnostics/xgboost_unweighted_test_decile_lift.png`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_test_decile_lift.png`
- `outputs/sprint2/diagnostics/sequence_cnn_unweighted_test_decile_lift.png`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_test_decile_lift.png`

## 12. Per-Genome And Per-Guide Limitations

Aggregate metrics must be interpreted alongside genome and guide composition.

Selected test per-genome AUROC:

| model | feature_set | hg19 AUROC | hg38 AUROC | mm10 AUROC |
| --- | --- | ---: | ---: | ---: |
| xgboost_unweighted | F4 | 0.769625 | 0.561654 | 0.611486 |
| tabular_mlp_unweighted | F3 | 0.491777 | 0.451693 | 0.743243 |
| sequence_cnn_unweighted | S1 | 0.730724 | 0.523828 | 0.511261 |
| sequence_cnn_plus_F4_late_fusion_unweighted | S1+F4 | 0.145399 | 0.525651 | 0.545045 |

The test set is dominated by hg19 rows, but hg19 has only 15 negatives among 1446 rows. The aggregate AUROC can therefore be shaped by cross-genome and cross-guide score offsets. This does not invalidate the locked split, but it limits claims about uniform within-guide or within-genome ranking.

Guide-level concentration also matters. In the test set, guide `9251` alone contributes 176 rows and 80 negatives. Only 8 of 29 test guides contain both classes. Reports should not overinterpret small metric differences as stable biological generalization.

Relevant diagnostics:

- `outputs/sprint2/diagnostics/xgboost_unweighted_per_genome_metrics.csv`
- `outputs/sprint2/diagnostics/xgboost_unweighted_test_per_guide_metrics.csv`
- `outputs/sprint2/diagnostics/tabular_mlp_unweighted_per_genome_metrics.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_unweighted_per_genome_metrics.csv`
- `outputs/sprint2/diagnostics/sequence_cnn_plus_F4_late_fusion_unweighted_per_genome_metrics.csv`

## 13. Final Baseline Ranking

Strongest overall non-graph baseline:

1. `xgboost_unweighted / F4`
2. `xgboost_unweighted / F3`
3. `xgboost_balanced_train_weights / F3-F4`

Best neural tabular baseline:

1. `tabular_mlp_unweighted / F3`
2. `tabular_mlp_unweighted / F4`

Best sequence/fusion neural result:

1. `sequence_cnn_plus_F4_late_fusion_unweighted / S1+F4`
2. `sequence_cnn_unweighted / S1`

Weak/debug baselines:

- Logistic Regression.
- Pure sequence BiLSTM.
- Sequence balanced-weight variants.
- CNN+F3 late fusion.

The final Sprint 2 modeling decision is:

> Freeze Sprint 2 modeling. XGBoost F3/F4 is the main non-graph benchmark for future GNN comparison. CNN + F4 late fusion was tested and produced only a modest ranking improvement over pure sequence CNN, so BiLSTM fusion, balanced fusion, and further neural architecture tuning are deferred.

## 14. Implications For Future GNN Work

The future GNN claim should not be:

> Classical ML is bad, therefore GNN.

That would be incorrect because XGBoost with context-rich features is strong.

The correct future comparison is:

> Strong non-graph baselines, especially XGBoost with F3/F4 context features, establish a serious comparison point. A GNN must improve over these under the same guide-level split, label policy, measured-only evaluation protocol, and metric policy to justify graph structure.

Future graph experiments should therefore compare against:

- `xgboost_unweighted / F4`
- `xgboost_unweighted / F3`
- `tabular_mlp_unweighted / F3`
- the best sequence/fusion neural result as secondary context

The same locked split and test-set rules should be reused. Test diagnostics from Sprint 2 must not be used to tune future GNN architectures or thresholds.

## 15. Reproducibility Commands

The commands used by Sprint 2 are documented in `docs/COMMANDS.md`.

Core commands:

```bash
uv run python scripts/build_splits.py --config configs/data/mak2022.yaml
uv run python scripts/build_features.py --config configs/data/mak2022.yaml
uv run python scripts/train.py --config configs/experiments/baseline_logistic_regression.yaml
uv run python scripts/train.py --config configs/experiments/baseline_xgboost.yaml
uv run python scripts/train.py --config configs/experiments/baseline_mlp.yaml
uv run python scripts/train.py --config configs/experiments/sequence_cnn_bilstm.yaml
uv run python scripts/train.py --config configs/experiments/sequence_cnn_late_fusion.yaml
uv run pytest -q
```

Final validation after the late-fusion slice:

```text
uv run ruff check scripts src tests
uv run pytest -q
```

The final test suite passed with 41 tests.
