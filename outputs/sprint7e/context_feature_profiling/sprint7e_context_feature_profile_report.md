# Sprint 7E Target-Observation Feature Profiling Report

## Contract

- Graph schema: `graph_c_context_observation`.
- Label/split: frozen `scheme_a` / `sprint2_main_seed42`.
- Visibility policy: `strict_inductive_primary`.
- This slice performs feature-family profiling only. It does not train models and does not select runs from test performance.

## Family Counts

| target_context_family | feature_columns |
| --- | --- |
| target_sequence_one_hot | 115 |
| experimental_epigenetic | 6 |
| computed_nucleosome_aggregates | 78 |
| computed_nucleosome_missingness | 13 |

## Group Summary

| target_context_family | feature_columns | rows | missing_values_after_preprocessing | feature_abs_mean | feature_abs_std | feature_mean | feature_std | nonzero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_sequence_one_hot | 115 | 11446 | 0 | 0.200000 | 0.400000 | 0.200000 | 0.400000 | 0.200000 |
| experimental_epigenetic | 6 | 11446 | 0 | 0.396321 | 1.004736 | 0.043237 | 1.079211 | 1.000000 |
| computed_nucleosome_aggregates | 78 | 11446 | 0 | 0.693340 | 0.686234 | 0.011833 | 0.975448 | 0.974359 |
| computed_nucleosome_missingness | 13 | 11446 | 0 | 0.508687 | 0.776418 | -0.045163 | 0.927118 | 1.000000 |

## Split/Label Distribution Summary

| split | label | target_context_family | rows | feature_columns | feature_abs_mean | feature_mean | feature_std | feature_min | feature_max | nonzero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0 | target_sequence_one_hot | 169 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| test | 0 | experimental_epigenetic | 169 | 6 | 0.291067 | -0.159356 | 0.563729 | -0.475774 | 5.873346 | 1.000000 |
| test | 0 | computed_nucleosome_aggregates | 169 | 78 | 0.675829 | 0.087342 | 0.886437 | -4.162539 | 9.733413 | 0.974359 |
| test | 0 | computed_nucleosome_missingness | 169 | 13 | 0.297427 | -0.297427 | 0.000000 | -0.297427 | -0.297427 | 1.000000 |
| test | 1 | target_sequence_one_hot | 1533 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| test | 1 | experimental_epigenetic | 1533 | 6 | 0.460385 | 0.142574 | 1.159122 | -0.475774 | 23.694244 | 1.000000 |
| test | 1 | computed_nucleosome_aggregates | 1533 | 78 | 0.711739 | 0.020341 | 0.968353 | -10.288490 | 25.915097 | 0.974359 |
| test | 1 | computed_nucleosome_missingness | 1533 | 13 | 0.383392 | -0.194777 | 0.604252 | -0.297427 | 3.362164 | 1.000000 |
| train | 0 | target_sequence_one_hot | 901 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| train | 0 | experimental_epigenetic | 901 | 6 | 0.328082 | -0.108912 | 0.768891 | -0.475774 | 20.462872 | 1.000000 |
| train | 0 | computed_nucleosome_aggregates | 901 | 78 | 0.705351 | 0.032015 | 0.952011 | -4.581245 | 12.178129 | 0.974359 |
| train | 0 | computed_nucleosome_missingness | 901 | 13 | 0.409676 | -0.163391 | 0.687424 | -0.297427 | 3.362164 | 1.000000 |
| train | 1 | target_sequence_one_hot | 7109 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| train | 1 | experimental_epigenetic | 7109 | 6 | 0.367280 | 0.013804 | 1.024753 | -0.475774 | 26.750224 | 1.000000 |
| train | 1 | computed_nucleosome_aggregates | 7109 | 78 | 0.691682 | -0.004058 | 0.991381 | -17.431105 | 89.361760 | 0.974359 |
| train | 1 | computed_nucleosome_missingness | 7109 | 13 | 0.563851 | 0.020708 | 1.031037 | -0.297427 | 3.362164 | 1.000000 |
| val | 0 | target_sequence_one_hot | 223 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| val | 0 | experimental_epigenetic | 223 | 6 | 0.304002 | -0.143282 | 0.662788 | -0.475774 | 8.414348 | 1.000000 |
| val | 0 | computed_nucleosome_aggregates | 223 | 78 | 0.711553 | 0.052602 | 1.011558 | -5.191039 | 44.954600 | 0.974359 |
| val | 0 | computed_nucleosome_missingness | 223 | 13 | 0.297427 | -0.297427 | 0.000000 | -0.297427 | -0.297427 | 1.000000 |
| val | 1 | target_sequence_one_hot | 1511 | 115 | 0.200000 | 0.200000 | 0.400000 | 0.000000 | 1.000000 | 0.200000 |
| val | 1 | experimental_epigenetic | 1511 | 6 | 0.534048 | 0.221842 | 1.412986 | -0.475774 | 31.750917 | 1.000000 |
| val | 1 | computed_nucleosome_aggregates | 1511 | 78 | 0.674586 | 0.051465 | 0.921402 | -10.288490 | 12.178129 | 0.974359 |
| val | 1 | computed_nucleosome_missingness | 1511 | 13 | 0.490114 | -0.067341 | 0.888304 | -0.297427 | 3.362164 | 1.000000 |

## Experimental Epigenetic Per-Feature Audit

This audit is diagnostic only. It checks whether the six direct experimental target-observation features have split/label distribution drift that could explain why masking this family collapses Sprint 7E rare-negative behavior. It does not introduce a new model-selection criterion.

### Per-Feature Split/Label Summary

| split | label | feature_column | source_feature_name | rows | mean | median | std | q25 | q75 | min | max | nonzero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0 | feature__epigen_ctcf | epigen_ctcf | 169 | -0.110235 | -0.307828 | 0.884117 | -0.307828 | -0.307828 | -0.307828 | 5.873346 | 1.000000 |
| test | 0 | feature__epigen_dnase | epigen_dnase | 169 | -0.117930 | -0.118560 | 0.006832 | -0.118560 | -0.118560 | -0.118560 | -0.031496 | 1.000000 |
| test | 0 | feature__epigen_rrbs | epigen_rrbs | 169 | -0.061368 | -0.061368 | 0.000000 | -0.061368 | -0.061368 | -0.061368 | -0.061368 | 1.000000 |
| test | 0 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 169 | -0.230896 | -0.239229 | 0.055996 | -0.239229 | -0.239229 | -0.239229 | 0.421149 | 1.000000 |
| test | 0 | feature__epigen_drip | epigen_drip | 169 | -0.125968 | -0.195213 | 0.632749 | -0.195213 | -0.195213 | -0.195213 | 5.655989 | 1.000000 |
| test | 0 | feature__MNase | MNase | 169 | -0.309742 | -0.475774 | 0.823981 | -0.475774 | -0.475774 | -0.475774 | 5.080552 | 1.000000 |
| test | 1 | feature__epigen_ctcf | epigen_ctcf | 1533 | 0.004357 | -0.307828 | 1.007805 | -0.307828 | -0.307828 | -0.307828 | 7.139369 | 1.000000 |
| test | 1 | feature__epigen_dnase | epigen_dnase | 1533 | 0.023871 | -0.118560 | 0.966107 | -0.118560 | -0.118560 | -0.118560 | 9.555200 | 1.000000 |
| test | 1 | feature__epigen_rrbs | epigen_rrbs | 1533 | -0.061100 | -0.061368 | 0.007408 | -0.061368 | -0.061368 | -0.061368 | 0.143875 | 1.000000 |
| test | 1 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 1533 | -0.006965 | -0.239229 | 1.040595 | -0.239229 | -0.239229 | -0.239229 | 9.472203 | 1.000000 |
| test | 1 | feature__epigen_drip | epigen_drip | 1533 | 0.128813 | -0.195213 | 1.311271 | -0.195213 | -0.195213 | -0.195213 | 5.655989 | 1.000000 |
| test | 1 | feature__MNase | MNase | 1533 | 0.766470 | 0.220904 | 1.680342 | -0.254308 | 1.191124 | -0.475774 | 23.694244 | 1.000000 |
| train | 0 | feature__epigen_ctcf | epigen_ctcf | 901 | 0.002334 | -0.307828 | 1.165767 | -0.307828 | -0.307828 | -0.307828 | 6.841481 | 1.000000 |
| train | 0 | feature__epigen_dnase | epigen_dnase | 901 | -0.072038 | -0.118560 | 0.494888 | -0.118560 | -0.118560 | -0.118560 | 6.256448 | 1.000000 |
| train | 0 | feature__epigen_rrbs | epigen_rrbs | 901 | -0.035855 | -0.061368 | 0.688194 | -0.061368 | -0.061368 | -0.061368 | 20.462872 | 1.000000 |
| train | 0 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 901 | -0.169567 | -0.239229 | 0.569139 | -0.239229 | -0.239229 | -0.239229 | 7.724145 | 1.000000 |
| train | 0 | feature__epigen_drip | epigen_drip | 901 | -0.136097 | -0.195213 | 0.527042 | -0.195213 | -0.195213 | -0.195213 | 5.655989 | 1.000000 |
| train | 0 | feature__MNase | MNase | 901 | -0.242252 | -0.475774 | 0.909207 | -0.475774 | -0.475774 | -0.475774 | 9.611514 | 1.000000 |
| train | 1 | feature__epigen_ctcf | epigen_ctcf | 7109 | -0.000296 | -0.307828 | 0.976984 | -0.307828 | -0.307828 | -0.307828 | 7.139369 | 1.000000 |
| train | 1 | feature__epigen_dnase | epigen_dnase | 7109 | 0.009130 | -0.118560 | 1.046403 | -0.118560 | -0.118560 | -0.118560 | 9.555200 | 1.000000 |
| train | 1 | feature__epigen_rrbs | epigen_rrbs | 7109 | 0.004544 | -0.061368 | 1.032730 | -0.061368 | -0.061368 | -0.061368 | 20.462872 | 1.000000 |
| train | 1 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 7109 | 0.021491 | -0.239229 | 1.039991 | -0.239229 | -0.239229 | -0.239229 | 9.472203 | 1.000000 |
| train | 1 | feature__epigen_drip | epigen_drip | 7109 | 0.017249 | -0.195213 | 1.043499 | -0.195213 | -0.195213 | -0.195213 | 5.655989 | 1.000000 |
| train | 1 | feature__MNase | MNase | 7109 | 0.030703 | -0.269550 | 1.006772 | -0.475774 | 0.167106 | -0.475774 | 26.750224 | 1.000000 |
| val | 0 | feature__epigen_ctcf | epigen_ctcf | 223 | -0.025636 | -0.307828 | 1.151893 | -0.307828 | -0.307828 | -0.307828 | 7.139369 | 1.000000 |
| val | 0 | feature__epigen_dnase | epigen_dnase | 223 | -0.118560 | -0.118560 | 0.000000 | -0.118560 | -0.118560 | -0.118560 | -0.118560 | 1.000000 |
| val | 0 | feature__epigen_rrbs | epigen_rrbs | 223 | -0.061368 | -0.061368 | 0.000000 | -0.061368 | -0.061368 | -0.061368 | -0.061368 | 1.000000 |
| val | 0 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 223 | -0.162190 | -0.239229 | 0.618444 | -0.239229 | -0.239229 | -0.239229 | 6.034356 | 1.000000 |
| val | 0 | feature__epigen_drip | epigen_drip | 223 | -0.193193 | -0.195213 | 0.016262 | -0.195213 | -0.195213 | -0.195213 | -0.007975 | 1.000000 |
| val | 0 | feature__MNase | MNase | 223 | -0.298745 | -0.475774 | 0.936992 | -0.475774 | -0.475774 | -0.475774 | 8.414348 | 1.000000 |
| val | 1 | feature__epigen_ctcf | epigen_ctcf | 1511 | -0.025440 | -0.307828 | 0.974505 | -0.307828 | -0.307828 | -0.307828 | 7.139369 | 1.000000 |
| val | 1 | feature__epigen_dnase | epigen_dnase | 1511 | 0.025836 | -0.118560 | 0.990948 | -0.118560 | -0.118560 | -0.118560 | 9.555200 | 1.000000 |
| val | 1 | feature__epigen_rrbs | epigen_rrbs | 1511 | -0.019939 | -0.061368 | 0.795118 | -0.061368 | -0.061368 | -0.061368 | 20.462872 | 1.000000 |
| val | 1 | feature__epigen_h3k4me3 | epigen_h3k4me3 | 1511 | -0.034170 | -0.239229 | 0.997624 | -0.239229 | -0.239229 | -0.239229 | 9.472203 | 1.000000 |
| val | 1 | feature__epigen_drip | epigen_drip | 1511 | 0.168293 | -0.195213 | 1.382503 | -0.195213 | -0.195213 | -0.195213 | 5.655989 | 1.000000 |
| val | 1 | feature__MNase | MNase | 1511 | 1.216470 | 0.404712 | 2.300661 | -0.131470 | 2.043435 | -0.475774 | 31.750917 | 1.000000 |

### Positive-Negative Standardized Mean Differences

| split | feature_column | source_feature_name | rows_negative | rows_positive | mean_negative | mean_positive | median_negative | median_positive | std_negative | std_positive | standardized_mean_difference_pos_minus_neg | abs_smd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | feature__MNase | MNase | 169 | 1533 | -0.309742 | 0.766470 | -0.475774 | 0.220904 | 0.823981 | 1.680342 | 0.813250 | 0.813250 |
| test | feature__epigen_h3k4me3 | epigen_h3k4me3 | 169 | 1533 | -0.230896 | -0.006965 | -0.239229 | -0.239229 | 0.055996 | 1.040595 | 0.303893 | 0.303893 |
| test | feature__epigen_drip | epigen_drip | 169 | 1533 | -0.125968 | 0.128813 | -0.195213 | -0.195213 | 0.632749 | 1.311271 | 0.247477 | 0.247477 |
| test | feature__epigen_dnase | epigen_dnase | 169 | 1533 | -0.117930 | 0.023871 | -0.118560 | -0.118560 | 0.006832 | 0.966107 | 0.207567 | 0.207567 |
| test | feature__epigen_ctcf | epigen_ctcf | 169 | 1533 | -0.110235 | 0.004357 | -0.307828 | -0.307828 | 0.884117 | 1.007805 | 0.120880 | 0.120880 |
| test | feature__epigen_rrbs | epigen_rrbs | 169 | 1533 | -0.061368 | -0.061100 | -0.061368 | -0.061368 | 0.000000 | 0.007408 | 0.051114 | 0.051114 |
| train | feature__MNase | MNase | 901 | 7109 | -0.242252 | 0.030703 | -0.475774 | -0.269550 | 0.909207 | 1.006772 | 0.284557 | 0.284557 |
| train | feature__epigen_h3k4me3 | epigen_h3k4me3 | 901 | 7109 | -0.169567 | 0.021491 | -0.239229 | -0.239229 | 0.569139 | 1.039991 | 0.227911 | 0.227911 |
| train | feature__epigen_drip | epigen_drip | 901 | 7109 | -0.136097 | 0.017249 | -0.195213 | -0.195213 | 0.527042 | 1.043499 | 0.185506 | 0.185506 |
| train | feature__epigen_dnase | epigen_dnase | 901 | 7109 | -0.072038 | 0.009130 | -0.118560 | -0.118560 | 0.494888 | 1.046403 | 0.099167 | 0.099167 |
| train | feature__epigen_rrbs | epigen_rrbs | 901 | 7109 | -0.035855 | 0.004544 | -0.061368 | -0.061368 | 0.688194 | 1.032730 | 0.046037 | 0.046037 |
| train | feature__epigen_ctcf | epigen_ctcf | 901 | 7109 | 0.002334 | -0.000296 | -0.307828 | -0.307828 | 1.165767 | 0.976984 | -0.002446 | 0.002446 |
| val | feature__MNase | MNase | 223 | 1511 | -0.298745 | 1.216470 | -0.475774 | 0.404712 | 0.936992 | 2.300661 | 0.862604 | 0.862604 |
| val | feature__epigen_drip | epigen_drip | 223 | 1511 | -0.193193 | 0.168293 | -0.195213 | -0.195213 | 0.016262 | 1.382503 | 0.369752 | 0.369752 |
| val | feature__epigen_dnase | epigen_dnase | 223 | 1511 | -0.118560 | 0.025836 | -0.118560 | -0.118560 | 0.000000 | 0.990948 | 0.206072 | 0.206072 |
| val | feature__epigen_h3k4me3 | epigen_h3k4me3 | 223 | 1511 | -0.162190 | -0.034170 | -0.239229 | -0.239229 | 0.618444 | 0.997624 | 0.154245 | 0.154245 |
| val | feature__epigen_rrbs | epigen_rrbs | 223 | 1511 | -0.061368 | -0.019939 | -0.061368 | -0.061368 | 0.000000 | 0.795118 | 0.073686 | 0.073686 |
| val | feature__epigen_ctcf | epigen_ctcf | 223 | 1511 | -0.025636 | -0.025440 | -0.307828 | -0.307828 | 1.151893 | 0.974505 | 0.000183 | 0.000183 |

## Run Matrix Freeze

Slice 2 remains source-family-defined: mask target sequence, experimental epigenetic, computed nucleosome aggregates, computed nucleosome missingness, and all nonsequence context. These runs are not selected or reordered from test diagnostics.

## Figure Index

- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_missingness.png`
- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_context_feature_group_distribution.png`
- `outputs/sprint7e/context_feature_profiling/figures/sprint7e_experimental_epigenetic_smd_by_split.png`
