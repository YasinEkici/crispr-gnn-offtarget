**Corrected Robustness and Uncertainty Analysis for One Fixed Guide-Disjoint CRISPR Off-Target Test Split** 

_Final corrected methodological file_ 

|**Item**|**Fixed study value**|
|---|---|
|**Evaluation design**|One fixed guide-disjoint held-out test split; no split, label,<br>threshold, or model-family changes.|
|**Test set**|1702 rows from 29 guide clusters.|
|**Class distribution**|1533 positives and 169 negatives; positive prevalence =<br>1533/1702 = 0.9007.|
|**Negative-class geometry**|Negatives occur in only 9 guides; one guide contains<br>80/169 negatives = 47.3% of all negatives.|
|**Primary metric**|AUPRC/AP, observed around 0.98-0.99.|
|**Secondary metrics**|MCC and specificity at a frozen threshold selected on<br>validation data only.|



## **1. Core corrected answer** 

The uncertainty analysis should be reported as three separate, non-interchangeable summaries. The guide-cluster bootstrap summarizes test-set uncertainty under guide-level resampling for the fixed heldout split. The paired guide-cluster bootstrap summarizes model-to-model differences on common guide resamples from that same split. Predeclared multi-seed retraining summarizes sensitivity of the training pipeline to stochastic optimization and initialization on the same fixed split. These three summaries must not be merged into a single generic confidence interval because they target different sources of variation. 

The main limitation is not the total row count. For cluster-aware inference, the effective information is governed by the number and diversity of guide clusters. The full test split has only 29 guides, and the negative-class information relevant to specificity and MCC is concentrated in only 9 guides, with one guide carrying 47.3% of all negatives. Therefore, the intervals should be described as finite-sample, guidecluster-aware compatibility summaries rather than exact or near-exact 95% coverage guarantees. 

## **2. Correct procedures** 

## **2.1 Guide-cluster bootstrap CIs for one fixed trained model** 

1. Use the held-out guide as the resampling unit. Do not resample individual rows. 

2. For each bootstrap replicate, sample 29 guides with replacement from the 29 held-out guides. 

3. Include all rows belonging to each selected guide. If a guide is drawn multiple times, either duplicate its rows or use equivalent guide-level sample weights. 

4. Recompute AUPRC/AP from the fixed model scores and resampled labels using the exact same AUPRC/AP definition used for the full split. Do not switch between average precision and trapezoidal PRAUC without saying so. 

5. For specificity and MCC, apply the already frozen validation-selected threshold in every replicate. Do not retune the threshold inside the test bootstrap. 

Corrected robustness/uncertainty analysis - fixed guide-disjoint split 

6. Construct the primary interval from the empirical bootstrap quantiles, for example the 2.5th and 97.5th percentiles for a nominal 95% summary. 

7. Report diagnostic information for the bootstrap distribution: undefined replicate rate, number of unique guides per replicate, number of negative-bearing guides per replicate, whether the dominant 80-negative guide is present, histograms or density plots, and leave-one-guide-out influence values. 

## **2.2 Paired guide-cluster bootstrap for model differences** 

1. For every bootstrap replicate, draw one guide resample and evaluate all compared models on exactly that same resampled set of guides. 

2. For each metric, compute the replicate-wise difference: Delta_b = metric(Model A on replicate b) - metric(Model B on replicate b). 

3. Build the interval directly from the empirical distribution of Delta_b. This is the inferential target for the model comparison. 

4. Do not compare two separate marginal CIs by visual overlap. Overlap or non-overlap of independent CIs is not the correct test of a paired model difference. 

5. If a thresholded metric is undefined for either model in a replicate, treat the paired difference as undefined for that replicate and report the undefined paired-difference rate. 

6. Interpret the paired interval as a fixed-split, guide-resampling comparison. It preserves the covariance induced by common guide composition, but it does not remove the few-cluster limitation. 

## **2.3 Predeclared multi-seed retraining on the same fixed split** 

1. Choose the seed set before looking at the results. 

2. Train the same pipeline independently under each seed, with the same data split, labels, thresholdselection protocol, and model family. 

3. Report every seed result for AUPRC/AP, MCC, specificity, and any other declared metric. Do not select the best seed and do not hide failed or weak seeds. 

4. Summarize the seed spread descriptively, for example with mean, standard deviation, minimum, maximum, and all individual seed values. 

5. Frame this as seed/training-stochasticity sensitivity conditional on the fixed split. Do not call it a confidence interval for external performance or a substitute for test-set sampling uncertainty. 

## **3. Metric-specific and sample-size warnings** 

## **3.1 AUPRC/AP near the ceiling under high positive prevalence** 

Precision depends on the class ratio, so the no-skill PR baseline is the positive prevalence. In this split, prevalence is 1533/1702 = 0.9007. The raw headroom from the no-skill baseline to the maximum possible value 1.0 is therefore only 0.0993. AUPRC/AP values around 0.98-0.99 are strong, but they are numerically compressed near the ceiling: 0.98 uses about 79.9% of the available prevalence-to-one headroom, and 0.99 uses about 89.9% of that headroom. 

Important correction: prevalence should be called a no-skill baseline, not a mathematical floor. An adversarial or very poor ranking can fall below the no-skill baseline. Therefore, write "baseline" or "reference level," not "floor." 

Because AUPRC/AP is bounded above by 1.0 and the baseline is already high, narrow-looking raw intervals may be visually misleading. Report the raw AUPRC/AP point estimate and interval together with the prevalence baseline, and inspect whether bootstrap replicates pile up near 1.0 or show a strongly discrete/lumpy distribution. 

Corrected robustness/uncertainty analysis - fixed guide-disjoint split 

## **3.2 Specificity and MCC with rare, guide-concentrated negatives** 

Specificity depends only on negative cases, and MCC depends on all four confusion-matrix cells. Although the test set contains 1702 rows, specificity is based on 169 negatives, and those 169 negatives are present in only 9 guide clusters. One guide contains 80 negatives, or 47.3% of the negative class. Under guide-cluster bootstrapping, that dominant guide is omitted in about (28/29)^29 = 36.1% of replicates. This makes thresholded negative-class metrics highly sensitive to guide composition. 

For this reason, thresholded-metric intervals are untrustworthy as nominal 95% procedures when they are dominated by a few guides, when many replicates are undefined, when the histogram is discrete or multimodal, when endpoints pile up at bounds, or when leave-one-guide-out omissions move the metric by an amount comparable to the interval width. In those cases, report the full-split estimate, bootstrap quantiles, failure/undefined rate, guide-level influence table, and a plain-language fragility warning. 

## **3.3 BCa versus percentile with only 29 clusters** 

BCa intervals are attractive in standard bootstrap theory because they adjust for bias and skewness and are transformation-invariant under suitable regularity conditions. However, BCa depends on a jackknife acceleration estimate. With only 29 guide clusters, bounded or nonsmooth metrics, and a few highly influential negative-bearing guides, the leave-one-guide jackknife can be unstable and the acceleration term can be dominated by one or two guides. 

The defensible reporting choice is therefore pragmatic rather than theoretically universal: use the guidecluster percentile interval as the primary transparent finite-sample summary, and compute BCa only as a sensitivity check. Trust BCa only if leave-one-guide jackknife diagnostics are finite, smooth, and not dominated by one or two influential guides. If percentile and BCa intervals disagree materially, report both and state that interval construction is unstable. 

## **3.4 Why overlapping independent CIs are not a model-comparison test** 

The scientific target for a model comparison is the uncertainty of the difference, not the visual overlap of two marginal intervals. Two 95% marginal CIs can overlap while a 95% CI for the difference excludes zero, and non-overlap can be overly conservative in other settings. Because the same guides are used to evaluate both models, the comparison is paired. The paired guide-bootstrap difference uses the same guide resample for both models and therefore preserves the covariance due to common guide composition. 

## **3.5 What single-split multi-seed variance captures and does not capture** 

Multi-seed retraining captures sensitivity to stochastic parts of the training pipeline: initialization, minibatch order, optimizer nondeterminism, dropout, data-order effects, and similar sources. It does not capture uncertainty from which guides happened to be in the fixed held-out split, and it does not estimate how results would change under a different guide-disjoint split or a full resampling/cross-validation protocol. Therefore, seed spread and guide-bootstrap spread should be reported side by side, not treated as substitutes. 

## **4. Claims licensed by the corrected analysis** 

## **4.1 If the paired-difference interval excludes zero** 

You may claim evidence of a directional model difference for the specific uncertainty target analyzed: guide-level resampling of the fixed held-out split, conditional on the trained models and frozen threshold protocol. The claim should not automatically be generalized to future retrainings, new guide populations, or different splits unless those sources of variation are also analyzed. 

Corrected robustness/uncertainty analysis - fixed guide-disjoint split 

## **4.2 If the paired-difference interval includes zero** 

You may say that the fixed-split guide-bootstrap analysis did not provide clear evidence of superiority at the stated confidence level, and that the observed data are compatible with no difference as well as with effect sizes inside the interval. You may not claim that the models are equivalent, identical, or non-inferior. Equivalence or non-inferiority requires a prespecified practically acceptable margin and an interval or test procedure designed for that margin, such as TOST/equivalence testing. 

## **5. Minimum reporting checklist** 

|**Item**|**What to report**|
|---|---|
|**Full-split point estimates**|Report AUPRC/AP, MCC, specificity, threshold, prevalence,<br>number of rows, number of guides, and class counts.|
|**Cluster-bootstrap design**|State that guides, not rows, were resampled; include number<br>of replicates and the exact metric definitions.|
|**Degenerate replicates**|Report undefined/failure rates for AUPRC/AP, specificity,<br>MCC, and paired differences.|
|**Guide composition diagnostics**|Report unique guides per replicate, negative-bearing guides<br>per replicate, and inclusion frequency/effect of the dominant<br>negative guide.|
|**Bootstrap distribution**|Provide histogram/density or quantile summaries; flag<br>bounded, lumpy, multimodal, or endpoint-piled distributions.|
|**Leave-one-guide influence**|Report the largest leave-one-guide-out metric changes,<br>especially for negative-bearing guides.|
|**Model comparison**|Report paired-difference intervals; do not use overlap of<br>marginal CIs.|
|**Multi-seed retraining**|Report all predeclared seeds and summarize them<br>descriptively; do not best-seed select.|
|**Claim discipline**|Use finite-sample compatibility language; avoid exact<br>coverage, broad generalization, or equivalence claims without<br>an equivalence margin.|



## **6. Red-flag statements to avoid** 

- Do not write: "AUPRC has a floor of 0.9007." Correct wording: "The no-skill PR baseline is 0.9007." 

- Do not write: "The 95% bootstrap CI has exact 95% coverage." Correct wording: "A finite-sample guide-cluster bootstrap compatibility interval was computed." 

- Do not write: "1702 rows make the CI reliable" without noting the 29-guide and 9-negative-bearingguide bottleneck. 

- Do not write: "Overlapping independent CIs show no difference" or "non-overlap proves a difference." Use the paired-difference CI. 

- Do not write: "The models are equivalent" when the paired-difference CI includes zero. Equivalence requires a prespecified equivalence margin. 

- Do not write: "Multi-seed variance estimates generalization uncertainty." It estimates trainingstochasticity sensitivity conditional on the fixed split. 

- Do not write: "BCa is automatically better here." With 29 clusters and influential guides, BCa can be unstable and should be treated as sensitivity evidence only. 

## **7. Implementation pseudocode** 

```
for b in 1..B:
```

```
    sampled_guides = sample_with_replacement(test_guides, size=29)
```

Corrected robustness/uncertainty analysis - fixed guide-disjoint split 

```
    replicate_rows = concatenate_rows_for(sampled_guides)
```

```
    # Single-model CI
```

```
    auprc_b = metric_auprc(y[replicate_rows], score_model[replicate_rows])
    pred_b  = score_model[replicate_rows] >= frozen_validation_threshold
    spec_b  = specificity(y[replicate_rows], pred_b)  # undefined if no negatives
    mcc_b   = mcc(y[replicate_rows], pred_b)          # undefined if denominator collapses
```

```
    # Paired model difference
```

```
    metric_A_b = metric(y[replicate_rows], score_A[replicate_rows], threshold_A_if_needed)
    metric_B_b = metric(y[replicate_rows], score_B[replicate_rows], threshold_B_if_needed)
    delta_b = metric_A_b - metric_B_b if both are defined else undefined
```

```
Report point estimate on the original full test split.
```

```
Report bootstrap quantiles, undefined rates, distribution diagnostics, and guide-level influence
summaries.
```

## **8. References** 

1. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1-26. 

2. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC. 

3. Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and Their Application. Cambridge University Press. 

4. Carpenter, J., & Bithell, J. (2000). Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians. Statistics in Medicine, 19(9), 1141-1164. 

5. Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. Journal of the Royal Statistical Society: Series B, 69(3), 369-390. 

6. Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317-372. 

7. Boyd, K., Eng, K. H., & Page, C. D. (2013). Area under the precision-recall curve: Point estimates and confidence intervals. In Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2013. 

8. Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. Proceedings of ICML 2006, 233-240. 

9. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432. 

10. Williams, C. K. I. (2021). The effect of class imbalance on precision-recall curves. Neural Computation, 33(4), 853-857. 

11. Schenker, N., & Gentleman, J. F. (2001). On judging the significance of differences by examining the overlap between confidence intervals. The American Statistician, 55(3), 182-186. 

12. Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of k-fold cross-validation. Journal of Machine Learning Research, 5, 1089-1105. 

13. Bethard, S. (2022). We need to talk about random seeds. arXiv:2210.13393. 

14. Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. Journal of Pharmacokinetics and Biopharmaceutics, 15(6), 657-680. 

15. Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. Social Psychological and Personality Science, 8(4), 355-362. 

Corrected robustness/uncertainty analysis - fixed guide-disjoint split 

