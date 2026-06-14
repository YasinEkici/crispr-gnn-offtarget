# No Unbiased Estimator of the Variance of K-Fold Cross-Validation

## Citation

Bengio, Y., & Grandvalet, Y. (2004). No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *Journal of Machine Learning Research*, 5, 1089-1105. https://www.jmlr.org/papers/v5/grandvalet04a.html

## Core Idea

The paper shows that there is no universal unbiased estimator of the variance of k-fold cross-validation under all distributions. More broadly, it is a caution against treating convenient resampling summaries as exact uncertainty estimates for generalization performance.

## Sprint 9 Relevance

Sprint 9 uses predeclared multi-seed fixed-split retraining for selected headline models. The result is useful, but it measures a narrow target: training stochasticity under the same split, labels, features, thresholds, and model family.

This paper supports the Sprint 9 claim boundary:

- multi-seed mean/std/min/max are descriptive sensitivity summaries;
- seed spread is not an external generalization confidence interval;
- seed variance and guide-cluster bootstrap variance target different sources of uncertainty;
- no best-seed selection is allowed.

## Project Adaptation

The original paper concerns k-fold cross-validation variance, not fixed-split multi-seed neural retraining. Sprint 9 uses it as a methodological warning: variance summaries from reused or fixed evaluation structures must be reported with precise scope.

## Claim Boundary

Do not cite this paper as saying multi-seed analysis is invalid. It supports disciplined wording: multi-seed results quantify stochastic training sensitivity conditional on the fixed split, not population-level model superiority.
