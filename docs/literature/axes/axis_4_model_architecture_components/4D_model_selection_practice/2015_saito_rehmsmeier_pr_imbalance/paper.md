# The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets

## Citation

Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

## Core Idea

The paper argues that PR plots can be more informative than ROC plots for imbalanced binary classification because ROC views can obscure practically important precision/recall behavior under skewed class distributions.

## Sprint 9 Relevance

The CRISPR-GNN measured-only test universe has positive prevalence around 0.9007 and rare negatives concentrated in a few guide clusters. Sprint 9 uses this reference to support the emphasis on AUPRC as the primary ranking metric and on negative-class threshold metrics as separate operating-point evidence.

The Sprint 9 report also follows the important wording correction: prevalence is the no-skill PR baseline, not a mathematical AUPRC floor.

## Project Adaptation

Sprint 9 reports:

- AUPRC as the primary threshold-free metric;
- AUROC as secondary context;
- MCC, specificity, and macro-F1 at frozen validation-selected thresholds;
- guide-cluster bootstrap intervals to account for guide-level dependence.

## Claim Boundary

This paper supports PR-focused evaluation under imbalance. It does not imply that higher single-run AUPRC is robust; that conclusion still requires paired guide-cluster bootstrap and seed-sensitivity evidence.
