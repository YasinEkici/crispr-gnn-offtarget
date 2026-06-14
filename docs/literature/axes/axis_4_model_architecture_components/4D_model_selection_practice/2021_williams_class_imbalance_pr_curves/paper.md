# The Effect of Class Imbalance on Precision-Recall Curves

## Citation

Williams, C. K. I. (2021). The effect of class imbalance on precision-recall curves. *Neural Computation*, 33(4), 853-857. https://doi.org/10.1162/neco_a_01362

## Core Idea

Williams analyzes how precision-recall curves depend on the class ratio in the evaluation set. Precision is prevalence-dependent, so changing the positive-to-negative ratio changes the PR curve even when classifier operating characteristics are otherwise comparable.

## Project Relevance

Sprint 9 needs this framing because the measured-only test set is strongly positive-heavy: 1533 positives and 169 negatives, positive prevalence 0.9007. The no-skill PR reference is therefore high, and raw AUPRC values should be reported with prevalence rather than compared naively to negative-heavy genome-wide retrieval papers.

## Project Difference

The paper is a metric note, not a CRISPR-specific model benchmark. It supports interpretation of AUPRC/AP under class imbalance, not any claim that one local model architecture is biologically superior.

## Claim Boundary

Use this paper to justify prevalence-aware AUPRC interpretation and optional baseline-adjusted PR headroom. Do not replace raw AUPRC/AP with a nonstandard normalized metric in headline reporting.
