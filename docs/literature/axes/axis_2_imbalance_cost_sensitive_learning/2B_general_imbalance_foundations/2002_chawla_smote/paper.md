# SMOTE: Synthetic Minority Over-sampling Technique

## Citation

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357. https://doi.org/10.1613/jair.953

## Core Idea

SMOTE is a classic data-level imbalance method. It creates synthetic minority-class examples by interpolating between existing minority examples in feature space, often combined with majority-class undersampling.

## Project Relevance

This paper is useful as an imbalance-method reference, but it is **not** a method the project adopted. Sprint 6 compared loss/sampling objectives under the frozen graph/evaluation contract; it did not create synthetic biological candidates, synthetic sequence pairs, or synthetic graph edges.

The reason is practical and scientific:

- CRISPR off-target examples are structured biological objects, not generic continuous tabular vectors.
- Candidate edges connect real sgRNA and target/context nodes; interpolating edge features can produce feature vectors without a valid sequence, genomic coordinate, chromatin context, or graph interpretation.
- Synthetic candidates would complicate leakage controls and graph artifact provenance.
- The project already had validation-only thresholding, weighted BCE, measured-only balanced sampling, and guide-disjoint evaluation as safer imbalance controls.

## Sprint 6/9 Claim Boundary

SMOTE can be cited when explaining the broader imbalance toolbox and why data-level synthetic oversampling was not selected for this graph/sequence setting. It should not be cited as evidence that SMOTE is appropriate for CRISPR-GNN graph artifacts without additional biological validity checks.

## Practical Takeaway

For this project, SMOTE remains a background reference: it motivates the distinction between data-level resampling and loss/threshold-level imbalance handling, while supporting the decision to avoid synthetic graph/sequence examples in the frozen evaluation workflow.
