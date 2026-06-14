# Area Under the Precision-Recall Curve: Point Estimates and Confidence Intervals

## Citation

Boyd, K., Eng, K. H., & Page, C. D. (2013). Area Under the Precision-Recall Curve: Point Estimates and Confidence Intervals. In *Machine Learning and Knowledge Discovery in Databases* (LNCS 8190, pp. 451-466). https://doi.org/10.1007/978-3-642-40994-3_29

## Core Idea

The paper studies point and interval estimation for area under the precision-recall curve. It emphasizes that PR-curve summaries have properties that differ from ROC summaries and require care when reporting uncertainty.

## Sprint 9 Relevance

AUPRC is the primary metric for the CRISPR-GNN project because the measured-only headline universe is imbalanced and negative recognition is central to interpretation. Sprint 9 uses this paper as the AUPRC uncertainty anchor: point estimates alone are not enough when candidate gains are small and the PR baseline is high.

Sprint 9 does not directly reproduce the paper's interval procedures. Instead, it uses guide-cluster bootstrap intervals because guide-level dependence is the dominant evaluation constraint in this project.

## Project Adaptation

The project reports AUPRC with:

- the measured-only positive prevalence as the no-skill PR baseline;
- finite-sample guide-cluster bootstrap intervals;
- paired guide-cluster AUPRC deltas for model comparisons;
- clear separation between AUPRC ranking and threshold-dependent MCC/specificity behavior.

## Claim Boundary

Use this reference to justify careful AUPRC interval reporting. Do not use it to imply that row-level AUPRC intervals are adequate for this guide-disjoint split; Sprint 9 resamples guides, not rows.
