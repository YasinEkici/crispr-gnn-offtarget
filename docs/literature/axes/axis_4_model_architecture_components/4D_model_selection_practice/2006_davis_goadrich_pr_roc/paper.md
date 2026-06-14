# The Relationship Between Precision-Recall and ROC Curves

## Citation

Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of ICML 2006*, 233-240. https://doi.org/10.1145/1143844.1143874

## Core Idea

The paper relates ROC and precision-recall spaces and explains why precision-recall views can be more informative for highly skewed binary classification problems. ROC and PR analyses are connected, but they emphasize different properties of classifier behavior under imbalance.

## Sprint 9 Relevance

Sprint 9 keeps AUPRC as the primary metric while reporting AUROC as secondary context. This reference supports that metric hierarchy: for the CRISPR measured-only headline universe, the positive prevalence is high and the rare negative class drives many operating-point concerns, so PR behavior and thresholded negative-class metrics must be interpreted explicitly.

## Project Adaptation

The project does not replace AUROC with AUPRC everywhere. It reports both, but uses AUPRC as the primary threshold-free metric and MCC/specificity/macro-F1 as frozen-threshold operating-point metrics. Sprint 9 then asks whether AUPRC gains and operating-point shifts survive guide-cluster uncertainty and seed sensitivity.

## Claim Boundary

This reference supports metric framing, not model superiority. It does not license treating AUPRC gains as robust unless paired guide-cluster intervals and multi-seed summaries support that claim.
