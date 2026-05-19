# Evaluation Protocol

This is a Sprint 0 skeleton. Fill in measured dataset counts and final split details during Sprint 1.

## Primary metric

AUPRC is the primary metric for binary off-target prediction.

## Secondary metrics

AUROC, F1, MCC, Precision@K, recall at fixed FPR, and confusion matrix.

## Split rules

- Random edge split is for debugging only.
- Final results require guide-level splits.
- Train and test guide IDs must not overlap.
- Test rows must contain only `measured=1`.
- Validation should prefer `measured=1`.
- Feature normalization must be fit on training data only.

## Comparison rules

- Compare models on the same dataset, label scheme, split, and metrics.
- Report graph schema and feature set for every run.
- Paper reproduction and binary classification are separate reporting tracks.
