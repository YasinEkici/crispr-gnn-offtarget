# Evaluation Protocol

This document defines the project evaluation rules after the Sprint 1 dataset and label audit.

## Primary Metric

AUPRC is the primary metric for binary off-target prediction.

Reason: the task is imbalanced and positive-class retrieval is more important than overall ranking performance alone.

## Secondary Metrics

Use secondary metrics for additional interpretation:

- AUROC
- F1
- MCC
- Precision@K
- Recall at fixed FPR
- Confusion matrix

For a future Mak paper-comparison track using Scheme B / CA, use the paper-aligned regression metrics only when the target, split, dataset, and setup match that comparison.

## Split Rules

Random edge split:

- Debug only.
- Must not be reported as final model performance.

Final split:

- Guide-level split.
- Train and test guide IDs must not overlap.
- Test rows must contain only `measured=1`.
- Validation should prefer `measured=1`.
- Training may include `measured=0` rows only as optional putative negatives with a label-noise caveat.
- Report measured composition for every split.

The audited dataset has 154 unique sgRNAs, with highly uneven target counts per guide. Later split code must account for large guides so that a few high-count guides do not dominate validation or test results.

## Label Rules

Primary binary label:

```text
Scheme A: cleavage_freq > 1e-5
```

Sensitivity label:

```text
Scheme C: cleavage_freq > 1e-3
```

Paper-comparison label:

```text
Scheme B: reproduced Mak CA / Box-Cox target
```

Scheme B is deferred and must not be treated as directly available from the raw dataset because the transformed `CA` column is absent.

## Leakage Rules

- Do not allow the same guide to appear in both train and test for guide-level evaluation.
- Feature normalization and imputation statistics must be fit on training data only.
- Similarity edges must be built without using labels.
- Context or graph edges must not encode target labels.
- Do not use test labels to build target-target or sgRNA-sgRNA similarity.
- If transductive graph access is used, document exactly which unlabeled test-time information is visible during training.

## Reporting Rules

Every model report must state:

- Dataset version/source.
- Label scheme.
- Split type.
- Train/validation/test measured composition.
- Primary metric AUPRC.
- Secondary metrics.
- Feature set.
- Graph schema, if any.
- Whether `measured=0` rows were used in training.
- Whether `experiment_id=18` was excluded or reported separately.
- Per-genome breakdown when claiming generalization beyond one genome.

## Comparison Rules

- Compare models only on the same dataset, label scheme, split, feature policy, and metrics.
- Do not compare guide-level results to random-split results as if they answer the same question.
- Do not claim reproduction of Mak et al. unless dataset, target, split, metrics, and architecture match the paper's setup.
- Keep paper reproduction and binary classification as separate reporting tracks.
