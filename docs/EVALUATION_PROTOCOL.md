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
- Rows with NaN `cleavage_freq` must be excluded from supervised train/validation/test label generation.
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

## Sprint 3 Graph Visibility Rules

The primary graph policy is strict-inductive and preserves the locked Sprint 2 main universe:

- Candidate-pair labels and split membership inherit `sprint2_main_seed42`; train, validation, and test remain measured-only and exclude `experiment_id=18`.
- Typed candidate-relation artifacts may contain every split with explicit masks; any later model loader must restrict supervised training edges to `split=train`.
- Training graph views contain only training candidate relations and training-only auxiliary similarity relations.
- Graph B validation/test guide-similarity queries may connect held-out guides only to training guides using label-free guide sequence.
- Graph C validation/test context-similarity queries may connect held-out observations only to training observations after train-fitted imputation and scaling.
- Auxiliary relation artifacts store visibility fragments: evaluation loads the training fragment together with the relevant validation or test fragment, never validation and test together.
- Validation or test labels, performance diagnostics, and model scores never affect auxiliary topology.
- Any later transductive sensitivity must be separately named, justified, and cannot replace strict-inductive primary reporting.

Graph A represents physical target sites with row-varying context on candidate edges. Graph C represents context observations keyed per source row. Their comparison must acknowledge that both topology and target semantics differ.

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

Every sprint that trains or evaluates a model must also generate report-ready visual outputs:

- AUPRC comparison with positive prevalence and the strongest comparable prior baseline; graph models must include `xgboost_unweighted / F4`.
- Precision-recall and ROC curves.
- Training/history curves for iterative neural models.
- Score-distribution or decile-lift diagnostics.
- Fixed-threshold diagnostics, with thresholds selected using validation data only.
- Per-guide and per-genome diagnostics appropriate to the reported claim.
- Model-specific interpretation or artifact-sanity figures when relevant, such as graph-view inspection, feature ablation, or attention summaries.
- Position-level perturbation/sensitivity views for stable sequence-bearing neural models where aligned guide-target sequence is an explicit input.
- Feature-distribution and feature-contribution summaries for epigenetic/context ablations, using a technically appropriate attribution method and clearly distinguishing Scheme A analysis from Mak paper reproduction.
- Positive-retrieval and across-guide variability summaries for imbalance-intervention comparisons.

All headline figures must use the locked guide-level main protocol. Any random-edge or exploratory diagnostic must be visibly marked debug-only. Test-set visualizations are reporting artifacts only and must not be used to select graph schemas, features, thresholds, hyperparameters, epochs, or model variants.

Interpretation figures must be treated conservatively:

- Perturbation/sensitivity maps adapt the CRISPR-Net position-level interpretation approach; they describe model response to altered inputs, not causal cleavage mechanisms.
- Context-feature contribution views adapt Mak et al.'s correlation/distribution and SHAP-style feature analysis; Scheme A classification under the locked split is not a reproduction of Mak's CA regression experiment.
- Attention weights are model signals, not biological explanations unless supported by separate evidence.
- Imbalance visualizations must use the unchanged main evaluation population. Data-level resampling distributions may be visualized for training audits only and may not replace evaluation on the locked test universe.

## Threshold Sensitivity At High Positive Prevalence

When the test positive prevalence is high (approximately 90% in the Mak 2022
measured-only universe), threshold-dependent metrics behave differently than in
the typical low-prevalence off-target prediction setting described in Gao et al.
(2020):

- **F1** is dominated by the positive class and will maximize at low thresholds
  that classify nearly everything as positive.
- **MCC** is highly sensitive to the number of true negatives; a low threshold
  producing few negative predictions will yield a low MCC even if AUPRC is high.
- **AUPRC** is threshold-free and remains the authoritative comparison metric
  regardless of prevalence. It is the only metric suitable for cross-model
  comparison at this positive rate.

Threshold-based metrics (F1, MCC, confusion matrices) are included as secondary
interpretation outputs and must not be used to rank models or to drive schema,
feature, hyperparameter, or threshold decisions. This is consistent with the
no-test-tuning contract. See `docs/DECISIONS.md` for the single-split CV
rationale and the Graph B MCC anomaly explanation.

## Comparison Rules

- Compare models only on the same dataset, label scheme, split, feature policy, and metrics.
- Do not compare guide-level results to random-split results as if they answer the same question.
- Do not claim reproduction of Mak et al. unless dataset, target, split, metrics, and architecture match the paper's setup.
- Keep paper reproduction and binary classification as separate reporting tracks.
