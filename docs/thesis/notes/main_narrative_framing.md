# Main Narrative Framing

## Working Title

Context-Aware Graph Neural Networks for CRISPR-Cas9 Off-Target Prediction: Ranking Performance and Rare Negative Recognition

## Core Thesis

This project should not be framed as a simple "GNN beats all baselines" story. The stronger and more defensible thesis is that context-aware graph modelling revealed where GNNs help, where they do not, and why evaluation must separate ranking performance from operating-point behavior.

The strongest non-graph baseline, XGBoost F4, remains the primary AUPRC bar under the frozen guide-disjoint measured-only contract. The strongest context-aware GNN variants are competitive in AUPRC and narrow the gap, but Sprint 9 robustness does not support a robust AUPRC superiority claim over XGBoost F4.

The major positive contribution is operating-point behavior: Graph C context-aware GATv2 variants, especially the target-context encoder line, substantially improved rare negative-class recognition at validation-locked thresholds. This appears in MCC, macro-F1, specificity, and true-negative recovery.

## Why Rare Negative Recognition Matters

The measured-only headline universe is positive-heavy, with positive prevalence around 0.9007. In this setting, high AUPRC can coexist with poor negative-class recognition if a model mostly predicts positives. Therefore, threshold-free ranking metrics and validation-locked operating-point metrics answer different scientific questions:

1. AUPRC/AUROC: how well the model ranks measured candidate sites.
2. MCC, macro-F1, specificity, and confusion matrix: whether the chosen operating point actually recognizes rare measured negatives.

Most CRISPR off-target literature emphasizes ranking metrics because many studies operate on genome-wide, highly negative-heavy candidate pools. This project's measured-only benchmark is different. Negative examples are fewer, guide-concentrated, and practically important; recovering them is not a side note.

## Claim Language

Preferred claim:

> The strongest GNN contribution is not a robust AUPRC win over XGBoost, but a shift in operating-point behavior: target-context-aware GATv2 models recover substantially more rare measured negatives while remaining competitive in AUPRC.

Avoid:

> The proposed GNN is state-of-the-art.

Avoid:

> The GNN robustly beats XGBoost.

Use instead:

> Under the frozen guide-disjoint measured-only contract, context-aware GNNs did not robustly surpass XGBoost F4 on primary AUPRC, but they achieved stronger rare-negative operating-point behavior.

## Evidence Chain

1. Sprint 4 established Graph A/B/C GCN baselines and kept the Sprint 2/3 evaluation contract frozen.
2. Sprint 5 showed that Graph A's strongest feature-family signal was S5F2 energy.
3. Sprint 6 showed that loss/sampling changes did not solve the remaining limitation; weighted BCE stayed best.
4. Sprint 7 showed that attention alone was not sufficient on Graph A, while Graph C GATv2 exposed a stronger context-aware signal.
5. Sprint 7D/7E localized the Graph C gain to target-observation context, especially experimental epigenetic context.
6. Sprint 7F/8A/8B improved the context-aware GNN line with target-context encoders, interaction modelling, and sequence/context fusion.
7. Sprint 9 showed that AUPRC gains are not robust superiority claims, while threshold-dependent negative-recognition gains are real but must be reported with seed/guide fragility.

## Reporting Stance

AUPRC remains the primary metric because it is threshold-free and literature-comparable. It should always be reported with positive prevalence and guide-disjoint split details.

MCC, macro-F1, specificity, and TN/FP/FN/TP should be treated as core secondary outcomes, not decorative diagnostics. They are central to the project's rare-negative recognition contribution.

Attention weights and context-feature importance should be described as interpretation signals, not biological causal proof.

The conclusion should embrace the two-axis result:

- Ranking: competitive context-aware GNNs, but XGBoost F4 remains the robust primary-AUPRC bar.
- Operating point: context-aware GNNs offer stronger rare-negative recognition under validation-locked thresholds.
