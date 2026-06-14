# Deep learning improves the ability of sgRNA off-target propensity prediction

## Citation

Liu, Q., Cheng, X., Liu, G., Li, B., & Liu, X. (2020). Deep learning improves the ability of sgRNA off-target propensity prediction. *BMC Bioinformatics*, 21, 51. https://doi.org/10.1186/s12859-020-3395-z

## Core Idea

CnnCrispr learns sgRNA-DNA sequence representations with a GloVe-style embedding and a CNN/BiLSTM deep model for off-target propensity prediction.

## Project Relevance

This is a useful sequence-based baseline reference for explaining why CRISPR off-target papers commonly report AUROC/AUPRC-like metrics and leave-one-guide style evaluations. It also motivates why sequence-only models remain strong baselines even when graph/context features are explored.

## Project Difference

The local project does not reproduce CnnCrispr. It evaluates graph/context models under a measured-only, positive-heavy, guide-disjoint split, so raw AUPRC values are not directly comparable to CnnCrispr's negative-heavy candidate-pool evaluations.

## Claim Boundary

Use this paper for sequence-baseline positioning and metric terminology. Avoid direct performance claims without prevalence, split, and dataset caveats.
