# Prediction of off-target specificity and cell-specific fitness of CRISPR-Cas System using attention boosted deep learning and network-based gene feature

## Citation

Liu, Q., He, D., & Xie, L. (2019). Prediction of off-target specificity and cell-specific fitness of CRISPR-Cas System using attention boosted deep learning and network-based gene feature. *PLOS Computational Biology*, 15(10), e1007480. https://doi.org/10.1371/journal.pcbi.1007480

## Core Idea

The paper combines attention-boosted deep sequence features with cell-specific, network-derived gene features for CRISPR specificity and fitness prediction.

## Project Relevance

This is relevant to Sprint 7/8 because it shows an earlier CRISPR direction where attention and biological context features are used together. It helps frame our Graph C work as a graph/context-aware extension rather than a purely sequence-only model.

## Project Difference

The project's attention mechanism is graph attention over graph topology and edge/context features, not the same architecture or feature source as AttnToMismatch_CNN. Sprint 7-9 attention/gate summaries are interpretation artifacts only.

## Claim Boundary

Use this paper to motivate attention/context-aware modeling. Do not claim biological causality from attention weights or direct metric comparability across datasets.
