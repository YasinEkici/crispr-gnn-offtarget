# Interpretable CRISPR/Cas9 Off-Target Activities With Mismatches and Indels Prediction Using BERT

## Citation

Luo, Y., Chen, Y., Xie, H., Zhu, W., & Zhang, G. (2024). Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT. *Computers in Biology and Medicine*, 169, 107932. https://doi.org/10.1016/j.compbiomed.2024.107932

## Core Idea

CRISPR-BERT applies a transformer/BERT-style sequence model to CRISPR-Cas9 off-target prediction with mismatches and indels. The paper is relevant to the broader question of whether learned sequence representations can outperform manually engineered mismatch encodings.

## Project Relevance

This is a future-work reference for stronger sequence encoders. It is especially relevant if the project later compares the current Graph C context encoder against transformer-based sequence baselines.

## Project Difference

The current Sprint 7/Sprint 8 line isolates graph topology, edge-aware attention, target context, and context encoder design. It does not yet implement a BERT-style sequence encoder.

## Claim Boundary

Use as background for sequence-transformer baselines. Do not cite as local evidence for the Graph C epigenetic-context finding.
