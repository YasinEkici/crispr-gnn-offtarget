# R-CRISPR: A Deep Learning Network to Predict Off-Target Activities with Mismatch, Insertion and Deletion in CRISPR-Cas9 System

## Citation

Niu, R., Peng, J., Zhang, Z., & Shang, X. (2021). R-CRISPR: A Deep Learning Network to Predict Off-Target Activities with Mismatch, Insertion and Deletion in CRISPR-Cas9 System. *Genes*, 12(12), 1878. https://doi.org/10.3390/genes12121878

## Core Idea

R-CRISPR is a deep sequence model for off-target activity prediction that explicitly considers mismatches, insertions, and deletions. It is part of the sequence-baseline family that extends beyond mismatch-only encodings.

## Project Relevance

This is useful context for Sprint 8B sequence-context modeling and future sequence encoder work. It reinforces that indel-aware and sequence-specific modeling can be a separate axis from graph/context architecture.

## Project Difference

The current project's headline graph runs are based on the frozen measured-only Scheme A contract and do not implement R-CRISPR's sequence architecture. Adding a CRISPR-Net/R-CRISPR-style encoder would be a separate model-development sprint, not a Sprint 7/8 attribution control.

## Claim Boundary

Use R-CRISPR for related-work positioning around sequence encoders and indels. Do not compare raw scores without dataset and split caveats.
