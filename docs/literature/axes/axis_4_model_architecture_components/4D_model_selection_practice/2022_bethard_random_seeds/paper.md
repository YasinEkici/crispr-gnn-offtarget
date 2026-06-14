# We Need to Talk About Random Seeds

## Citation

Bethard, S. (2022). We need to talk about random seeds. *arXiv preprint* arXiv:2210.13393. https://arxiv.org/abs/2210.13393

## Core Idea

Bethard argues that random seeds are often misused in neural-network experiments. The note distinguishes safer uses, such as measuring training sensitivity or building ensembles, from riskier uses, such as treating one fixed seed as proof of replicability or using seed variation alone as a broad performance-comparison test.

## Project Relevance

Sprint 9 reports multi-seed retraining as training-stochasticity sensitivity conditional on the same fixed guide-disjoint split. This paper supports the claim boundary: seed spread is useful, but it is not the same as external generalization uncertainty.

## Project Difference

The paper is an opinion/methodology note, not a CRISPR benchmark or a statistical CI procedure. It should be used as reporting guidance, not as a replacement for guide-cluster bootstrap or paired-difference analysis.

## Claim Boundary

Use this to support careful seed reporting and to avoid best-seed selection. Do not use it to claim that multi-seed variance alone proves or disproves model superiority.
