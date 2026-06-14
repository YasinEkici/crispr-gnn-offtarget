# Crispr-SGRU: Prediction of CRISPR/Cas9 Off-Target Activities with Mismatches and Indels Using Stacked BiGRU

## Citation

Zhang, G., Luo, Y., Xie, H., & Dai, Z. (2024). Crispr-SGRU: Prediction of CRISPR/Cas9 Off-Target Activities with Mismatches and Indels Using Stacked BiGRU. *International Journal of Molecular Sciences*, 25(20), 10945. https://doi.org/10.3390/ijms252010945

## Core Idea

Crispr-SGRU is a sequence-based off-target model for mismatch and indel cases. It uses recurrent sequence modeling, including stacked bidirectional GRU components, to learn sgRNA-DNA interaction patterns.

## Project Relevance

The paper is useful as a modern sequence baseline and as background for why recurrent encoders remain relevant for CRISPR mismatch/indel modeling. Its use of Dice-style loss also connects loosely to Sprint 6's imbalance/loss-comparison scope.

## Project Difference

Sprint 6 locally found generalized Dice underperformed weighted BCE in the fixed Graph A + S5F2 setting. This does not invalidate Crispr-SGRU; it means any Dice-style objective must be justified and retested under this project's own graph/data contract.

## Claim Boundary

Use as sequence-model related work and optional future baseline context. Do not use it to retroactively justify changing the Sprint 6 winning weighted-BCE contract.
