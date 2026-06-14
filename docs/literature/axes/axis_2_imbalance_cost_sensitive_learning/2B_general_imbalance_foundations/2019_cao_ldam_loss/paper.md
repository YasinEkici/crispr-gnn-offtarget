# Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss

## Citation

Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. *NeurIPS 2019*. https://arxiv.org/abs/1906.07413

## Core Idea

LDAM introduces label-distribution-aware margins for imbalanced learning and pairs them with deferred reweighting. The method is designed to improve minority-class generalization in long-tailed classification.

## Project Relevance

LDAM is useful as an advanced imbalance-loss reference when explaining why Sprint 6 considered only a bounded set of predeclared objectives. It belongs to the broader class of cost-sensitive and margin-based alternatives to weighted BCE, focal loss, Dice, and Tversky.

## Project Difference

LDAM was not implemented or tested in Sprint 6. The project should not imply that weighted BCE is globally superior to LDAM; it only beat the predeclared objectives tested under fixed Graph A + S5F2_energy.

## Claim Boundary

Use this paper as future-work / background. Adding LDAM would be a new predeclared experiment, not a post-hoc Sprint 6 conclusion.
