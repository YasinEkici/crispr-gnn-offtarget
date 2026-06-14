# On Judging the Significance of Differences by Examining the Overlap Between Confidence Intervals

## Citation

Schenker, N., & Gentleman, J. F. (2001). On Judging the Significance of Differences by Examining the Overlap Between Confidence Intervals. *The American Statistician*, 55(3), 182-186. https://doi.org/10.1198/000313001317097960

## Core Idea

This paper is the Sprint 9 anchor for avoiding the common mistake of comparing two separate marginal confidence intervals by eye. The relevant uncertainty target for a model comparison is the interval for the **difference**, not the amount of overlap between two independently reported intervals.

## Sprint 9 Relevance

Sprint 9 compares fixed-split CRISPR off-target models on the same held-out guide set. Because every candidate is evaluated on common guides, the comparison is paired. The correct analysis is therefore a paired guide-cluster bootstrap of `metric(model A) - metric(model B)` under the same guide resample.

This paper justifies the Sprint 9 rule:

- do not infer model superiority from marginal CI overlap;
- compute paired-difference intervals directly;
- report whether the paired delta interval excludes zero;
- if the paired interval includes zero, state that the analysis did not provide clear evidence of superiority, not that the models are equivalent.

## Project Adaptation

The paper is not CRISPR-specific and does not prescribe guide-level resampling. Sprint 9 adapts the principle to the project setting by pairing model predictions within each resampled `grna_target_id` cluster. This preserves covariance induced by shared guide composition while respecting the guide-disjoint evaluation contract.

## Claim Boundary

Use this reference to support paired-difference reporting. Do not use it to claim equivalence, non-inferiority, or external generalization. Those require separate margins or additional split/population evidence.
