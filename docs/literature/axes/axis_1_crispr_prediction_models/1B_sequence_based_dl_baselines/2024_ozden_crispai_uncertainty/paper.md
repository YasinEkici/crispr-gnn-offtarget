# Learning to quantify uncertainty in off-target activity for CRISPR guide RNAs

## Citation

Ozden, F., & Minary, P. (2024). Learning to quantify uncertainty in off-target activity for CRISPR guide RNAs. *Nucleic Acids Research*, 52(18), e87. https://doi.org/10.1093/nar/gkae759

## Core Idea

crispAI models off-target cleavage activity probabilistically and reports uncertainty estimates rather than only point predictions. The method uses a count-noise model to better represent uncertainty in cleavage activity data and proposes uncertainty-aware genome-wide sgRNA scoring.

## Project Relevance

Sprint 9 is not a probabilistic modeling sprint, but it is an uncertainty sprint. crispAI is relevant because it shows that uncertainty quantification is becoming central in CRISPR off-target prediction. The project's current uncertainty layer is evaluation-side robustness: guide-cluster bootstrap, paired-difference bootstrap, and multi-seed fixed-split retraining.

## Project Difference

crispAI estimates predictive uncertainty from the model; Sprint 9 estimates evaluation and training sensitivity from saved predictions and repeated training. These are complementary, not interchangeable.

## Claim Boundary

Use this paper as future-work context for probabilistic off-target models. Do not treat Sprint 9 bootstrap intervals as the same object as crispAI predictive uncertainty.
