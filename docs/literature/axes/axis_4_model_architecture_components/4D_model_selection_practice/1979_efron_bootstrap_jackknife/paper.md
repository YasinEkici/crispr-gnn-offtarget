# Bootstrap Methods: Another Look at the Jackknife

## Citation

Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1-26. https://doi.org/10.1214/aos/1176344552

## Core Idea

This foundational paper introduced the bootstrap as a general resampling method and connected it to the jackknife. It motivates estimating sampling variability by repeatedly resampling from the observed data and recomputing the statistic of interest.

## Sprint 9 Relevance

Sprint 9 uses bootstrap resampling to quantify uncertainty around fixed-split model metrics and model differences. The project-specific resampling unit is not an individual row; it is the held-out guide cluster (`grna_target_id`) because the evaluation contract is guide-disjoint and outcomes are dependent within guide.

## Project Adaptation

Sprint 9 adapts the bootstrap foundation as:

- guide-cluster bootstrap for single-model metric compatibility intervals;
- paired guide-cluster bootstrap for model deltas;
- percentile intervals as the primary transparent finite-sample summary;
- BCa intervals only as sensitivity checks because 29 clusters and one dominant negative guide make jackknife acceleration unstable.

## Claim Boundary

This reference supports the general bootstrap strategy. It does not by itself guarantee exact 95% coverage for Sprint 9; the report therefore uses finite-sample guide-cluster compatibility language.
