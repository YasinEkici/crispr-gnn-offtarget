# Bootstrapping Clustered Data

## Citation

Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 69(3), 369-390. https://doi.org/10.1111/j.1467-9868.2007.00593.x

## Core Idea

Field and Welsh discuss bootstrap procedures for clustered data, where observations are not exchangeable at the row level because they are grouped into clusters. The paper is relevant whenever the scientific unit of resampling should be the cluster rather than the individual row.

## Project Relevance

Sprint 9 uses guide-cluster bootstrap summaries because held-out examples are grouped by guide and the negative class is concentrated in a small number of guides. This paper supports the methodological choice to resample guides/clusters rather than individual candidate rows.

## Project Difference

The project applies a pragmatic guide-cluster bootstrap to fixed saved model predictions and thresholded metrics. It does not claim to implement every clustered-bootstrap variant discussed in the paper.

## Claim Boundary

Use this as support for cluster-aware resampling. Do not claim exact nominal coverage in Sprint 9, because the project has only 29 held-out guide clusters and 9 negative-bearing guides.
