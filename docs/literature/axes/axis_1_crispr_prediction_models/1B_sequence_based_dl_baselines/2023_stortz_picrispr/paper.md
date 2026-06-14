# piCRISPR: physically informed deep learning models for CRISPR/Cas9 off-target cleavage prediction

## Citation

Stortz, F., Mak, J. K., & Minary, P. (2023). piCRISPR: physically informed deep learning models for CRISPR/Cas9 off-target cleavage prediction. *Artificial Intelligence in the Life Sciences*, 3, 100075. https://doi.org/10.1016/j.ailsci.2023.100075

## Core Idea

piCRISPR studies CRISPR-Cas9 off-target cleavage prediction on the crisprSQL/Mak feature lineage using physically informed sequence, chromatin, accessibility, and nucleosome-related descriptors. It is one of the closest methodological neighbors to this project because it combines off-target prediction with biologically motivated context features rather than using sequence alone.

## Project Relevance

This paper is a direct related-work anchor for Sprint 5 through Sprint 8:

- Sprint 5 found energy/binding features to be the strongest Graph A GCN signal.
- Sprint 7E/7F found target-observation context, especially experimental epigenetic context, to be central for Graph C GATv2 behavior.
- piCRISPR supports the broader premise that physically informed and chromatin/context features can be predictive for off-target cleavage.

## Project Difference

The project does not claim to reproduce piCRISPR. The local contribution is a guide-disjoint graph workflow with explicit graph schemas, edge-aware GATv2 variants, frozen validation thresholds, and robustness quantification. piCRISPR is used as a close precedent for feature motivation and related-work positioning.

## Claim Boundary

Use this paper to motivate context/physical descriptors in CRISPR off-target modeling. Do not use it to claim that our Graph C epigenetic signal is biologically causal; Sprint 7-9 results remain predictive and fixed-split evaluation evidence.
