# XGBoost: A Scalable Tree Boosting System

## Bibliographic Metadata

- Authors: Tianqi Chen; Carlos Guestrin
- Year: 2016
- Venue: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
- DOI: https://doi.org/10.1145/2939672.2939785
- Local PDF: `original.pdf`

## Project Relevance

This paper is the primary methodological reference for XGBoost. The thesis uses XGBoost F1--F4 as a strong non-graph tabular baseline, so the model family should be cited independently from CRISPR-specific papers that happen to use XGBoost.

## Key Takeaways for This Thesis

- XGBoost is a scalable gradient tree boosting system for supervised learning.
- Its relevance here is methodological: it justifies the boosted-tree tabular baseline used to compare against graph models.
- The citation should not be used to imply biological interpretation. It supports the model choice, not the CRISPR feature semantics.
- In thesis wording, Chen and Guestrin (2016) can be cited when XGBoost is first introduced as the tabular reference model.

## Thesis Usage

Recommended location:

- `docs/thesis/latex/btu_template/chapters/02_materyal_yontem.tex`, "Model Aileleri" section.

Recommended claim boundary:

- Cite XGBoost as the boosted-tree implementation used for the tabular baseline.
- Keep performance claims tied to local results and tables, not to the XGBoost paper.
