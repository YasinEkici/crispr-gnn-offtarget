# Is Attention Interpretable?

## Bibliographic Metadata

- Authors: Sofia Serrano; Noah A. Smith
- Year: 2019
- Venue: Proceedings of ACL 2019
- DOI: https://doi.org/10.18653/v1/P19-1282
- ACL Anthology: https://aclanthology.org/P19-1282/
- Local PDF: `original.pdf`

## Project Relevance

This paper is a methodological caution for interpreting attention weights. The thesis reports attention-based and modulation-based graph models, but it should avoid treating learned internal weights as direct biological causal evidence.

## Key Takeaways for This Thesis

- Attention weights may correlate with importance in some settings, but they are not a guaranteed explanation.
- Manipulating attention can reveal cases where attention magnitude and prediction impact diverge.
- This supports the thesis wording that attention, gate, FiLM, and feature-masking signals are model-behavior evidence and require separate biological validation before causal interpretation.

## Thesis Usage

Recommended location:

- `docs/thesis/latex/btu_template/chapters/04_sonuc_oneriler.tex`, "Sınırlılıklar" section.

Recommended claim boundary:

- Cite Serrano and Smith (2019) together with or instead of Jain and Wallace (2019) when discussing attention interpretability limits.
- Keep biological claims grounded in CRISPR/chromatin literature and local ablation evidence.
