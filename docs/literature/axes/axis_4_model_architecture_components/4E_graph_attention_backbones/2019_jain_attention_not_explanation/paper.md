# Attention is not Explanation

## Bibliographic Metadata

- Authors: Sarthak Jain; Byron C. Wallace
- Year: 2019
- Venue: Proceedings of NAACL-HLT 2019
- DOI: https://doi.org/10.18653/v1/N19-1357
- ACL Anthology: https://aclanthology.org/N19-1357/
- Local PDF: `original.pdf`

## Project Relevance

The thesis uses GAT/GATv2 and discusses attention, gate, FiLM, and masking outputs as model-behavior evidence. This paper supports a conservative interpretation boundary: attention weights and related internal signals should not be treated as direct biological explanations without additional validation.

## Key Takeaways for This Thesis

- Attention mechanisms can improve modeling but do not automatically provide faithful explanations.
- Alternative attention patterns can sometimes preserve similar predictions, which weakens naive causal readings of attention weights.
- For this thesis, the paper is useful as a cautionary citation when stating that attention/gate/FiLM signals are model-behavior indicators rather than biological causality evidence.

## Thesis Usage

Recommended locations:

- `docs/thesis/latex/btu_template/chapters/04_sonuc_oneriler.tex`, "Sınırlılıklar" section.
- Optional: `docs/thesis/latex/btu_template/chapters/02_materyal_yontem.tex`, where attention outputs are framed as interpretation aids rather than causal evidence.

Recommended claim boundary:

- Use the citation to support caution around attention interpretability.
- Do not overgeneralize it to all explanation methods or all GNN attention settings.
