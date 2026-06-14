# CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction

## Citation

Toufikuzzaman, M., Samee, M. A. H., & Rahman, M. S. (2024). CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction. *Briefings in Bioinformatics*, 25(2), bbad530. https://doi.org/10.1093/bib/bbad530

## Core Idea

CRISPR-DIPOFF presents interpretable deep learning models for CRISPR-Cas9 off-target prediction using sequence inputs, genetic-algorithm-based hyperparameter optimization, and integrated-gradient interpretation.

## Project Relevance

This is useful for related work around interpretability and sequence-only baselines. It provides a contrast to the project direction: rather than focusing only on sequence saliency, this project explicitly tests graph topology, target-context node features, edge-aware attention, and robustness under guide-disjoint evaluation.

## Project Difference

The project should not position attention weights, gates, or FiLM summaries as biological explanations in the same way a dedicated interpretation study might. Sprint 9 explicitly limits such artifacts to interpretation signals, not causal biology.

## Claim Boundary

Use CRISPR-DIPOFF to show that interpretability is an active theme in off-target prediction. Do not use it as a direct metric comparator unless dataset, split, label, and prevalence differences are stated.
