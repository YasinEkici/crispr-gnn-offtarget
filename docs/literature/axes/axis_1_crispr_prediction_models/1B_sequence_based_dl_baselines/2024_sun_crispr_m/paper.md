# CRISPR-M: Predicting sgRNA Off-Target Effect Using a Multi-View Deep Learning Network

## Citation

Sun, J., Guo, J., & Liu, J. (2024). CRISPR-M: Predicting sgRNA off-target effect using a multi-view deep learning network. *PLOS Computational Biology*, 20(3), e1011972. https://doi.org/10.1371/journal.pcbi.1011972

## Core Idea

CRISPR-M frames off-target prediction as a multi-view sequence-learning problem for sgRNA-DNA pairs with mismatches and indels. The method combines complementary sequence encodings through deep learning rather than relying on a single handcrafted representation.

## Project Relevance

This is useful related work for future sequence-encoder or hybrid graph-sequence experiments. It helps position any later CRISPR-Net-style or transformer-style encoder work against modern sequence baselines.

## Project Difference

The current project uses guide-disjoint evaluation on the local crisprSQL-derived measured-only universe and focuses on Graph C context-aware GNN variants. CRISPR-M is not reproduced here, and its reported results should not be compared directly without matching data, splits, and metrics.

## Claim Boundary

Use as literature context for sequence encoder design. Do not cite it as evidence that adding a sequence encoder will improve this project's Graph C GATv2 setting without a controlled local experiment.
