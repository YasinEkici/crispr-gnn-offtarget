# A systematic study of the class imbalance problem in convolutional neural networks

## Citation

Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249-259. https://doi.org/10.1016/j.neunet.2018.07.011

## Core Idea

This paper empirically studies class imbalance in CNNs and compares common responses such as oversampling, undersampling, two-phase training, and thresholding.

## Project Relevance

It is useful background for why deep models require explicit imbalance controls and why thresholding/sampling should be treated as part of the model contract rather than adjusted after seeing test results.

## Project Difference

The paper studies image CNN settings, not CRISPR graph artifacts. Sprint 6 therefore tested imbalance objectives directly under the project's graph/evaluation contract instead of importing a generic oversampling recommendation.

## Claim Boundary

Use as general deep-learning imbalance evidence. Do not use it to justify synthetic biological graph examples without separate validity checks.
