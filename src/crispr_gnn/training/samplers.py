"""Sprint 6 training-time samplers (measured-only).

The only sampling strategy in the Sprint 6 predeclared run list (``S6R7``) is a
balanced supervised-edge subsample: keep every training negative (the rare class)
and subsample the majority positives to a target class ratio, deterministically
per epoch. This changes only the training batch composition; the locked
measured-only evaluation universe is never touched.

These helpers operate on the training supervised labels only and return a boolean
mask to be ANDed with the existing supervision mask in the trainer. They do not
read validation/test data.
"""

from __future__ import annotations

import numpy as np


def balanced_subsample_mask(
    labels: np.ndarray,
    *,
    target_ratio: float = 1.0,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Keep all negatives; subsample positives to ``target_ratio * n_negatives``.

    Here the rare class is the negative (label 0), so negatives are always kept and
    the majority positives (label 1) are subsampled. Selection is deterministic for
    a given ``(seed, epoch)`` pair so runs are reproducible and auditable.

    Returns a boolean mask over ``labels`` (True = include this row in the epoch's
    supervised loss).
    """
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    labels = np.asarray(labels)
    int_labels = (labels >= 0.5).astype(int)
    negative_idx = np.flatnonzero(int_labels == 0)
    positive_idx = np.flatnonzero(int_labels == 1)

    mask = np.zeros(labels.shape[0], dtype=bool)
    mask[negative_idx] = True  # rare class always fully kept

    n_keep_positive = int(round(target_ratio * negative_idx.shape[0]))
    if n_keep_positive >= positive_idx.shape[0]:
        mask[positive_idx] = True
        return mask

    rng = np.random.default_rng([int(seed), int(epoch)])
    chosen = rng.choice(positive_idx, size=n_keep_positive, replace=False)
    mask[chosen] = True
    return mask
