"""Sprint 6 loss-registry and sampler tests.

Covers: finite outputs, the weighted-BCE regression guard (so the later trainer
refactor cannot change Sprint 4/5 behavior), reduction identities (focal->BCE at
gamma=0, Tversky->Dice at alpha=beta=0.5), and the direction guards that validate
the inverted-prevalence orientation (raising the negative-protection knob must
raise the loss on a false-positive-heavy batch).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from crispr_gnn.models.losses import (
    SUPPORTED_LOSSES,
    build_loss,
    focal_loss,
    generalized_dice_loss,
    tversky_loss,
    weighted_bce_loss,
)
from crispr_gnn.training.samplers import balanced_subsample_mask


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(64)
    targets = (torch.rand(64) > 0.4).float()
    return logits, targets


def _fp_heavy_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    """Negatives (y=0) predicted strongly positive -> excess false positives."""
    logits = torch.tensor([3.0, 3.0, 3.0, 3.0, 2.5, 2.5], dtype=torch.float32)
    targets = torch.zeros(6, dtype=torch.float32)  # all true negatives, all misclassified
    return logits, targets


def test_all_losses_return_finite_scalars() -> None:
    logits, targets = _fixture()
    train_labels = targets.clone()
    for name in SUPPORTED_LOSSES:
        params: dict[str, float] = {}
        if name == "focal":
            params = {"gamma": 2.0, "alpha": 0.25}
        elif name == "tversky":
            params = {"alpha": 0.70, "beta": 0.30}
        elif name == "class_balanced_bce":
            params = {"beta": 0.999}
        loss_fn = build_loss(name, params, train_labels=train_labels)
        value = loss_fn(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_weighted_bce_matches_torch_baseline_regression_guard() -> None:
    """Registry weighted_bce must equal nn.BCEWithLogitsLoss(pos_weight=...)."""
    logits, targets = _fixture()
    pos_weight = torch.tensor(0.1267, dtype=torch.float32)
    expected = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    got = weighted_bce_loss(logits, targets, pos_weight=pos_weight)
    assert torch.allclose(got, expected, atol=1e-7)

    # 'auto' pos_weight reproduces negatives/positives from the train labels.
    loss_fn = build_loss("weighted_bce", {"pos_weight": "auto"}, train_labels=targets)
    n_pos = float((targets >= 0.5).sum())
    n_neg = float(targets.numel() - (targets >= 0.5).sum())
    auto_expected = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(n_neg / n_pos, dtype=torch.float32)
    )(logits, targets)
    assert torch.allclose(loss_fn(logits, targets), auto_expected, atol=1e-7)


def test_bce_unweighted_matches_plain_bce() -> None:
    logits, targets = _fixture()
    loss_fn = build_loss("bce_unweighted", {"pos_weight": 1.0})
    expected = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(loss_fn(logits, targets), expected, atol=1e-7)


def test_focal_reduces_to_half_bce_at_gamma0_alpha_half() -> None:
    logits, targets = _fixture()
    focal = focal_loss(logits, targets, gamma=0.0, alpha=0.5)
    half_bce = 0.5 * F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, half_bce, atol=1e-6)


def test_tversky_reduces_to_soft_dice_at_alpha_beta_half() -> None:
    logits, targets = _fixture()
    tv = tversky_loss(logits, targets, alpha=0.5, beta=0.5, epsilon=0.0)
    p = torch.sigmoid(logits)
    tp = (p * targets).sum()
    fp = (p * (1 - targets)).sum()
    fn = ((1 - p) * targets).sum()
    dice = 1.0 - 2.0 * tp / (2.0 * tp + fp + fn)
    assert torch.allclose(tv, dice, atol=1e-6)


def test_focal_direction_lower_alpha_protects_rare_negatives() -> None:
    """On a false-positive-heavy batch, lower alpha (more negative weight) -> higher loss."""
    logits, targets = _fp_heavy_fixture()
    protect_negatives = focal_loss(logits, targets, gamma=2.0, alpha=0.25)
    protect_positives = focal_loss(logits, targets, gamma=2.0, alpha=0.75)
    assert protect_negatives > protect_positives


def test_tversky_direction_higher_alpha_penalizes_false_positives() -> None:
    """alpha>beta must penalize false positives more than the inverse setting."""
    logits, targets = _fp_heavy_fixture()
    specificity_oriented = tversky_loss(logits, targets, alpha=0.70, beta=0.30)
    recall_oriented = tversky_loss(logits, targets, alpha=0.30, beta=0.70)
    assert specificity_oriented > recall_oriented


def test_generalized_dice_upweights_rare_negative() -> None:
    """Misclassifying the rare class must cost more than misclassifying the majority."""
    # 10 positives, 2 negatives (negatives rare). Compare: only the rare negatives
    # wrong vs only an equal count of the majority positives wrong.
    base_logits = torch.cat([torch.full((10,), 4.0), torch.full((2,), -4.0)])
    targets = torch.cat([torch.ones(10), torch.zeros(2)])

    wrong_negatives = base_logits.clone()
    wrong_negatives[10:] = 4.0  # 2 rare negatives now predicted positive

    wrong_positives = base_logits.clone()
    wrong_positives[:2] = -4.0  # 2 majority positives now predicted negative

    loss_neg_wrong = generalized_dice_loss(wrong_negatives, targets)
    loss_pos_wrong = generalized_dice_loss(wrong_positives, targets)
    assert loss_neg_wrong > loss_pos_wrong


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported Sprint 6 loss"):
        build_loss("dice_focal_unicorn", {})


def test_balanced_subsample_keeps_all_negatives_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random(500) > 0.1).astype(int)  # ~90% positive, negatives rare
    n_neg = int((labels == 0).sum())

    mask_a = balanced_subsample_mask(labels, target_ratio=1.0, seed=42, epoch=0)
    mask_b = balanced_subsample_mask(labels, target_ratio=1.0, seed=42, epoch=0)

    # All negatives kept.
    assert bool(mask_a[labels == 0].all())
    # Positives subsampled to ~target_ratio * n_neg.
    assert int(mask_a[labels == 1].sum()) == n_neg
    # Deterministic for the same (seed, epoch).
    assert np.array_equal(mask_a, mask_b)
    # Different epoch generally changes the positive selection.
    mask_c = balanced_subsample_mask(labels, target_ratio=1.0, seed=42, epoch=1)
    assert not np.array_equal(mask_a, mask_c)
    # Never touches data outside the provided labels (mask length matches).
    assert mask_a.shape[0] == labels.shape[0]
