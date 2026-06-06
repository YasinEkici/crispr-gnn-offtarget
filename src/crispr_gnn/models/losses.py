"""Sprint 6 imbalance / loss registry.

These objectives are the controlled variable of Sprint 6 (exec plan
``docs/exec-plans/active/006-sprint6-imbalance-loss-comparison.md``). Everything
else is held at the Sprint 5 best setting (Graph A + ``S5F2_energy``).

Direction note (this project is prevalence-inverted vs. the imbalance
literature): the rare class is the **negative**, and the dominant failure mode is
excess false positives (negatives predicted positive). Class-asymmetric
parameters are therefore oriented to protect the negative class:

- focal ``alpha`` weights the positive (``y=1``) term, so ``alpha<0.5`` up-weights
  the rare negative term ``(1-alpha)``;
- Tversky ``alpha`` is the false-positive weight, so ``alpha>beta`` penalizes the
  negatives-called-positive errors (documented specificity-oriented use; the
  inverse of Salehi 2017's rare-positive default, equivalent to applying the
  standard setting with the negative class as foreground);
- generalized Dice up-weights the rare class through inverse-volume class
  weights.

All losses consume raw logits and float ``{0,1}`` targets, mirroring how
``BCEWithLogitsLoss`` is called in ``src/crispr_gnn/training/gcn.py``. Loss inputs
are expected to already be float32 (the trainer casts under AMP).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

SUPPORTED_LOSSES: tuple[str, ...] = (
    "weighted_bce",
    "bce_unweighted",
    "focal",
    "generalized_dice",
    "tversky",
    "class_balanced_bce",
)


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | float | None,
) -> torch.Tensor:
    """Weighted BCE-with-logits. Identical to the Sprint 4/5 baseline objective."""
    pw = _as_scalar_tensor(pos_weight, logits) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    """Binary focal loss (Lin 2017), numerically stable via the CE-with-logits base.

    ``gamma`` is class-agnostic (focuses on hard/misclassified examples). ``alpha``
    weights the positive term and ``(1-alpha)`` the negative term, so ``alpha<0.5``
    up-weights this project's rare negative class. At ``gamma=0`` this reduces to
    alpha-weighted BCE; with ``alpha=0.5`` that is ``0.5 * BCE``.
    """
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # -log p_t
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).clamp(min=0.0).pow(gamma) * ce
    return loss.mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Soft Tversky loss on the positive class (Salehi 2017).

    ``TI = TP / (TP + alpha*FP + beta*FN)``, loss = ``1 - TI``. ``alpha`` is the
    false-positive weight: ``alpha>beta`` penalizes negatives-called-positive more
    (raises specificity), which targets this project's failure mode. At
    ``alpha=beta=0.5`` this equals the soft Dice loss on the positive class.
    """
    p = torch.sigmoid(logits)
    tp = (p * targets).sum()
    fp = (p * (1.0 - targets)).sum()
    fn = ((1.0 - p) * targets).sum()
    index = (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)
    return 1.0 - index


def generalized_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Generalized Dice loss (Sudre 2017) over both binary classes.

    Each class is weighted by inverse squared volume ``1/(sum(label)^2)`` computed
    from the batch, which up-weights the rare negative class. This is the
    minority-aware Dice; plain single-class Dice on the majority-positive class is
    avoided (degenerate at ~90% prevalence).
    """
    p = torch.sigmoid(logits)
    pos_target = targets
    neg_target = 1.0 - targets
    pos_pred = p
    neg_pred = 1.0 - p

    w_pos = 1.0 / (pos_target.sum().pow(2) + epsilon)
    w_neg = 1.0 / (neg_target.sum().pow(2) + epsilon)

    intersection = w_pos * (pos_pred * pos_target).sum() + w_neg * (neg_pred * neg_target).sum()
    denominator = (
        w_pos * (pos_pred.sum() + pos_target.sum())
        + w_neg * (neg_pred.sum() + neg_target.sum())
    )
    return 1.0 - 2.0 * intersection / (denominator + epsilon)


def class_balanced_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    beta: float,
    train_labels: torch.Tensor,
) -> torch.Tensor:
    """Class-balanced BCE via effective number of samples (Cui 2019).

    Per-class weight ``1 / ((1-beta^n_c)/(1-beta))`` from train-set class counts,
    normalized to sum to the number of classes. A smoother alternative to naive
    inverse-frequency weighting; up-weights the rare negative class.
    """
    labels = train_labels.detach()
    n_pos = float((labels >= 0.5).sum().item())
    n_neg = float((labels < 0.5).sum().item())
    w_pos = _effective_weight(n_pos, beta)
    w_neg = _effective_weight(n_neg, beta)
    scale = 2.0 / (w_pos + w_neg)
    w_pos *= scale
    w_neg *= scale
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    sample_weight = w_pos * targets + w_neg * (1.0 - targets)
    return (sample_weight * ce).mean()


def build_loss(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    train_labels: torch.Tensor | None = None,
) -> LossFn:
    """Return a ``(logits, targets) -> scalar`` callable for a predeclared loss.

    Rejects any loss name outside the Sprint 6 predeclared set.
    """
    key = str(name).strip().lower()
    if key not in SUPPORTED_LOSSES:
        raise ValueError(
            f"Unsupported Sprint 6 loss '{name}'. Predeclared losses: {sorted(SUPPORTED_LOSSES)}"
        )
    params = dict(params or {})

    if key == "weighted_bce":
        pos_weight = _resolve_pos_weight(params.get("pos_weight", "auto"), train_labels)
        return lambda logits, targets: weighted_bce_loss(logits, targets, pos_weight=pos_weight)

    if key == "bce_unweighted":
        pos_weight = params.get("pos_weight", 1.0)
        if float(pos_weight) == 1.0:
            return lambda logits, targets: weighted_bce_loss(logits, targets, pos_weight=None)
        return lambda logits, targets: weighted_bce_loss(logits, targets, pos_weight=float(pos_weight))

    if key == "focal":
        gamma = float(params["gamma"])
        alpha = float(params["alpha"])
        return lambda logits, targets: focal_loss(logits, targets, gamma=gamma, alpha=alpha)

    if key == "tversky":
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        epsilon = float(params.get("epsilon", 1.0))
        return lambda logits, targets: tversky_loss(
            logits, targets, alpha=alpha, beta=beta, epsilon=epsilon
        )

    if key == "generalized_dice":
        epsilon = float(params.get("epsilon", 1.0))
        return lambda logits, targets: generalized_dice_loss(logits, targets, epsilon=epsilon)

    # key == "class_balanced_bce"
    if train_labels is None:
        raise ValueError("class_balanced_bce requires train_labels for effective-number weights")
    beta = float(params.get("beta", 0.999))
    labels = train_labels
    return lambda logits, targets: class_balanced_bce_loss(
        logits, targets, beta=beta, train_labels=labels
    )


def _resolve_pos_weight(
    value: Any,
    train_labels: torch.Tensor | None,
) -> torch.Tensor | float | None:
    if isinstance(value, str) and value.strip().lower() == "auto":
        if train_labels is None:
            raise ValueError("weighted_bce pos_weight='auto' requires train_labels")
        return _auto_pos_weight(train_labels)
    if value is None:
        return None
    return float(value)


def _auto_pos_weight(train_labels: torch.Tensor) -> torch.Tensor:
    """pos_weight = negatives / positives, matching the Sprint 4/5 baseline."""
    labels = train_labels.detach()
    positives = (labels >= 0.5).sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=labels.device)
    return (negatives.float() / positives.float()).to(torch.float32)


def _effective_weight(n: float, beta: float) -> float:
    if n <= 0:
        return 1.0
    effective_number = (1.0 - beta**n) / (1.0 - beta)
    return 1.0 / effective_number


def _as_scalar_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.tensor(float(value), dtype=reference.dtype, device=reference.device)
