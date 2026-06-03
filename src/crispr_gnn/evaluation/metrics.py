"""Sprint 2 binary classification metrics and validation thresholding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_K_VALUES = (100, 500)
DEFAULT_FPR_LEVELS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float
    policy: str = "validation_max_f1"


def select_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> ThresholdSelection:
    """Choose a classification threshold from validation scores only."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] == 0:
        raise ValueError("Cannot select a threshold from an empty validation set")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.shape[0] == 0:
        return ThresholdSelection(threshold=0.5, f1=0.0)

    f1_values = _safe_f1_from_precision_recall(precision[:-1], recall[:-1])
    best_index = int(np.nanargmax(f1_values))
    return ThresholdSelection(threshold=float(thresholds[best_index]), f1=float(f1_values[best_index]))


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    prefix: str = "",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    fpr_levels: tuple[float, ...] = DEFAULT_FPR_LEVELS,
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if labels.shape[0] == 0:
        raise ValueError("Cannot compute metrics for an empty target array")

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        f"{prefix}rows": int(labels.shape[0]),
        f"{prefix}positive_rate": float(labels.mean()),
        f"{prefix}auprc": _safe_average_precision(labels, scores),
        f"{prefix}auroc": _safe_auroc(labels, scores),
        f"{prefix}f1": float(f1_score(labels, predictions, zero_division=0)),
        f"{prefix}macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        f"{prefix}mcc": float(matthews_corrcoef(labels, predictions)),
        f"{prefix}specificity": _safe_ratio(tn, tn + fp),
        f"{prefix}sensitivity": _safe_ratio(tp, tp + fn),
        f"{prefix}tn": int(tn),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tp": int(tp),
    }

    for k in k_values:
        result[f"{prefix}precision_at_{k}"] = precision_at_k(labels, scores, k)
    result[f"{prefix}precision_at_top_1pct"] = precision_at_fraction(labels, scores, fraction=0.01)
    for level in fpr_levels:
        key_level = int(round(level * 100))
        result[f"{prefix}recall_at_fpr_{key_level}pct"] = recall_at_max_fpr(labels, scores, max_fpr=level)
    return result


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    capped_k = min(k, labels.shape[0])
    if capped_k == 0:
        return 0.0

    sorted_scores = np.sort(scores)[::-1]
    boundary = sorted_scores[capped_k - 1]
    above_mask = scores > boundary
    tied_mask = scores == boundary
    above_count = int(above_mask.sum())
    tied_count = int(tied_mask.sum())
    needed_from_ties = capped_k - above_count
    above_positives = float(labels[above_mask].sum())
    tied_positive_rate = float(labels[tied_mask].mean()) if tied_count else 0.0
    expected_positives = above_positives + needed_from_ties * tied_positive_rate
    return float(expected_positives / capped_k)


def precision_at_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    labels = np.asarray(y_true, dtype=int)
    k = max(1, int(np.ceil(labels.shape[0] * fraction)))
    return precision_at_k(labels, y_score, k)


def recall_at_max_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if np.unique(labels).shape[0] < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = tpr[fpr <= max_fpr]
    return float(eligible.max()) if eligible.shape[0] else 0.0


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, y_score))


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=int)
    if np.unique(labels).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(labels, y_score))


def _safe_f1_from_precision_recall(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    denominator = precision + recall
    return np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")
