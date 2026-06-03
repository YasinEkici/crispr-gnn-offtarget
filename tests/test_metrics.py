import numpy as np

from crispr_gnn.evaluation.metrics import (
    binary_classification_metrics,
    precision_at_k,
    recall_at_max_fpr,
    select_threshold_by_f1,
)


def test_select_threshold_by_f1_uses_validation_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.9])

    selection = select_threshold_by_f1(y_true, y_score)

    assert selection.policy == "validation_max_f1"
    assert selection.threshold in set(y_score)
    assert selection.f1 > 0


def test_binary_classification_metrics_include_required_outputs() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = binary_classification_metrics(y_true, y_score, threshold=0.5, prefix="test_")

    assert metrics["test_auprc"] == 1.0
    assert metrics["test_auroc"] == 1.0
    assert metrics["test_f1"] == 1.0
    assert metrics["test_macro_f1"] == 1.0
    assert metrics["test_mcc"] == 1.0
    assert metrics["test_specificity"] == 1.0
    assert metrics["test_sensitivity"] == 1.0
    assert metrics["test_tn"] == 2
    assert metrics["test_fp"] == 0
    assert metrics["test_fn"] == 0
    assert metrics["test_tp"] == 2
    assert "test_precision_at_100" in metrics
    assert "test_recall_at_fpr_1pct" in metrics


def test_precision_at_k_caps_to_available_rows() -> None:
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7])

    assert precision_at_k(y_true, y_score, 10) == 2 / 3


def test_precision_at_k_handles_tied_scores_by_expected_precision() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])

    assert precision_at_k(y_true, y_score, 2) == 0.5


def test_recall_at_max_fpr_returns_nan_for_single_class() -> None:
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7])

    assert np.isnan(recall_at_max_fpr(y_true, y_score, 0.01))
