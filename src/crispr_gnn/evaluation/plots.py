"""Report-ready Sprint 2 baseline plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def write_baseline_plots(
    results: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(predictions)
    paths = [
        _write_auprc_bar(results, output_path / "logistic_regression_feature_set_auprc.png"),
        _write_pr_curves(prediction_rows, output_path / "logistic_regression_pr_curves.png"),
        _write_roc_curves(prediction_rows, output_path / "logistic_regression_roc_curves.png"),
    ]
    return paths


def _write_auprc_bar(results: pd.DataFrame, path: Path) -> Path:
    rows = results.loc[results["model_name"].isin(["dummy_prior", "logistic_regression"])].copy()
    rows = rows.sort_values(["model_name", "feature_set"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = rows["model_name"] + " / " + rows["feature_set"]
    ax.bar(x_labels, rows["test_auprc"], color=["#87919c" if name == "dummy_prior" else "#2a7f62" for name in rows["model_name"]])
    ax.axhline(rows["test_positive_rate"].iloc[0], color="#444444", linestyle="--", linewidth=1.2, label="test prevalence")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Sprint 2 measured-only baseline AUPRC")
    ax.set_ylim(0, min(1.0, max(0.05, float(rows["test_auprc"].max()) + 0.05)))
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_pr_curves(predictions: list[dict[str, object]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in predictions:
        if row["model_name"] != "logistic_regression":
            continue
        PrecisionRecallDisplay.from_predictions(
            row["y_true"],
            row["y_score"],
            name=str(row["feature_set"]),
            ax=ax,
            plot_chance_level=False,
        )
    ax.set_title("Logistic Regression precision-recall curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_roc_curves(predictions: list[dict[str, object]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in predictions:
        if row["model_name"] != "logistic_regression":
            continue
        RocCurveDisplay.from_predictions(
            row["y_true"],
            row["y_score"],
            name=str(row["feature_set"]),
            ax=ax,
            plot_chance_level=False,
        )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Logistic Regression ROC curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
