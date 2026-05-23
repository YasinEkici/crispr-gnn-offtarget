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
        _write_auprc_bar(results, output_path / "logistic_regression_feature_set_auprc.png", model_names=["dummy_prior", "logistic_regression"]),
        _write_pr_curves(prediction_rows, output_path / "logistic_regression_pr_curves.png", model_name="logistic_regression"),
        _write_roc_curves(prediction_rows, output_path / "logistic_regression_roc_curves.png", model_name="logistic_regression"),
    ]
    return paths


def write_xgboost_plots(
    results: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(predictions)
    return [
        _write_auprc_bar(
            results,
            output_path / "xgboost_feature_set_auprc.png",
            model_names=["xgboost_unweighted", "xgboost_balanced_train_weights"],
        ),
        _write_pr_curves(prediction_rows, output_path / "xgboost_unweighted_pr_curves.png", model_name="xgboost_unweighted"),
        _write_roc_curves(prediction_rows, output_path / "xgboost_unweighted_roc_curves.png", model_name="xgboost_unweighted"),
        _write_pr_curves(
            prediction_rows,
            output_path / "xgboost_balanced_train_weights_pr_curves.png",
            model_name="xgboost_balanced_train_weights",
        ),
        _write_roc_curves(
            prediction_rows,
            output_path / "xgboost_balanced_train_weights_roc_curves.png",
            model_name="xgboost_balanced_train_weights",
        ),
    ]


def _write_auprc_bar(results: pd.DataFrame, path: Path, *, model_names: list[str]) -> Path:
    rows = results.loc[results["model_name"].isin(model_names)].copy()
    rows = rows.sort_values(["model_name", "feature_set"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = rows["model_name"] + " / " + rows["feature_set"]
    ax.bar(x_labels, rows["test_auprc"], color=[_model_color(name) for name in rows["model_name"]])
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


def _write_pr_curves(predictions: list[dict[str, object]], path: Path, *, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in predictions:
        if row["model_name"] != model_name or row.get("split", "test") != "test":
            continue
        PrecisionRecallDisplay.from_predictions(
            row["y_true"],
            row["y_score"],
            name=str(row["feature_set"]),
            ax=ax,
            plot_chance_level=False,
        )
    ax.set_title(f"{_display_model_name(model_name)} precision-recall curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_roc_curves(predictions: list[dict[str, object]], path: Path, *, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in predictions:
        if row["model_name"] != model_name or row.get("split", "test") != "test":
            continue
        RocCurveDisplay.from_predictions(
            row["y_true"],
            row["y_score"],
            name=str(row["feature_set"]),
            ax=ax,
            plot_chance_level=False,
        )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    ax.set_title(f"{_display_model_name(model_name)} ROC curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _model_color(model_name: str) -> str:
    colors = {
        "dummy_prior": "#87919c",
        "logistic_regression": "#2a7f62",
        "xgboost_unweighted": "#2b6cb0",
        "xgboost_balanced_train_weights": "#b7791f",
    }
    return colors.get(model_name, "#4a5568")


def _display_model_name(model_name: str) -> str:
    names = {
        "logistic_regression": "Logistic Regression",
        "xgboost_unweighted": "XGBoost unweighted",
        "xgboost_balanced_train_weights": "XGBoost balanced train weights",
    }
    return names.get(model_name, model_name)
