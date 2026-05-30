"""Diagnostic tables and plots for Sprint 2 baseline sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

from crispr_gnn.data.splits import LABEL_COLUMN


GCN_DIAGNOSTIC_TABLES = [
    "gcn_score_direction.csv",
    "gcn_fixed_threshold_metrics.csv",
    "gcn_score_deciles.csv",
]


def write_logistic_regression_diagnostics(
    assigned: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
) -> tuple[list[Path], list[Path]]:
    return write_model_diagnostics(
        assigned,
        predictions,
        output_dir,
        model_name="logistic_regression",
        artifact_prefix="logistic_regression",
        display_name="Logistic Regression",
    )


def write_model_diagnostics(
    assigned: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
    *,
    model_name: str,
    artifact_prefix: str,
    display_name: str,
) -> tuple[list[Path], list[Path]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_df = _prediction_frame(assigned, predictions)
    model_predictions = prediction_df.loc[prediction_df["model_name"] == model_name].copy()
    if model_predictions.empty:
        raise ValueError(f"No {display_name} predictions available for diagnostics")

    tables = [
        _write_score_direction_table(model_predictions, output_path / f"{artifact_prefix}_score_direction.csv"),
        _write_fixed_threshold_table(model_predictions, output_path / f"{artifact_prefix}_fixed_threshold_metrics.csv"),
        _write_per_genome_table(model_predictions, output_path / f"{artifact_prefix}_per_genome_metrics.csv"),
        _write_per_guide_table(model_predictions, output_path / f"{artifact_prefix}_test_per_guide_metrics.csv"),
        _write_decile_table(model_predictions, output_path / f"{artifact_prefix}_score_deciles.csv"),
    ]
    histogram_feature_set = _preferred_histogram_feature_set(model_predictions)
    figures = [
        _write_score_histogram(
            model_predictions,
            output_path / f"{artifact_prefix}_{histogram_feature_set.lower()}_score_histograms.png",
            display_name=display_name,
            feature_set=histogram_feature_set,
        ),
        _write_test_decile_lift(model_predictions, output_path / f"{artifact_prefix}_test_decile_lift.png", display_name=display_name),
        _write_test_per_genome_auroc(model_predictions, output_path / f"{artifact_prefix}_test_per_genome_auroc.png", display_name=display_name),
    ]
    return tables, figures


def write_gcn_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Write Sprint 4 GCN diagnostic tables from model outputs.

    Threshold-dependent diagnostics use the validation-selected threshold
    recorded in the results table. This function does not tune thresholds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_rows = _require_gcn_results(results)
    prediction_rows = _require_gcn_predictions(predictions)
    threshold = _gcn_validation_threshold(result_rows)
    tables = [
        _write_gcn_score_direction_table(prediction_rows, output_path / "gcn_score_direction.csv"),
        _write_gcn_fixed_threshold_table(prediction_rows, output_path / "gcn_fixed_threshold_metrics.csv", threshold=threshold),
        _write_gcn_decile_table(prediction_rows, output_path / "gcn_score_deciles.csv"),
    ]
    if "genome" in prediction_rows.columns:
        tables.append(_write_gcn_per_genome_table(prediction_rows, output_path / "gcn_per_genome_metrics.csv"))
    if "grna_target_id" in prediction_rows.columns:
        tables.append(_write_gcn_per_guide_table(prediction_rows, output_path / "gcn_test_per_guide_metrics.csv"))
    return tables


def write_gcn_report(
    results: pd.DataFrame,
    diagnostic_tables: Iterable[Path],
    figure_paths: Iterable[Path],
    report_path: str | Path,
    *,
    run_label: str = "pending_full_run",
) -> Path:
    """Write the Sprint 4 GCN Markdown report shell from structured artifacts."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_rows = _require_gcn_results(results)
    first = result_rows.iloc[0]
    diagnostics = [Path(item) for item in diagnostic_tables]
    figures = [Path(item) for item in figure_paths]
    lines = [
        "# Sprint 4 GCN Report",
        "",
        f"Run label: `{run_label}`",
        "",
        "## Contract",
        "",
        f"- Label scheme: `{first['label_scheme']}`.",
        f"- Split ID: `{first['split_id']}`.",
        f"- Visibility policy: `{first['visibility_policy']}`.",
        "- Thresholds are selected from validation only.",
        "- Test diagnostics are interpretation-only and cannot drive model or schema decisions.",
        "",
        "## Baseline Reference",
        "",
        f"- Required comparison: `{first['baseline_reference']}`.",
        f"- Baseline test AUPRC: `{float(first['baseline_test_auprc']):.6f}`.",
        f"- Test positive prevalence for GCN result: `{float(first['test_positive_rate']):.6f}`.",
        "",
        "## Result Summary",
        "",
        _markdown_table(
            result_rows[
                [
                    "model_name",
                    "graph_schema",
                    "feature_set",
                    "test_auprc",
                    "test_auroc",
                    "test_f1",
                    "test_mcc",
                ]
            ]
        ),
        "",
        "## Artifact Index",
        "",
        "Diagnostic tables:",
        *[f"- `{item.as_posix()}`" for item in diagnostics],
        "",
        "Figures:",
        *[f"- `{item.as_posix()}`" for item in figures],
        "",
        "## Interpretation Boundaries",
        "",
        "- Graph-view visualizations are bounded sanity checks, not performance claims.",
        "- Smoke or mocked outputs are not final Sprint 4 performance evidence.",
        "- Graph C, when added later, must not be described as topology-only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prediction_frame(assigned: pd.DataFrame, predictions: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = []
    metadata_columns = ["split", LABEL_COLUMN, "grna_target_id", "genome", "experiment_id", "measured"]
    for prediction in predictions:
        row_index = np.asarray(prediction["row_index"])
        part = assigned.loc[row_index, metadata_columns].copy()
        part["model_name"] = str(prediction["model_name"])
        part["feature_set"] = str(prediction["feature_set"])
        part["prediction_split"] = str(prediction["split"])
        part["score"] = np.asarray(prediction["y_score"], dtype=float)
        part["y_true"] = np.asarray(prediction["y_true"], dtype=int)
        rows.append(part)
    frame = pd.concat(rows, axis=0, ignore_index=True)
    if not (frame[LABEL_COLUMN].to_numpy(dtype=int) == frame["y_true"].to_numpy(dtype=int)).all():
        raise ValueError("Prediction labels do not align with assigned row labels")
    return frame


def _write_score_direction_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "feature_set", "prediction_split"], sort=True):
        y_true = part["y_true"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": keys[0],
                "feature_set": keys[1],
                "split": keys[2],
                "rows": int(part.shape[0]),
                "positives": int(y_true.sum()),
                "negatives": int((y_true == 0).sum()),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auprc_inverted": _safe_auprc(y_true, -score),
                "auroc": _safe_auroc(y_true, score),
                "auroc_inverted": _safe_auroc(y_true, -score),
                "mean_score_positive": _mean_or_nan(score[y_true == 1]),
                "mean_score_negative": _mean_or_nan(score[y_true == 0]),
                "median_score_positive": _median_or_nan(score[y_true == 1]),
                "median_score_negative": _median_or_nan(score[y_true == 0]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_fixed_threshold_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "feature_set", "prediction_split"], sort=True):
        y_true = part["y_true"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        y_pred = (score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model_name": keys[0],
                "feature_set": keys[1],
                "split": keys[2],
                "threshold": 0.5,
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_per_genome_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "feature_set", "prediction_split", "genome"], dropna=False, sort=True):
        y_true = part["y_true"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(_ranking_row(keys, part, y_true, score, group_key="genome"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_per_guide_table(df: pd.DataFrame, path: Path) -> Path:
    test = df.loc[df["prediction_split"] == "test"].copy()
    rows = []
    for keys, part in test.groupby(["model_name", "feature_set", "grna_target_id"], dropna=False, sort=True):
        y_true = part["y_true"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(_ranking_row(keys, part, y_true, score, group_key="grna_target_id"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_decile_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "feature_set", "prediction_split"], sort=True):
        ranked = part.sort_values("score", ascending=False).copy()
        ranked["score_rank"] = np.arange(1, ranked.shape[0] + 1)
        ranked["score_decile"] = np.ceil(ranked["score_rank"] * 10 / ranked.shape[0]).astype(int)
        for decile, decile_part in ranked.groupby("score_decile", sort=True):
            labels = decile_part["y_true"].to_numpy(dtype=int)
            rows.append(
                {
                    "model_name": keys[0],
                    "feature_set": keys[1],
                    "split": keys[2],
                    "score_decile": int(decile),
                    "rows": int(decile_part.shape[0]),
                    "positives": int(labels.sum()),
                    "negatives": int((labels == 0).sum()),
                    "positive_rate": float(labels.mean()),
                    "mean_score": float(decile_part["score"].mean()),
                    "min_score": float(decile_part["score"].min()),
                    "max_score": float(decile_part["score"].max()),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_score_histogram(df: pd.DataFrame, path: Path, *, display_name: str, feature_set: str) -> Path:
    selected = df.loc[df["feature_set"] == feature_set].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split_name in zip(axes, ["val", "test"], strict=True):
        part = selected.loc[selected["prediction_split"] == split_name]
        ax.hist(part.loc[part["y_true"] == 1, "score"], bins=30, alpha=0.65, label="positive", color="#2a7f62")
        ax.hist(part.loc[part["y_true"] == 0, "score"], bins=30, alpha=0.65, label="negative", color="#9b4d48")
        ax.set_title(f"{display_name} {feature_set} scores: {split_name}")
        ax.set_xlabel("Predicted probability for class 1")
        ax.set_ylabel("Rows")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _preferred_histogram_feature_set(df: pd.DataFrame) -> str:
    feature_sets = sorted(df["feature_set"].astype(str).unique().tolist())
    if "F4" in feature_sets:
        return "F4"
    if "S1" in feature_sets:
        return "S1"
    return feature_sets[-1]


def _write_test_decile_lift(df: pd.DataFrame, path: Path, *, display_name: str) -> Path:
    deciles_path = path.with_suffix(".tmp.csv")
    _write_decile_table(df, deciles_path)
    deciles = pd.read_csv(deciles_path)
    deciles_path.unlink(missing_ok=True)
    test = deciles.loc[deciles["split"] == "test"].copy()
    prevalence = float(df.loc[df["prediction_split"] == "test", "y_true"].mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    for feature_set, part in test.groupby("feature_set", sort=True):
        ax.plot(part["score_decile"], part["positive_rate"], marker="o", label=feature_set)
    ax.axhline(prevalence, color="#555555", linestyle="--", linewidth=1.2, label="test prevalence")
    ax.set_xlabel("Score decile (1 = highest scores)")
    ax.set_ylabel("Positive rate")
    ax.set_title(f"{display_name} test decile lift")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_test_per_genome_auroc(df: pd.DataFrame, path: Path, *, display_name: str) -> Path:
    per_genome_path = path.with_suffix(".tmp.csv")
    _write_per_genome_table(df, per_genome_path)
    per_genome = pd.read_csv(per_genome_path)
    per_genome_path.unlink(missing_ok=True)
    test = per_genome.loc[per_genome["split"] == "test"].copy()
    pivot = test.pivot(index="feature_set", columns="genome", values="auroc")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="random")
    ax.set_ylabel("AUROC")
    ax.set_title(f"{display_name} test AUROC by genome")
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _ranking_row(keys: tuple[object, ...], part: pd.DataFrame, y_true: np.ndarray, score: np.ndarray, *, group_key: str) -> dict[str, object]:
    base = {
        "model_name": keys[0],
        "feature_set": keys[1],
    }
    if group_key == "genome":
        base.update({"split": keys[2], "genome": keys[3]})
    else:
        base.update({"grna_target_id": keys[2]})
    return {
        **base,
        "rows": int(part.shape[0]),
        "positives": int(y_true.sum()),
        "negatives": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()),
        "auprc": _safe_auprc(y_true, score),
        "auroc": _safe_auroc(y_true, score),
        "mean_score_positive": _mean_or_nan(score[y_true == 1]),
        "mean_score_negative": _mean_or_nan(score[y_true == 0]),
        "mean_score": float(score.mean()),
    }


def _safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.shape[0] else float("nan")


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.shape[0] else float("nan")


def _require_gcn_results(results: pd.DataFrame) -> pd.DataFrame:
    _require_table_columns(
        results,
        {
            "model_name",
            "feature_set",
            "graph_schema",
            "label_scheme",
            "split_id",
            "visibility_policy",
            "threshold",
            "threshold_selection_split",
            "baseline_reference",
            "baseline_test_auprc",
            "test_positive_rate",
            "test_auprc",
            "test_auroc",
            "test_f1",
            "test_mcc",
        },
        "GCN results",
    )
    if not (results["threshold_selection_split"] == "validation").all():
        raise ValueError("GCN diagnostics require validation-selected thresholds")
    return results.copy()


def _require_gcn_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_table_columns(
        predictions,
        {"model_name", "feature_set", "graph_schema", "split", "label", "score"},
        "GCN predictions",
    )
    return predictions.copy()


def _gcn_validation_threshold(results: pd.DataFrame) -> float:
    thresholds = results["threshold"].dropna().astype(float).unique()
    if len(thresholds) != 1:
        raise ValueError("GCN diagnostics require exactly one validation-selected threshold for this path")
    return float(thresholds[0])


def _write_gcn_score_direction_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "graph_schema", "feature_set", "split"], sort=True):
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": keys[0],
                "graph_schema": keys[1],
                "feature_set": keys[2],
                "split": keys[3],
                "rows": int(part.shape[0]),
                "positives": int(y_true.sum()),
                "negatives": int((y_true == 0).sum()),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auprc_inverted": _safe_auprc(y_true, -score),
                "auroc": _safe_auroc(y_true, score),
                "auroc_inverted": _safe_auroc(y_true, -score),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gcn_fixed_threshold_table(df: pd.DataFrame, path: Path, *, threshold: float) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "graph_schema", "feature_set", "split"], sort=True):
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model_name": keys[0],
                "graph_schema": keys[1],
                "feature_set": keys[2],
                "split": keys[3],
                "threshold": float(threshold),
                "threshold_selection_split": "validation",
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gcn_decile_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "graph_schema", "feature_set", "split"], sort=True):
        ranked = part.sort_values("score", ascending=False).copy()
        ranked["score_rank"] = np.arange(1, ranked.shape[0] + 1)
        ranked["score_decile"] = np.ceil(ranked["score_rank"] * 10 / ranked.shape[0]).astype(int)
        for decile, decile_part in ranked.groupby("score_decile", sort=True):
            labels = decile_part["label"].to_numpy(dtype=int)
            rows.append(
                {
                    "model_name": keys[0],
                    "graph_schema": keys[1],
                    "feature_set": keys[2],
                    "split": keys[3],
                    "score_decile": int(decile),
                    "rows": int(decile_part.shape[0]),
                    "positive_rate": float(labels.mean()),
                    "mean_score": float(decile_part["score"].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gcn_per_genome_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "graph_schema", "feature_set", "split", "genome"], dropna=False, sort=True):
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": keys[0],
                "graph_schema": keys[1],
                "feature_set": keys[2],
                "split": keys[3],
                "genome": keys[4],
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auroc": _safe_auroc(y_true, score),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_gcn_per_guide_table(df: pd.DataFrame, path: Path) -> Path:
    test = df.loc[df["split"] == "test"].copy()
    rows = []
    for keys, part in test.groupby(["model_name", "graph_schema", "feature_set", "grna_target_id"], dropna=False, sort=True):
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": keys[0],
                "graph_schema": keys[1],
                "feature_set": keys[2],
                "grna_target_id": keys[3],
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auroc": _safe_auroc(y_true, score),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _require_table_columns(df: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")
