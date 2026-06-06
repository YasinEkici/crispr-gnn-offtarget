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
    *,
    schema_label: str | None = None,
) -> list[Path]:
    """Write Sprint 4 GCN diagnostic tables from model outputs.

    Threshold-dependent diagnostics use the validation-selected threshold
    recorded in the results table. This function does not tune thresholds.
    When schema_label is provided files use a gcn_{schema_label}_ filename prefix;
    the caller is responsible for passing the full target directory as output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"gcn_{schema_label}" if schema_label else "gcn"
    result_rows = _require_gcn_results(results)
    prediction_rows = _require_gcn_predictions(predictions)
    thresholds = _gcn_validation_thresholds(result_rows)
    tables = [
        _write_gcn_score_direction_table(prediction_rows, output_path / f"{prefix}_score_direction.csv"),
        _write_gcn_fixed_threshold_table(prediction_rows, output_path / f"{prefix}_fixed_threshold_metrics.csv", thresholds=thresholds),
        _write_gcn_decile_table(prediction_rows, output_path / f"{prefix}_score_deciles.csv"),
    ]
    if "genome" in prediction_rows.columns:
        tables.append(_write_gcn_per_genome_table(prediction_rows, output_path / f"{prefix}_per_genome_metrics.csv"))
    if "grna_target_id" in prediction_rows.columns:
        tables.append(_write_gcn_per_guide_table(prediction_rows, output_path / f"{prefix}_test_per_guide_metrics.csv"))
    return tables


def write_sprint6_imbalance_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Write Sprint 6 imbalance-specific diagnostic tables.

    These diagnostics are additive to the Sprint 4/5 GCN reporting path. They
    group by ``run_id`` so loss-comparison rows that share the same model,
    graph schema, and feature set remain distinguishable.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_rows = _require_sprint6_results(results)
    prediction_rows = _require_sprint6_predictions(predictions)
    thresholds = _sprint6_validation_thresholds(result_rows)
    tables = [
        _write_sprint6_score_direction_table(
            prediction_rows,
            output_path / "imbalance_score_direction.csv",
        ),
        _write_sprint6_threshold_metrics_table(
            result_rows,
            prediction_rows,
            output_path / "imbalance_threshold_metrics.csv",
            thresholds=thresholds,
        ),
        _write_sprint6_score_deciles_table(
            prediction_rows,
            output_path / "imbalance_score_deciles.csv",
        ),
        _write_sprint6_per_guide_metrics_table(
            result_rows,
            prediction_rows,
            output_path / "imbalance_per_guide_metrics.csv",
            thresholds=thresholds,
        ),
        _write_sprint6_per_guide_distribution_table(
            result_rows,
            prediction_rows,
            output_path / "imbalance_per_guide_metric_distribution.csv",
            thresholds=thresholds,
        ),
        _write_sprint6_positive_retrieval_summary_table(
            result_rows,
            prediction_rows,
            output_path / "imbalance_positive_retrieval_summary.csv",
            thresholds=thresholds,
        ),
        _write_sprint6_negative_retrieval_summary_table(
            result_rows,
            prediction_rows,
            output_path / "imbalance_negative_retrieval_summary.csv",
            thresholds=thresholds,
        ),
    ]
    if "genome" in prediction_rows.columns:
        tables.append(
            _write_sprint6_per_genome_metrics_table(
                result_rows,
                prediction_rows,
                output_path / "imbalance_per_genome_metrics.csv",
                thresholds=thresholds,
            )
        )
    return tables


def write_gcn_report(
    results: pd.DataFrame,
    diagnostic_tables: Iterable[Path],
    figure_paths: Iterable[Path],
    report_path: str | Path,
    *,
    run_label: str = "pending_full_run",
    root: Path | None = None,
    title: str = "Sprint 4 GCN Report",
    evidence_label: str = "Sprint 4",
) -> Path:
    """Write a GCN Markdown report shell from structured artifacts.

    When root is provided, artifact paths in the report are made relative to root
    so the report does not contain machine-specific absolute paths.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result_rows = _require_gcn_results(results)
    first = result_rows.iloc[0]

    def _format_path(p: Path) -> str:
        if root is not None:
            try:
                return p.relative_to(root).as_posix()
            except ValueError:
                pass
        return p.as_posix()

    diagnostics = [Path(item) for item in diagnostic_tables]
    figures = [Path(item) for item in figure_paths]
    lines = [
        f"# {title}",
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
        _markdown_table(result_rows[_gcn_report_summary_columns(result_rows)]),
        "",
        "## Artifact Index",
        "",
        "Diagnostic tables:",
        *[f"- `{_format_path(item)}`" for item in diagnostics],
        "",
        "Figures:",
        *[f"- `{_format_path(item)}`" for item in figures],
        "",
        "## Interpretation Boundaries",
        "",
        "- Graph-view visualizations are bounded sanity checks, not performance claims.",
        f"- Smoke or mocked outputs are not final {evidence_label} performance evidence.",
        "- Graph C must not be described as topology-only; it changes both topology and target semantics/context representation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _gcn_report_summary_columns(result_rows: pd.DataFrame) -> list[str]:
    columns = [
        "model_name",
        "graph_schema",
        "feature_set",
        "target_node_representation",
        "test_auprc",
        "test_auroc",
        "test_f1",
        "test_macro_f1",
        "test_mcc",
        "test_specificity",
    ]
    if "target_semantics" in result_rows.columns:
        columns.insert(4, "target_semantics")
    return [column for column in columns if column in result_rows.columns]


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
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "specificity": _safe_ratio(tn, tn + fp),
                "sensitivity": _safe_ratio(tp, tp + fn),
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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


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


def _require_sprint6_results(results: pd.DataFrame) -> pd.DataFrame:
    _require_table_columns(
        results,
        {
            "run_id",
            "model_name",
            "feature_set",
            "graph_schema",
            "loss",
            "threshold",
            "threshold_selection_split",
            "test_positive_rate",
            "test_auprc",
            "test_auroc",
            "test_mcc",
            "test_specificity",
            "test_tn",
            "test_fp",
            "test_fn",
            "test_tp",
        },
        "Sprint 6 results",
    )
    if results["run_id"].astype(str).duplicated().any():
        duplicates = sorted(results.loc[results["run_id"].astype(str).duplicated(), "run_id"].astype(str).unique())
        raise ValueError(f"Sprint 6 results require unique run_id values: {duplicates}")
    if not (results["threshold_selection_split"] == "validation").all():
        raise ValueError("Sprint 6 diagnostics require validation-selected thresholds")
    return results.copy()


def _require_sprint6_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_table_columns(
        predictions,
        {"run_id", "model_name", "feature_set", "graph_schema", "split", "label", "score"},
        "Sprint 6 predictions",
    )
    return predictions.copy()


def _sprint6_validation_thresholds(results: pd.DataFrame) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for run_id, part in results.groupby("run_id", sort=True):
        values = part["threshold"].dropna().astype(float).unique()
        if len(values) != 1:
            raise ValueError(f"Sprint 6 diagnostics require one validation-selected threshold for {run_id}")
        thresholds[str(run_id)] = float(values[0])
    return thresholds


def _sprint6_run_metadata(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    optional_columns = [
        "run_id",
        "loss",
        "loss_params",
        "sampling",
        "role",
        "test_auprc",
        "test_mcc",
        "test_specificity",
    ]
    for _, row in results.iterrows():
        metadata[str(row["run_id"])] = {
            column: row[column]
            for column in optional_columns
            if column in results.columns
        }
    return metadata


def _write_sprint6_score_direction_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["run_id", "split"], sort=True):
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        rows.append(
            {
                "run_id": keys[0],
                "split": keys[1],
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
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sprint6_threshold_metrics_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    metadata = _sprint6_run_metadata(results)
    rows = []
    for keys, part in df.groupby(["run_id", "split"], sort=True):
        run_id = str(keys[0])
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= thresholds[run_id]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **metadata.get(run_id, {}),
                "run_id": run_id,
                "split": keys[1],
                "threshold": thresholds[run_id],
                "threshold_selection_split": "validation",
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "specificity": _safe_ratio(tn, tn + fp),
                "tnr": _safe_ratio(tn, tn + fp),
                "sensitivity": _safe_ratio(tp, tp + fn),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sprint6_score_deciles_table(df: pd.DataFrame, path: Path) -> Path:
    rows = []
    for keys, part in df.groupby(["run_id", "split"], sort=True):
        ranked = part.sort_values("score", ascending=False).copy()
        ranked["score_rank"] = np.arange(1, ranked.shape[0] + 1)
        ranked["score_decile"] = np.ceil(ranked["score_rank"] * 10 / ranked.shape[0]).astype(int)
        for decile, decile_part in ranked.groupby("score_decile", sort=True):
            labels = decile_part["label"].to_numpy(dtype=int)
            rows.append(
                {
                    "run_id": keys[0],
                    "split": keys[1],
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


def _sprint6_per_guide_frame(
    results: pd.DataFrame,
    df: pd.DataFrame,
    *,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    metadata = _sprint6_run_metadata(results)
    test = df.loc[df["split"] == "test"].copy()
    rows = []
    for keys, part in test.groupby(["run_id", "grna_target_id"], dropna=False, sort=True):
        run_id = str(keys[0])
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        y_pred = (score >= thresholds[run_id]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **metadata.get(run_id, {}),
                "run_id": run_id,
                "grna_target_id": keys[1],
                "rows": int(part.shape[0]),
                "positives": int(y_true.sum()),
                "negatives": int((y_true == 0).sum()),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auroc": _safe_auroc(y_true, score),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "positive_retrieval_rate": _safe_ratio(tp, tp + fn),
                "negative_retrieval_tnr": _safe_ratio(tn, tn + fp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def _write_sprint6_per_guide_metrics_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    _sprint6_per_guide_frame(results, df, thresholds=thresholds).to_csv(path, index=False)
    return path


def _write_sprint6_per_guide_distribution_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    per_guide = _sprint6_per_guide_frame(results, df, thresholds=thresholds)
    metrics = ["auprc", "auroc", "mcc", "macro_f1", "positive_retrieval_rate", "negative_retrieval_tnr"]
    rows = []
    for run_id, part in per_guide.groupby("run_id", sort=True):
        for metric in metrics:
            values = part[metric].dropna().astype(float)
            rows.append(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "guides_with_metric": int(values.shape[0]),
                    "mean": float(values.mean()) if values.shape[0] else float("nan"),
                    "q10": float(values.quantile(0.10)) if values.shape[0] else float("nan"),
                    "median": float(values.median()) if values.shape[0] else float("nan"),
                    "q90": float(values.quantile(0.90)) if values.shape[0] else float("nan"),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sprint6_positive_retrieval_summary_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    rows = []
    metadata = _sprint6_run_metadata(results)
    test = df.loc[df["split"] == "test"].copy()
    for run_id, part in test.groupby("run_id", sort=True):
        run_id = str(run_id)
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= thresholds[run_id]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **metadata.get(run_id, {}),
                "run_id": run_id,
                "threshold": thresholds[run_id],
                "positives": int(tp + fn),
                "retrieved_positives": int(tp),
                "missed_positives": int(fn),
                "positive_retrieval_rate": _safe_ratio(tp, tp + fn),
                "precision_at_threshold": _safe_ratio(tp, tp + fp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sprint6_negative_retrieval_summary_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    rows = []
    metadata = _sprint6_run_metadata(results)
    test = df.loc[df["split"] == "test"].copy()
    for run_id, part in test.groupby("run_id", sort=True):
        run_id = str(run_id)
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= thresholds[run_id]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **metadata.get(run_id, {}),
                "run_id": run_id,
                "threshold": thresholds[run_id],
                "negatives": int(tn + fp),
                "retrieved_negatives": int(tn),
                "negatives_called_positive": int(fp),
                "negative_retrieval_tnr": _safe_ratio(tn, tn + fp),
                "false_positive_rate": _safe_ratio(fp, tn + fp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_sprint6_per_genome_metrics_table(
    results: pd.DataFrame,
    df: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
) -> Path:
    metadata = _sprint6_run_metadata(results)
    rows = []
    for keys, part in df.groupby(["run_id", "split", "genome"], dropna=False, sort=True):
        run_id = str(keys[0])
        y_true = part["label"].to_numpy(dtype=int)
        score = part["score"].to_numpy(dtype=float)
        y_pred = (score >= thresholds[run_id]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                **metadata.get(run_id, {}),
                "run_id": run_id,
                "split": keys[1],
                "genome": keys[2],
                "rows": int(part.shape[0]),
                "positive_rate": float(y_true.mean()),
                "auprc": _safe_auprc(y_true, score),
                "auroc": _safe_auroc(y_true, score),
                "specificity": _safe_ratio(tn, tn + fp),
                "tnr": _safe_ratio(tn, tn + fp),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _gcn_validation_thresholds(results: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    thresholds: dict[tuple[str, str, str], float] = {}
    for keys, part in results.groupby(["model_name", "graph_schema", "feature_set"], sort=True):
        values = part["threshold"].dropna().astype(float).unique()
        if len(values) != 1:
            raise ValueError(f"GCN diagnostics require one validation-selected threshold for {keys}")
        thresholds[(str(keys[0]), str(keys[1]), str(keys[2]))] = float(values[0])
    return thresholds


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


def _write_gcn_fixed_threshold_table(df: pd.DataFrame, path: Path, *, thresholds: dict[tuple[str, str, str], float]) -> Path:
    rows = []
    for keys, part in df.groupby(["model_name", "graph_schema", "feature_set", "split"], sort=True):
        threshold_key = (str(keys[0]), str(keys[1]), str(keys[2]))
        if threshold_key not in thresholds:
            raise ValueError(f"No validation-selected threshold for GCN diagnostics group: {threshold_key}")
        threshold = thresholds[threshold_key]
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
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "specificity": _safe_ratio(tn, tn + fp),
                "sensitivity": _safe_ratio(tp, tp + fn),
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
