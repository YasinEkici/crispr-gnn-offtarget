"""Report-ready Sprint 2 baseline plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


GCN_REQUIRED_FIGURES = [
    "gcn_graph_schema_auprc_comparison.png",
    "gcn_pr_curves.png",
    "gcn_roc_curves.png",
    "gcn_training_curves.png",
    "gcn_score_distributions.png",
    "gcn_confusion_matrices.png",
    "gcn_decile_lift.png",
    "gcn_per_genome_metrics.png",
    "graph_view_sanity_example.png",
]


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


def write_mlp_plots(
    results: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(predictions)
    paths = [
        _write_auprc_bar(
            results,
            output_path / "tabular_mlp_feature_set_auprc.png",
            model_names=["tabular_mlp_unweighted", "tabular_mlp_balanced_train_weights"],
        ),
        _write_pr_curves(prediction_rows, output_path / "tabular_mlp_unweighted_pr_curves.png", model_name="tabular_mlp_unweighted"),
        _write_roc_curves(prediction_rows, output_path / "tabular_mlp_unweighted_roc_curves.png", model_name="tabular_mlp_unweighted"),
    ]
    if any(row["model_name"] == "tabular_mlp_balanced_train_weights" for row in prediction_rows):
        paths.extend(
            [
                _write_pr_curves(
                    prediction_rows,
                    output_path / "tabular_mlp_balanced_train_weights_pr_curves.png",
                    model_name="tabular_mlp_balanced_train_weights",
                ),
                _write_roc_curves(
                    prediction_rows,
                    output_path / "tabular_mlp_balanced_train_weights_roc_curves.png",
                    model_name="tabular_mlp_balanced_train_weights",
                ),
            ]
        )
    return paths


def write_sequence_plots(
    results: pd.DataFrame,
    predictions: Iterable[dict[str, object]],
    output_dir: str | Path,
    *,
    artifact_prefix: str = "sequence",
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_rows = list(predictions)
    model_names = sorted(results["model_name"].unique().tolist())
    paths = [
        _write_auprc_bar(
            results,
            output_path / f"{artifact_prefix}_feature_set_auprc.png",
            model_names=model_names,
        )
    ]
    for model_name in model_names:
        paths.extend(
            [
                _write_pr_curves(prediction_rows, output_path / f"{model_name}_pr_curves.png", model_name=model_name),
                _write_roc_curves(prediction_rows, output_path / f"{model_name}_roc_curves.png", model_name=model_name),
            ]
        )
    return paths


def write_gcn_plots(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    training_history: pd.DataFrame,
    output_dir: str | Path,
    *,
    graph_view: object | None = None,
    sequence_position_sensitivity: pd.DataFrame | None = None,
) -> list[Path]:
    """Write Sprint 4 GCN report-ready figures from structured run outputs.

    The plotting path consumes run metadata and validation-selected thresholds;
    it does not select models, schemas, epochs, or thresholds from test scores.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_rows = _require_gcn_result_columns(results)
    prediction_rows = _require_gcn_prediction_columns(predictions)
    threshold = _validation_selected_threshold(result_rows)
    paths = [
        _write_gcn_schema_auprc_comparison(result_rows, output_path / "gcn_graph_schema_auprc_comparison.png"),
        _write_gcn_pr_curves(prediction_rows, output_path / "gcn_pr_curves.png"),
        _write_gcn_roc_curves(prediction_rows, output_path / "gcn_roc_curves.png"),
        _write_gcn_training_curves(training_history, output_path / "gcn_training_curves.png"),
        _write_gcn_score_distributions(prediction_rows, output_path / "gcn_score_distributions.png"),
        _write_gcn_confusion_matrices(prediction_rows, output_path / "gcn_confusion_matrices.png", threshold=threshold),
        _write_gcn_decile_lift(prediction_rows, output_path / "gcn_decile_lift.png"),
        _write_gcn_per_genome_metrics(prediction_rows, output_path / "gcn_per_genome_metrics.png"),
        _write_graph_view_sanity_example(graph_view, output_path / "graph_view_sanity_example.png"),
    ]
    if sequence_position_sensitivity is not None:
        paths.append(
            _write_gcn_sequence_position_sensitivity(
                sequence_position_sensitivity,
                output_path / "gcn_sequence_position_sensitivity.png",
            )
        )
    return paths


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


def _write_gcn_schema_auprc_comparison(results: pd.DataFrame, path: Path) -> Path:
    rows = results.copy()
    rows["plot_label"] = rows["graph_schema"].astype(str) + "\n" + rows["feature_set"].astype(str)
    baseline_name = str(rows["baseline_reference"].dropna().iloc[0])
    baseline_auprc = float(rows["baseline_test_auprc"].dropna().iloc[0])
    prevalence = float(rows["test_positive_rate"].dropna().iloc[0])
    labels = [baseline_name, *rows["plot_label"].tolist()]
    values = [baseline_auprc, *rows["test_auprc"].astype(float).tolist()]
    colors = ["#2b6cb0", *[_model_color(str(name)) for name in rows["model_name"]]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(prevalence, color="#444444", linestyle="--", linewidth=1.2, label=f"test prevalence={prevalence:.3f}")
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Sprint 4 GCN schema AUPRC comparison")
    ax.set_ylim(0, min(1.0, max(values + [prevalence]) + 0.05))
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_pr_curves(predictions: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    test = predictions.loc[predictions["split"] == "test"].copy()
    for label, part in _gcn_prediction_groups(test):
        PrecisionRecallDisplay.from_predictions(
            part["label"].to_numpy(dtype=int),
            part["score"].to_numpy(dtype=float),
            name=label,
            ax=ax,
            plot_chance_level=False,
        )
    if not test.empty:
        prevalence = float(test["label"].mean())
        ax.axhline(prevalence, color="#444444", linestyle="--", linewidth=1, label=f"test prevalence={prevalence:.3f}")
    ax.set_title("Sprint 4 GCN precision-recall curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_roc_curves(predictions: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    test = predictions.loc[predictions["split"] == "test"].copy()
    for label, part in _gcn_prediction_groups(test):
        if part["label"].nunique() < 2:
            continue
        RocCurveDisplay.from_predictions(
            part["label"].to_numpy(dtype=int),
            part["score"].to_numpy(dtype=float),
            name=label,
            ax=ax,
            plot_chance_level=False,
        )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Sprint 4 GCN ROC curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_training_curves(history: pd.DataFrame, path: Path) -> Path:
    _require_columns(history, {"epoch", "train_loss", "val_auprc"}, "GCN training history")
    fig, ax_loss = plt.subplots(figsize=(8, 5))
    for label, part in _history_groups(history):
        ax_loss.plot(part["epoch"], part["train_loss"], marker="o", label=f"{label} train loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_val = ax_loss.twinx()
    for label, part in _history_groups(history):
        ax_val.plot(part["epoch"], part["val_auprc"], marker="s", linestyle="--", label=f"{label} val AUPRC")
    ax_val.set_ylabel("Validation AUPRC")
    handles_1, labels_1 = ax_loss.get_legend_handles_labels()
    handles_2, labels_2 = ax_val.get_legend_handles_labels()
    ax_loss.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8)
    ax_loss.set_title("Sprint 4 GCN training curves")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_score_distributions(predictions: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ["val", "test"], strict=True):
        part = predictions.loc[predictions["split"] == split]
        ax.hist(part.loc[part["label"] == 1, "score"], bins=20, alpha=0.65, label="positive", color="#2a7f62")
        ax.hist(part.loc[part["label"] == 0, "score"], bins=20, alpha=0.65, label="negative", color="#9b4d48")
        ax.set_title(f"GCN scores: {split}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Rows")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_confusion_matrices(predictions: pd.DataFrame, path: Path, *, threshold: float) -> Path:
    test = predictions.loc[predictions["split"] == "test"].copy()
    groups = list(_gcn_prediction_groups(test))
    if not groups:
        raise ValueError("No GCN test predictions are available for confusion matrices")
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, (label, part) in zip(axes.ravel(), groups, strict=True):
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["negative", "positive"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{label}\nvalidation-selected threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_decile_lift(predictions: pd.DataFrame, path: Path) -> Path:
    test = predictions.loc[predictions["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, part in _gcn_prediction_groups(test):
        deciles = _decile_summary(part)
        ax.plot(deciles["score_decile"], deciles["positive_rate"], marker="o", label=label)
    if not test.empty:
        ax.axhline(float(test["label"].mean()), color="#555555", linestyle="--", linewidth=1.2, label="test prevalence")
    ax.set_xlabel("Score decile (1 = highest scores)")
    ax.set_ylabel("Positive rate")
    ax.set_title("Sprint 4 GCN test decile lift")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_per_genome_metrics(predictions: pd.DataFrame, path: Path) -> Path:
    if "genome" not in predictions.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.05, 0.5, "Per-genome metrics unavailable: predictions do not include genome metadata.")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    test = predictions.loc[predictions["split"] == "test"].copy()
    rows = []
    for (graph_schema, genome), part in test.groupby(["graph_schema", "genome"], dropna=False, sort=True):
        rows.append(
            {
                "graph_schema": graph_schema,
                "genome": genome,
                "positive_rate": float(part["label"].mean()),
                "rows": int(part.shape[0]),
            }
        )
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="genome", columns="graph_schema", values="positive_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Test positive rate")
    ax.set_title("Sprint 4 GCN per-genome diagnostic")
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_graph_view_sanity_example(graph_view: object | None, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    if graph_view is None:
        ax.text(
            0.05,
            0.55,
            "Graph view sanity example unavailable in this run.\n"
            "Slice 3 tests use a bounded materialized or mocked loader view.",
            fontsize=10,
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    view_name = str(getattr(graph_view, "view_name", "unknown"))
    graph_name = str(getattr(graph_view, "graph_name", "unknown"))
    node_types = list(getattr(graph_view, "node_types", []))
    edge_types = list(getattr(graph_view, "edge_types", []))
    y_positions = {node_type: index for index, node_type in enumerate(node_types)}
    node_positions: dict[tuple[str, int], tuple[float, float]] = {}
    max_nodes_per_type = 6
    for node_type in node_types:
        node_store = graph_view[node_type]
        audit_ids = list(getattr(node_store, "audit_node_ids", []))
        count = min(int(getattr(node_store, "num_nodes", len(audit_ids))), max_nodes_per_type)
        for index in range(count):
            node_positions[(node_type, index)] = (float(index), float(y_positions[node_type]))
            label = str(audit_ids[index]) if index < len(audit_ids) else str(index)
            ax.scatter(index, y_positions[node_type], s=220, color="#edf2f7", edgecolor="#2d3748", zorder=3)
            ax.text(index, y_positions[node_type], label[:10], ha="center", va="center", fontsize=7)
        ax.text(-0.6, y_positions[node_type], node_type, ha="right", va="center", fontsize=9, fontweight="bold")

    edge_count = 0
    for edge_type in edge_types:
        storage = graph_view[edge_type]
        source_type, relation_type, target_type = edge_type
        edge_index = getattr(storage, "edge_index", None)
        if edge_index is None:
            continue
        edge_array = edge_index.detach().cpu().numpy()
        for source, target in edge_array[:, :20].T:
            source_key = (source_type, int(source))
            target_key = (target_type, int(target))
            if source_key not in node_positions or target_key not in node_positions:
                continue
            sx, sy = node_positions[source_key]
            tx, ty = node_positions[target_key]
            ax.plot([sx, tx], [sy, ty], color="#4a5568", alpha=0.45, linewidth=1)
            if edge_count < 4:
                ax.text((sx + tx) / 2, (sy + ty) / 2 + 0.05, relation_type, fontsize=6, color="#2d3748")
            edge_count += 1

    ax.set_title(f"Bounded strict-inductive graph view sanity example\nschema={graph_name}, view={view_name}")
    ax.text(
        0.01,
        0.01,
        "Bounded, non-exhaustive visualization from a materialized/model-facing view; not a performance claim.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_gcn_sequence_position_sensitivity(sensitivity: pd.DataFrame, path: Path) -> Path:
    _require_columns(sensitivity, {"position", "mean_score_delta"}, "GCN sequence-position sensitivity")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sensitivity["position"], sensitivity["mean_score_delta"], color="#805ad5")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Aligned sequence position")
    ax.set_ylabel("Mean score delta")
    ax.set_title("GCN sequence-position sensitivity diagnostic")
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
        "tabular_mlp_unweighted": "#6b46c1",
        "tabular_mlp_balanced_train_weights": "#d53f8c",
        "sequence_cnn_unweighted": "#0f766e",
        "sequence_bilstm_unweighted": "#7c2d12",
        "sequence_cnn_balanced_train_weights": "#14b8a6",
        "sequence_bilstm_balanced_train_weights": "#f97316",
    }
    return colors.get(model_name, "#4a5568")


def _display_model_name(model_name: str) -> str:
    names = {
        "logistic_regression": "Logistic Regression",
        "xgboost_unweighted": "XGBoost unweighted",
        "xgboost_balanced_train_weights": "XGBoost balanced train weights",
        "tabular_mlp_unweighted": "Tabular MLP unweighted",
        "tabular_mlp_balanced_train_weights": "Tabular MLP balanced train weights",
        "sequence_cnn_unweighted": "Sequence CNN unweighted",
        "sequence_bilstm_unweighted": "Sequence BiLSTM unweighted",
        "sequence_cnn_balanced_train_weights": "Sequence CNN balanced train weights",
        "sequence_bilstm_balanced_train_weights": "Sequence BiLSTM balanced train weights",
    }
    return names.get(model_name, model_name)


def _require_gcn_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        results,
        {
            "model_name",
            "feature_set",
            "graph_schema",
            "test_auprc",
            "test_positive_rate",
            "baseline_reference",
            "baseline_test_auprc",
            "threshold",
            "threshold_selection_split",
        },
        "GCN results",
    )
    if not (results["threshold_selection_split"] == "validation").all():
        raise ValueError("GCN plotting requires validation-selected thresholds")
    return results.copy()


def _require_gcn_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        predictions,
        {"model_name", "feature_set", "graph_schema", "split", "label", "score"},
        "GCN predictions",
    )
    return predictions.copy()


def _validation_selected_threshold(results: pd.DataFrame) -> float:
    thresholds = results["threshold"].dropna().astype(float).unique()
    if len(thresholds) != 1:
        raise ValueError("GCN plotting requires exactly one validation-selected threshold for this Slice 3 path")
    return float(thresholds[0])


def _gcn_prediction_groups(df: pd.DataFrame):
    group_columns = ["graph_schema", "feature_set"]
    for keys, part in df.groupby(group_columns, sort=True):
        yield " / ".join(str(value) for value in keys), part


def _history_groups(history: pd.DataFrame):
    group_columns = [column for column in ["graph_schema", "feature_set"] if column in history.columns]
    if not group_columns:
        yield "GCN", history.sort_values("epoch")
        return
    for keys, part in history.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        yield " / ".join(str(value) for value in keys), part.sort_values("epoch")


def _decile_summary(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values("score", ascending=False).copy()
    ranked["score_rank"] = np.arange(1, ranked.shape[0] + 1)
    ranked["score_decile"] = np.ceil(ranked["score_rank"] * 10 / ranked.shape[0]).astype(int)
    rows = []
    for decile, part in ranked.groupby("score_decile", sort=True):
        rows.append(
            {
                "score_decile": int(decile),
                "rows": int(part.shape[0]),
                "positive_rate": float(part["label"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _require_columns(df: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")
