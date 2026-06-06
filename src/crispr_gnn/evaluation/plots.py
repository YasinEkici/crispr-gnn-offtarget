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
    "gcn_view_sanity_example.png",
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
    schema_label: str | None = None,
    graph_view: object | None = None,
    sequence_position_sensitivity: pd.DataFrame | None = None,
) -> list[Path]:
    """Write Sprint 4 GCN report-ready figures from structured run outputs.

    The plotting path consumes run metadata and validation-selected thresholds;
    it does not select models, schemas, epochs, or thresholds from test scores.
    When schema_label is provided files use a gcn_{schema_label}_ filename prefix;
    the caller is responsible for passing the full target directory as output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"gcn_{schema_label}" if schema_label else "gcn"
    result_rows = _require_gcn_result_columns(results)
    prediction_rows = _require_gcn_prediction_columns(predictions)
    thresholds = _validation_selected_thresholds(result_rows)
    paths = [
        _write_gcn_schema_auprc_comparison(result_rows, output_path / f"{prefix}_graph_schema_auprc_comparison.png"),
        _write_gcn_pr_curves(prediction_rows, output_path / f"{prefix}_pr_curves.png"),
        _write_gcn_roc_curves(prediction_rows, output_path / f"{prefix}_roc_curves.png"),
        _write_gcn_training_curves(training_history, output_path / f"{prefix}_training_curves.png"),
        _write_gcn_score_distributions(prediction_rows, output_path / f"{prefix}_score_distributions.png"),
        _write_gcn_confusion_matrices(prediction_rows, output_path / f"{prefix}_confusion_matrices.png", thresholds=thresholds),
        _write_gcn_decile_lift(prediction_rows, output_path / f"{prefix}_decile_lift.png"),
        _write_gcn_per_genome_metrics(prediction_rows, output_path / f"{prefix}_per_genome_metrics.png"),
        _write_graph_view_sanity_example(graph_view, output_path / f"{prefix}_view_sanity_example.png"),
    ]
    if sequence_position_sensitivity is not None:
        paths.append(
            _write_gcn_sequence_position_sensitivity(
                sequence_position_sensitivity,
                output_path / f"{prefix}_sequence_position_sensitivity.png",
            )
        )
    return paths


def write_sprint6_imbalance_plots(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    training_history: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Write Sprint 6 imbalance/loss-comparison figures.

    This is separate from ``write_gcn_plots`` so Sprint 4/5 report output names
    and grouping behavior remain unchanged. All comparisons group by ``run_id``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_rows = _require_sprint6_result_columns(results)
    prediction_rows = _require_sprint6_prediction_columns(predictions)
    thresholds = _sprint6_thresholds(result_rows)
    reference = _sprint6_reference_context(result_rows)
    return [
        _write_sprint6_auprc_comparison(
            result_rows,
            output_path / "imbalance_auprc_comparison.png",
            reference=reference,
        ),
        _write_sprint6_pr_curves(
            prediction_rows,
            output_path / "imbalance_pr_curves.png",
            reference=reference,
        ),
        _write_sprint6_threshold_metrics(
            result_rows,
            output_path / "imbalance_threshold_metrics.png",
            reference=reference,
        ),
        _write_sprint6_score_distributions(
            prediction_rows,
            output_path / "imbalance_score_distributions.png",
            reference=reference,
        ),
        _write_sprint6_per_guide_metric_distribution(
            result_rows,
            prediction_rows,
            output_path / "imbalance_per_guide_metric_distribution.png",
            thresholds=thresholds,
            reference=reference,
        ),
        _write_sprint6_positive_retrieval_summary(
            result_rows,
            output_path / "imbalance_positive_retrieval_summary.png",
            reference=reference,
        ),
        _write_sprint6_negative_retrieval_summary(
            result_rows,
            output_path / "imbalance_negative_retrieval_summary.png",
            reference=reference,
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


def _write_gcn_confusion_matrices(predictions: pd.DataFrame, path: Path, *, thresholds: dict[tuple[str, str], float]) -> Path:
    test = predictions.loc[predictions["split"] == "test"].copy()
    groups = list(_gcn_prediction_groups(test))
    if not groups:
        raise ValueError("No GCN test predictions are available for confusion matrices")
    columns = min(3, len(groups))
    rows = int(np.ceil(len(groups) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    for ax, (label, part) in zip(axes.ravel(), groups):
        graph_schema = str(part["graph_schema"].iloc[0])
        feature_set = str(part["feature_set"].iloc[0])
        threshold_key = (graph_schema, feature_set)
        if threshold_key not in thresholds:
            raise ValueError(f"No validation-selected threshold for GCN confusion matrix group: {threshold_key}")
        threshold = thresholds[threshold_key]
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


def _write_sprint6_auprc_comparison(
    results: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    rows = _sprint6_sorted_results(results)
    labels = [
        "xgboost_unweighted\n/ F4",
        "Sprint 5\nS5F2_energy",
        *[_short_run_label(value) for value in rows["run_id"]],
    ]
    values = [
        float(reference["baseline_auprc"]),
        float(reference["sprint5_auprc"]),
        *rows["test_auprc"].astype(float).tolist(),
    ]
    colors = ["#2b6cb0", "#5f6f52", *["#4a5568" for _ in rows.index]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(
        float(reference["prevalence"]),
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        label=f"test prevalence={float(reference['prevalence']):.6f}",
    )
    ax.set_ylabel("Test AUPRC")
    ax.set_title("Sprint 6 loss comparison: AUPRC-first headline view")
    ax.set_ylim(0, min(1.0, max(values + [float(reference["prevalence"])]) + 0.05))
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="lower right")
    _annotate_sprint6_reference(ax, reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_pr_curves(
    predictions: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    test = predictions.loc[predictions["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, part in test.groupby("run_id", sort=True):
        PrecisionRecallDisplay.from_predictions(
            part["label"].to_numpy(dtype=int),
            part["score"].to_numpy(dtype=float),
            name=_short_run_label(run_id),
            ax=ax,
            plot_chance_level=False,
        )
    ax.axhline(
        float(reference["prevalence"]),
        color="#444444",
        linestyle="--",
        linewidth=1,
        label=f"test prevalence={float(reference['prevalence']):.6f}",
    )
    ax.set_title("Sprint 6 precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    _annotate_sprint6_reference(ax, reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_threshold_metrics(
    results: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    rows = _sprint6_sorted_results(results)
    labels = [_short_run_label(value) for value in rows["run_id"]]
    x = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, rows["test_specificity"].astype(float), width=width, label="specificity/TNR", color="#2a7f62")
    ax.bar(x, rows["test_mcc"].astype(float), width=width, label="MCC", color="#805ad5")
    if "test_macro_f1" in rows.columns:
        ax.bar(x + width, rows["test_macro_f1"].astype(float), width=width, label="macro F1", color="#b7791f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Validation-threshold test metric")
    ax.set_title("Sprint 6 threshold metrics at validation-selected thresholds")
    ax.legend(loc="best")
    _annotate_sprint6_reference(ax, reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_score_distributions(
    predictions: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    test = predictions.loc[predictions["split"] == "test"].copy()
    run_ids = sorted(test["run_id"].astype(str).unique())
    columns = min(4, max(1, len(run_ids)))
    rows = int(np.ceil(len(run_ids) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows), squeeze=False)
    for ax in axes.ravel()[len(run_ids):]:
        ax.axis("off")
    for ax, run_id in zip(axes.ravel(), run_ids, strict=False):
        part = test.loc[test["run_id"].astype(str) == run_id]
        ax.hist(part.loc[part["label"] == 1, "score"], bins=18, alpha=0.65, label="positive", color="#2a7f62")
        ax.hist(part.loc[part["label"] == 0, "score"], bins=18, alpha=0.65, label="negative", color="#9b4d48")
        ax.set_title(_short_run_label(run_id))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Rows")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Sprint 6 test score distributions by loss")
    _annotate_sprint6_reference(axes.ravel()[0], reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_per_guide_metric_distribution(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
    *,
    thresholds: dict[str, float],
    reference: dict[str, float | str],
) -> Path:
    per_guide = _sprint6_per_guide_metrics_for_plot(results, predictions, thresholds=thresholds)
    run_ids = _sprint6_sorted_results(results)["run_id"].astype(str).tolist()
    tnr_values = [
        per_guide.loc[per_guide["run_id"].astype(str) == run_id, "negative_retrieval_tnr"].dropna().to_numpy(dtype=float)
        for run_id in run_ids
    ]
    pos_values = [
        per_guide.loc[per_guide["run_id"].astype(str) == run_id, "positive_retrieval_rate"].dropna().to_numpy(dtype=float)
        for run_id in run_ids
    ]
    positions = np.arange(len(run_ids))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(tnr_values, positions=positions - 0.18, widths=0.28, patch_artist=True, boxprops={"facecolor": "#9b4d48"})
    ax.boxplot(pos_values, positions=positions + 0.18, widths=0.28, patch_artist=True, boxprops={"facecolor": "#2a7f62"})
    ax.set_xticks(positions)
    ax.set_xticklabels([_short_run_label(value) for value in run_ids], rotation=25, ha="right")
    ax.set_ylabel("Per-guide retrieval rate")
    ax.set_title("Sprint 6 per-guide positive and negative retrieval distribution")
    ax.plot([], color="#9b4d48", label="negative retrieval/TNR")
    ax.plot([], color="#2a7f62", label="positive retrieval")
    ax.legend(loc="best")
    _annotate_sprint6_reference(ax, reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_positive_retrieval_summary(
    results: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    rows = _sprint6_sorted_results(results)
    labels = [_short_run_label(value) for value in rows["run_id"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, rows["test_sensitivity"].astype(float), color="#2a7f62")
    ax.set_ylabel("Positive retrieval / sensitivity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sprint 6 positive retrieval at validation-selected threshold")
    ax.tick_params(axis="x", labelrotation=25)
    _annotate_sprint6_reference(ax, reference)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_sprint6_negative_retrieval_summary(
    results: pd.DataFrame,
    path: Path,
    *,
    reference: dict[str, float | str],
) -> Path:
    rows = _sprint6_sorted_results(results)
    labels = [_short_run_label(value) for value in rows["run_id"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, rows["test_specificity"].astype(float), color="#9b4d48")
    ax.set_ylabel("Negative retrieval / TNR / specificity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sprint 6 negative retrieval at validation-selected threshold")
    ax.tick_params(axis="x", labelrotation=25)
    _annotate_sprint6_reference(ax, reference)
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


def _require_sprint6_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        results,
        {
            "run_id",
            "loss",
            "threshold",
            "threshold_selection_split",
            "test_auprc",
            "test_positive_rate",
            "baseline_reference",
            "baseline_test_auprc",
            "test_specificity",
            "test_sensitivity",
            "test_mcc",
            "test_tn",
            "test_fp",
            "test_fn",
            "test_tp",
        },
        "Sprint 6 results",
    )
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Sprint 6 plotting requires unique run_id values")
    if not (results["threshold_selection_split"] == "validation").all():
        raise ValueError("Sprint 6 plotting requires validation-selected thresholds")
    return results.copy()


def _require_sprint6_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        predictions,
        {"run_id", "split", "label", "score"},
        "Sprint 6 predictions",
    )
    return predictions.copy()


def _sprint6_thresholds(results: pd.DataFrame) -> dict[str, float]:
    return {str(row["run_id"]): float(row["threshold"]) for _, row in results.iterrows()}


def _sprint6_reference_context(results: pd.DataFrame) -> dict[str, float | str]:
    first = results.iloc[0]
    sprint5_auprc = 0.976585
    sprint5_mcc = 0.477933
    if "prior_sprint5_s5f2_test_auprc" in results.columns:
        sprint5_auprc = float(first["prior_sprint5_s5f2_test_auprc"])
    if "prior_sprint5_s5f2_test_mcc" in results.columns:
        sprint5_mcc = float(first["prior_sprint5_s5f2_test_mcc"])
    prevalence = float(first["test_positive_rate"])
    if "prior_test_positive_prevalence" in results.columns:
        prevalence = float(first["prior_test_positive_prevalence"])
    return {
        "baseline_name": str(first["baseline_reference"]),
        "baseline_auprc": float(first["baseline_test_auprc"]),
        "baseline_mcc": float(first.get("baseline_test_mcc", np.nan)),
        "sprint5_auprc": sprint5_auprc,
        "sprint5_mcc": sprint5_mcc,
        "prevalence": prevalence,
    }


def _sprint6_sorted_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = results.copy()
    if "run_order" in rows.columns:
        rows["_order"] = rows["run_order"].astype(int)
    else:
        rows["_order"] = np.arange(len(rows))
    return rows.sort_values(["_order", "run_id"]).drop(columns=["_order"])


def _short_run_label(value: object) -> str:
    text = str(value)
    for marker in ["_S6R", "-S6R"]:
        if marker in text:
            return "S6R" + text.split(marker, maxsplit=1)[1]
    return text


def _annotate_sprint6_reference(ax: plt.Axes, reference: dict[str, float | str]) -> None:
    text = (
        f"prevalence={float(reference['prevalence']):.6f}\n"
        f"F4 AUPRC={float(reference['baseline_auprc']):.6f}\n"
        f"S5F2 AUPRC={float(reference['sprint5_auprc']):.6f}"
    )
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#a0aec0", "alpha": 0.85},
    )


def _sprint6_per_guide_metrics_for_plot(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if "grna_target_id" not in predictions.columns:
        return pd.DataFrame(
            {
                "run_id": results["run_id"].astype(str),
                "grna_target_id": "unavailable",
                "positive_retrieval_rate": np.nan,
                "negative_retrieval_tnr": np.nan,
            }
        )
    test = predictions.loc[predictions["split"] == "test"].copy()
    rows = []
    for keys, part in test.groupby(["run_id", "grna_target_id"], dropna=False, sort=True):
        run_id = str(keys[0])
        y_true = part["label"].to_numpy(dtype=int)
        y_pred = (part["score"].to_numpy(dtype=float) >= thresholds[run_id]).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tn, fp, fn, tp = matrix
        rows.append(
            {
                "run_id": run_id,
                "grna_target_id": keys[1],
                "positive_retrieval_rate": _plot_safe_ratio(tp, tp + fn),
                "negative_retrieval_tnr": _plot_safe_ratio(tn, tn + fp),
            }
        )
    return pd.DataFrame(rows)


def _plot_safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


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


def _validation_selected_thresholds(results: pd.DataFrame) -> dict[tuple[str, str], float]:
    thresholds: dict[tuple[str, str], float] = {}
    for keys, part in results.groupby(["graph_schema", "feature_set"], sort=True):
        values = part["threshold"].dropna().astype(float).unique()
        if len(values) != 1:
            raise ValueError(f"GCN plotting requires one validation-selected threshold for {keys}")
        thresholds[(str(keys[0]), str(keys[1]))] = float(values[0])
    return thresholds


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
