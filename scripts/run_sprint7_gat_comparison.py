from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.graph.graph_schemas import GRAPH_A  # noqa: E402
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader, validate_gcn_headline_config  # noqa: E402
from crispr_gnn.training.gcn import (  # noqa: E402
    collect_graph_a_attention_summary,
    gcn_run_config_from_mapping,
    train_graph_a_gcn,
)
from crispr_gnn.utils.config import load_yaml  # noqa: E402


REFERENCE_RUN_ID = "S7R0_gcn_reference"
HEADLINE_RUN_IDS = ("S7R1_gat_edge_aware", "S7R2_gatv2_edge_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 7 Graph A GAT/GATv2 architecture comparison.")
    parser.add_argument("--config", default="configs/sweeps/sprint7_gat_gatv2.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for smoke/debug runs.")
    parser.add_argument("--run", action="append", default=None, help="Run only this predeclared run ID. May repeat.")
    parser.add_argument("--run-id", default=None, help="Batch ID recorded in manifest and per-run IDs.")
    parser.add_argument(
        "--include-optional-runs",
        action="store_true",
        help="Permit explicitly selected optional runs. Optional runs are never selected by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_sprint7_gat_comparison(
        config_path=ROOT / args.config,
        batch_id=args.run_id,
        max_epochs=args.max_epochs,
        selected_run_ids=args.run,
        include_optional_runs=args.include_optional_runs,
    )
    return 0


def run_sprint7_gat_comparison(
    *,
    config_path: str | Path,
    batch_id: str | None = None,
    max_epochs: int | None = None,
    selected_run_ids: list[str] | None = None,
    include_optional_runs: bool = False,
) -> Path:
    config = load_yaml(config_path)
    _validate_sprint7_base_config(config)
    run_batch_id = batch_id or _batch_id(config)
    run_specs = _selected_run_specs(config, selected_run_ids, include_optional_runs=include_optional_runs)
    include_reference = selected_run_ids is None or REFERENCE_RUN_ID in set(selected_run_ids)

    graph_dir = ROOT / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_A)
    _validate_sprint7_graph_artifacts(materialized.manifest)

    output_dir = ROOT / str(config.get("outputs", {}).get("output_dir", "outputs/sprint7"))
    diagnostics_dir = output_dir / "diagnostics"
    figures_dir = output_dir / "figures"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "graph_artifact_provenance.json"
    _write_graph_provenance(provenance_path, graph_dir=graph_dir, manifest=materialized.manifest, run_specs=run_specs)

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    all_attention: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, object]] = []

    if include_reference:
        reference = _sprint6_reference_row(config, batch_id=run_batch_id)
        all_results.append(pd.DataFrame([reference]))
        manifest_runs.append(
            {
                "run_id": reference["run_id"],
                "predeclared_id": REFERENCE_RUN_ID,
                "role": "sprint6_weighted_bce_reference_no_retrain",
                "source": reference["source_run_id"],
                "checkpoint_path": None,
            }
        )

    for order, run_spec in enumerate(run_specs, start=1 if include_reference else 0):
        run_config_mapping = _config_for_run(config, run_spec, max_epochs=max_epochs)
        run_config = gcn_run_config_from_mapping(run_config_mapping)
        run_id = f"{run_batch_id}_{run_spec['id']}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(run_dir / "resolved_config.yaml", run_config_mapping)
        (run_dir / "runtime.json").write_text(
            json.dumps(_runtime_info(str(run_config.device)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_path = run_dir / "model.pt"
        results, predictions, history = train_graph_a_gcn(
            materialized,
            run_config,
            checkpoint_path=checkpoint_path,
        )
        _insert_run_metadata(results, predictions, history, run_id=run_id, run_spec=run_spec, order=order, config=config)
        attention = collect_graph_a_attention_summary(
            materialized,
            run_config,
            checkpoint_path=checkpoint_path,
            split="test",
        )
        attention.insert(0, "run_id", run_id)
        attention.insert(1, "predeclared_run_id", run_spec["id"])
        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)
        attention.to_csv(run_dir / "attention_summary.csv", index=False)
        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        all_attention.append(attention)
        manifest_runs.append(
            {
                "run_id": run_id,
                "predeclared_id": run_spec["id"],
                "architecture": run_spec["architecture"],
                "edge_aware_attention": run_spec.get("edge_aware_attention", True),
                "role": run_spec.get("role"),
                "metrics_path": _relative(run_dir / "metrics.csv"),
                "training_history_path": _relative(run_dir / "training_history.csv"),
                "attention_summary_path": _relative(run_dir / "attention_summary.csv"),
                "checkpoint_path": _relative(checkpoint_path),
                "resolved_config_path": _relative(run_dir / "resolved_config.yaml"),
            }
        )
        row = results.iloc[0]
        print(
            f"{run_spec['id']}: test_auprc={float(row['test_auprc']):.6f}, "
            f"test_mcc={float(row['test_mcc']):.6f}, "
            f"tn={int(row['test_tn'])}, fp={int(row['test_fp'])}"
        )

    result_table = pd.concat(all_results, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    history_table = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    attention_table = pd.concat(all_attention, ignore_index=True) if all_attention else pd.DataFrame()
    _validate_consolidated_run_ids(result_table)

    results_path = output_dir / "gat_comparison.csv"
    predictions_path = diagnostics_dir / "gat_predictions.csv"
    history_path = diagnostics_dir / "gat_training_history.csv"
    attention_path = diagnostics_dir / "attention_weight_summary.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)
    attention_table.to_csv(attention_path, index=False)

    diagnostic_tables = _write_sprint7_diagnostics(result_table, prediction_table, attention_table, diagnostics_dir)
    figure_paths = _write_sprint7_figures(result_table, prediction_table, history_table, attention_table, figures_dir)
    report_path = _write_sprint7_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        output_dir / "gat_report.md",
        batch_id=run_batch_id,
    )
    manifest_path = output_dir / "gat_run_manifest.json"
    _write_run_manifest(
        manifest_path,
        config_path=Path(config_path),
        batch_id=run_batch_id,
        runs=manifest_runs,
        results_path=results_path,
        predictions_path=predictions_path,
        history_path=history_path,
        attention_path=attention_path,
        report_path=report_path,
        diagnostics=diagnostic_tables,
        figures=figure_paths,
        provenance_path=provenance_path,
        optional_runs_available=[run["id"] for run in config.get("optional_runs", [])],
    )

    print(f"Run batch: {run_batch_id}")
    print(f"Output directory: {_relative(output_dir)}")
    print(f"Results: {_relative(results_path)}")
    print(f"Report: {_relative(report_path)}")
    print(
        result_table[
            [
                "predeclared_run_id",
                "architecture",
                "edge_aware_attention",
                "test_auprc",
                "test_auroc",
                "test_macro_f1",
                "test_mcc",
                "test_specificity",
                "test_tn",
                "test_fp",
                "test_fn",
                "test_tp",
            ]
        ].to_string(index=False)
    )
    return output_dir


def _validate_sprint7_base_config(config: Mapping[str, Any]) -> None:
    validate_gcn_headline_config(config)
    if config.get("sprint") != "sprint7":
        raise ValueError("Sprint 7 GAT runner requires sprint: sprint7")
    if config.get("task") != "sprint7_gat_gatv2_attention":
        raise ValueError("Sprint 7 GAT runner requires task: sprint7_gat_gatv2_attention")
    if config.get("graph", {}).get("schema") != GRAPH_A:
        raise ValueError("Sprint 7 architecture comparison is frozen to Graph A")
    features = config.get("features", {})
    if features.get("feature_set") != "S5F2_energy" or features.get("edge_feature_sets") != ["s5f2_energy"]:
        raise ValueError("Sprint 7 architecture comparison is frozen to S5F2_energy")
    training = config.get("training", {})
    if str(training.get("loss", "")).lower() != "weighted_bce":
        raise ValueError("Sprint 7 must freeze Sprint 6 best loss: weighted_bce")
    if training.get("loss_params", {}).get("pos_weight") != "auto":
        raise ValueError("Sprint 7 weighted BCE must keep pos_weight: auto")
    evaluation = config.get("evaluation", {})
    if evaluation.get("regime") != "measured_only":
        raise ValueError("Sprint 7 headline architecture comparison must remain measured-only")
    if evaluation.get("threshold_policy") != "validation_max_f1":
        raise ValueError("Sprint 7 threshold policy must be validation_max_f1")
    configured_ids = [str(run.get("id")) for run in config.get("runs", [])]
    if configured_ids != list(HEADLINE_RUN_IDS):
        raise ValueError("Sprint 7 headline run list must preserve the frozen S7R1-S7R2 order")


def _selected_run_specs(
    config: Mapping[str, Any],
    selected_run_ids: list[str] | None,
    *,
    include_optional_runs: bool,
) -> list[dict[str, Any]]:
    headline = {str(run["id"]): dict(run) for run in config.get("runs", [])}
    optional = {str(run["id"]): dict(run) for run in config.get("optional_runs", [])}
    if selected_run_ids is None:
        return [headline[run_id] for run_id in HEADLINE_RUN_IDS]
    selected = []
    for run_id in selected_run_ids:
        if run_id == REFERENCE_RUN_ID:
            continue
        if run_id in headline:
            selected.append(headline[run_id])
        elif run_id in optional:
            if not include_optional_runs:
                raise ValueError(
                    f"Optional Sprint 7 run '{run_id}' requires --include-optional-runs. "
                    "Optional edge-blind controls are out of scope for the default runner."
                )
            selected.append(optional[run_id])
        else:
            allowed = sorted([REFERENCE_RUN_ID, *headline, *optional])
            raise ValueError(f"Unknown Sprint 7 run ID '{run_id}'. Allowed IDs: {allowed}")
    return selected


def _config_for_run(
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    max_epochs: int | None,
) -> dict[str, Any]:
    run_config = deepcopy(dict(config))
    model = dict(run_config.get("model", {}))
    model["name"] = str(run_spec["model_name"])
    model["architecture"] = str(run_spec["architecture"])
    model["attention"] = dict(run_spec.get("attention", {}))
    if "edge_aware_attention" in run_spec:
        model["attention"]["edge_aware"] = bool(run_spec["edge_aware_attention"])
    run_config["model"] = model
    training = dict(run_config.get("training", {}))
    training["loss"] = "weighted_bce"
    training["loss_params"] = {"pos_weight": "auto"}
    if max_epochs is not None:
        training["max_epochs"] = int(max_epochs)
        training["min_epochs"] = 1
        training["patience"] = min(2, int(max_epochs))
        training["device"] = "cpu"
        training["use_compile"] = False
        training["use_amp"] = False
    run_config["training"] = training
    run_config["sprint7_run"] = {
        "id": run_spec["id"],
        "role": run_spec.get("role"),
        "architecture": run_spec["architecture"],
        "edge_aware_attention": run_spec.get("edge_aware_attention", True),
        "attention": dict(run_spec.get("attention", {})),
        "loss": "weighted_bce",
        "loss_params": {"pos_weight": "auto"},
    }
    return run_config


def _validate_sprint7_graph_artifacts(manifest: Mapping[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_A:
        raise ValueError("Sprint 7 requires Graph A materialized artifacts")
    feature_tables = manifest.get("feature_tables", {})
    if "S5F2_energy" not in feature_tables and "s5f2_energy" not in feature_tables:
        raise ValueError("Sprint 7 requires the Sprint 5 S5F2_energy feature table")


def _insert_run_metadata(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    *,
    run_id: str,
    run_spec: Mapping[str, Any],
    order: int,
    config: Mapping[str, Any],
) -> None:
    result_payload = {
        "run_id": run_id,
        "run_order": order,
        "predeclared_run_id": run_spec["id"],
        "role": run_spec.get("role"),
        "controlled_variable": "model_architecture",
        "sprint6_reference_test_auprc": config.get("prior_context", {})
        .get("sprint6_wbce", {})
        .get("test_auprc"),
        "sprint6_reference_test_mcc": config.get("prior_context", {})
        .get("sprint6_wbce", {})
        .get("test_mcc"),
        "prior_test_positive_prevalence": config.get("prior_context", {}).get("test_positive_prevalence"),
    }
    for offset, (column, value) in enumerate(result_payload.items()):
        results.insert(offset, column, value)
    _insert_or_assign(predictions, 0, "run_id", run_id)
    _insert_or_assign(predictions, 1, "predeclared_run_id", run_spec["id"])
    _insert_or_assign(predictions, 2, "architecture", run_spec["architecture"])
    _insert_or_assign(history, 0, "run_id", run_id)
    _insert_or_assign(history, 1, "predeclared_run_id", run_spec["id"])
    _insert_or_assign(history, 2, "architecture", run_spec["architecture"])


def _insert_or_assign(df: pd.DataFrame, loc: int, column: str, value: object) -> None:
    if column in df.columns:
        df[column] = value
    else:
        df.insert(loc, column, value)


def _sprint6_reference_row(config: Mapping[str, Any], *, batch_id: str) -> dict[str, object]:
    source = _load_sprint6_reference_from_disk(config)
    prior = config.get("prior_context", {}).get("sprint6_wbce", {})
    return {
        "run_id": f"{batch_id}_{REFERENCE_RUN_ID}",
        "run_order": 0,
        "predeclared_run_id": REFERENCE_RUN_ID,
        "role": "sprint6_weighted_bce_reference_no_retrain",
        "controlled_variable": "model_architecture",
        "source_run_id": source.get("run_id", "S6R0_wbce"),
        "sprint": "sprint7",
        "label_scheme": "scheme_a",
        "split_id": "sprint2_main_seed42",
        "seed": int(config.get("seed", 42)),
        "training_regime": "measured_only",
        "model_name": "gcn_graph_a_sprint6_reference",
        "architecture": "gcn",
        "feature_set": "S5F2_energy",
        "graph_schema": GRAPH_A,
        "visibility_policy": "strict_inductive_primary",
        "target_node_representation": "zero_type_feature",
        "loss": "weighted_bce",
        "checkpoint_policy": "validation_auprc",
        "checkpoint_selection_split": "validation",
        "threshold_policy": "validation_max_f1",
        "threshold_selection_split": "validation",
        "edge_feature_sets": "s5f2_energy",
        "edge_feature_columns": int(source.get("edge_feature_columns", prior.get("edge_feature_columns", 268))),
        "edge_aware_attention": None,
        "attention_heads": None,
        "attention_concat": None,
        "attention_dropout": None,
        "self_loop_edge_fill": None,
        "gatv2_share_weights": None,
        "parameter_count": source.get("parameter_count"),
        "baseline_reference": "xgboost_unweighted / F4",
        "baseline_test_auprc": 0.992522,
        "baseline_test_auroc": 0.938416,
        "baseline_test_mcc": 0.345198,
        "weighted_bce_pos_weight": source.get("weighted_bce_pos_weight", prior.get("weighted_bce_pos_weight")),
        "test_positive_rate": float(source.get("test_positive_rate", prior.get("test_positive_rate", 0.900705))),
        "test_auprc": float(source.get("test_auprc", prior.get("test_auprc", 0.976935))),
        "test_auroc": float(source.get("test_auroc", prior.get("test_auroc", 0.819972))),
        "test_macro_f1": float(source.get("test_macro_f1", prior.get("test_macro_f1", 0.722208))),
        "test_mcc": float(source.get("test_mcc", prior.get("test_mcc", 0.483719))),
        "test_specificity": float(source.get("test_specificity", prior.get("test_specificity", 0.289941))),
        "test_sensitivity": float(source.get("test_sensitivity", prior.get("test_sensitivity", 0.996087))),
        "test_tn": int(source.get("test_tn", prior.get("test_tn", 49))),
        "test_fp": int(source.get("test_fp", prior.get("test_fp", 120))),
        "test_fn": int(source.get("test_fn", prior.get("test_fn", 6))),
        "test_tp": int(source.get("test_tp", prior.get("test_tp", 1527))),
        "notes": "Existing Sprint 6 S6R0 weighted-BCE GCN reference; not retrained in Sprint 7 runner.",
    }


def _load_sprint6_reference_from_disk(config: Mapping[str, Any]) -> dict[str, object]:
    path = ROOT / str(config.get("references", {}).get("sprint6_results_csv", "outputs/sprint6/loss_comparison/sprint6_loss_comparison_results.csv"))
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if "predeclared_run_id" not in table.columns:
        return {}
    rows = table.loc[table["predeclared_run_id"].astype(str) == "S6R0_wbce"]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _write_sprint7_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    attention: pd.DataFrame,
    diagnostics_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    metric_columns = [
        "run_id",
        "predeclared_run_id",
        "architecture",
        "edge_aware_attention",
        "test_auprc",
        "test_auroc",
        "test_macro_f1",
        "test_mcc",
        "test_specificity",
        "test_sensitivity",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]
    threshold_path = diagnostics_dir / "gat_threshold_metrics.csv"
    results[[column for column in metric_columns if column in results.columns]].to_csv(threshold_path, index=False)
    paths.append(threshold_path)

    deltas = results.copy()
    reference = deltas.loc[deltas["predeclared_run_id"] == REFERENCE_RUN_ID]
    reference_auprc = float(reference.iloc[0]["test_auprc"]) if not reference.empty else float("nan")
    deltas["delta_auprc_vs_sprint6_wbce"] = deltas["test_auprc"].astype(float) - reference_auprc
    deltas["delta_auprc_vs_xgboost_f4"] = deltas["test_auprc"].astype(float) - 0.992522
    delta_path = diagnostics_dir / "gat_comparison_deltas.csv"
    deltas[
        [
            "run_id",
            "predeclared_run_id",
            "architecture",
            "test_auprc",
            "delta_auprc_vs_sprint6_wbce",
            "delta_auprc_vs_xgboost_f4",
            "test_mcc",
            "test_specificity",
        ]
    ].to_csv(delta_path, index=False)
    paths.append(delta_path)

    attention_contract_path = diagnostics_dir / "attention_contract_summary.csv"
    if attention.empty:
        pd.DataFrame(
            [
                {
                    "note": "No attention rows were produced; reference-only selection or no trained attention run.",
                    "interpretation": "attention summaries are model-interpretation signals, not causal biological evidence",
                }
            ]
        ).to_csv(attention_contract_path, index=False)
    else:
        attention.to_csv(attention_contract_path, index=False)
    paths.append(attention_contract_path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        per_guide = _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "architecture", "grna_target_id"],
        )
        per_guide_path = diagnostics_dir / "gat_per_guide_score_summary.csv"
        per_guide.to_csv(per_guide_path, index=False)
        paths.append(per_guide_path)
        per_genome = _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "architecture", "genome"],
        )
        per_genome_path = diagnostics_dir / "gat_per_genome_score_summary.csv"
        per_genome.to_csv(per_genome_path, index=False)
        paths.append(per_genome_path)
        deciles = _score_deciles(test_predictions)
        deciles_path = diagnostics_dir / "gat_score_deciles.csv"
        deciles.to_csv(deciles_path, index=False)
        paths.append(deciles_path)
    return paths


def _write_sprint7_figures(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    attention: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    paths: list[Path] = []
    rows = results.sort_values("run_order")
    labels = rows["predeclared_run_id"].astype(str).tolist()

    path = figures_dir / "gat_model_auprc_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, rows["test_auprc"].astype(float))
    ax.axhline(0.992522, color="black", linestyle="--", linewidth=1, label="XGBoost F4")
    ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "gat_threshold_metrics.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_rows = rows.set_index("predeclared_run_id")[["test_mcc", "test_specificity", "test_macro_f1"]].astype(float)
    metric_rows.plot(kind="bar", ax=ax)
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        path = figures_dir / "gat_pr_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for run_id, group in test_predictions.groupby("run_id"):
            precision, recall, _thresholds = precision_recall_curve(group["label"], group["score"])
            ax.plot(recall, precision, label=str(group["predeclared_run_id"].iloc[0]))
        ax.axhline(0.900705, color="gray", linestyle="--", linewidth=1, label="prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "gat_roc_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for run_id, group in test_predictions.groupby("run_id"):
            fpr, tpr, _thresholds = roc_curve(group["label"], group["score"])
            ax.plot(fpr, tpr, label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "gat_score_distributions.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        for run_id, group in test_predictions.groupby("run_id"):
            ax.hist(group["score"], bins=20, alpha=0.45, label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Rows")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "gat_per_guide_metric_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        per_guide = _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "architecture", "grna_target_id"],
        )
        grouped = [
            group["mean_score"].astype(float).to_numpy()
            for _run_id, group in per_guide.groupby("run_id", sort=False)
        ]
        box_labels = [
            str(group["predeclared_run_id"].iloc[0])
            for _run_id, group in per_guide.groupby("run_id", sort=False)
        ]
        ax.boxplot(grouped, tick_labels=box_labels)
        ax.set_ylabel("Per-guide mean test score")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if not history.empty:
        path = figures_dir / "gat_training_curves.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        for run_id, group in history.groupby("run_id"):
            ax.plot(group["epoch"], group["val_auprc"], marker="o", label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation AUPRC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if not attention.empty:
        path = figures_dir / "attention_weight_summary.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot = (
            attention.groupby(["predeclared_run_id", "edge_kind"], dropna=False)["attention_mean"]
            .mean()
            .unstack("edge_kind")
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean attention weight")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_sprint7_report(
    results: pd.DataFrame,
    diagnostic_tables: list[Path],
    figure_paths: list[Path],
    path: Path,
    *,
    batch_id: str,
) -> Path:
    rows = results.sort_values("run_order").copy()
    summary_columns = [
        "predeclared_run_id",
        "architecture",
        "edge_aware_attention",
        "attention_heads",
        "test_auprc",
        "test_auroc",
        "test_macro_f1",
        "test_mcc",
        "test_specificity",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]
    summary = rows[[column for column in summary_columns if column in rows.columns]].copy()
    report = f"""# Sprint 7 GAT/GATv2 Attention Comparison Report

Run batch: `{batch_id}`

## Contract

- Graph schema: `graph_a_minimal_physical_target`.
- Feature set: `S5F2_energy`.
- Loss: Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Split/evaluation: frozen Sprint 2 guide-disjoint split, measured-only headline, validation-only checkpoint and threshold.
- Controlled variable: architecture only (`GATConv` vs `GATv2Conv`), with the Sprint 6 GCN result carried as a no-retrain reference.
- Edge-aware attention: candidate edge features enter message passing via PyG `edge_attr`/`edge_dim`; reverse candidate edges duplicate the same edge features; self-loop edge features are zero-filled.
- Primary metric: AUPRC. Specificity, MCC, and macro F1 are secondary negative-class diagnostics.
- Attention weights are interpretation-only model artifacts, not causal biological evidence.

## Result Summary

{_markdown_table(summary)}

## Artifact Index

Diagnostic tables:
{chr(10).join(f'- `{_relative(item)}`' for item in diagnostic_tables)}

Figures:
{chr(10).join(f'- `{_relative(item)}`' for item in figure_paths)}
"""
    path.write_text(report, encoding="utf-8")
    return path


def _per_group_score_summary(predictions: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    return (
        predictions.groupby(group_columns, dropna=False)
        .agg(
            rows=("label", "size"),
            positives=("label", "sum"),
            positive_rate=("label", "mean"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )


def _score_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, group in predictions.groupby("run_id"):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        bucket_count = min(10, len(ranked))
        if bucket_count == 0:
            continue
        ranked["score_decile"] = pd.qcut(ranked.index + 1, q=bucket_count, labels=False, duplicates="drop") + 1
        for decile, decile_group in ranked.groupby("score_decile"):
            rows.append(
                {
                    "run_id": run_id,
                    "predeclared_run_id": decile_group["predeclared_run_id"].iloc[0],
                    "architecture": decile_group["architecture"].iloc[0],
                    "score_decile": int(decile),
                    "rows": int(len(decile_group)),
                    "positives": int(decile_group["label"].sum()),
                    "positive_rate": float(decile_group["label"].mean()),
                    "mean_score": float(decile_group["score"].mean()),
                    "min_score": float(decile_group["score"].min()),
                    "max_score": float(decile_group["score"].max()),
                }
            )
    return pd.DataFrame(rows)


def _write_graph_provenance(
    path: Path,
    *,
    graph_dir: Path,
    manifest: Mapping[str, Any],
    run_specs: list[Mapping[str, Any]],
) -> None:
    graph_a_dir = graph_dir / GRAPH_A
    payload = {
        "provenance_type": "sprint7_graph_a_s5f2_energy_artifacts_sha256",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "artifact_dir": _relative(graph_dir),
        "graph_schema": manifest.get("graph_name"),
        "split_id": manifest.get("split_id"),
        "label_scheme": manifest.get("label_scheme"),
        "visibility_policy": manifest.get("metadata", {}).get("visibility_policy"),
        "feature_set": "S5F2_energy",
        "feature_table_columns": {
            name: manifest.get("feature_tables", {}).get(name)
            for name in ["S5F2_energy", "s5f2_energy"]
            if name in manifest.get("feature_tables", {})
        },
        "run_specs": [
            {
                "id": run["id"],
                "architecture": run["architecture"],
                "edge_aware_attention": run.get("edge_aware_attention", True),
                "attention": run.get("attention", {}),
                "role": run.get("role"),
            }
            for run in run_specs
        ],
        "metadata": manifest.get("metadata", {}),
        "preprocessing": manifest.get("preprocessing", {}),
        "files": _artifact_file_hashes(graph_a_dir),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_manifest(
    path: Path,
    *,
    config_path: Path,
    batch_id: str,
    runs: list[dict[str, object]],
    results_path: Path,
    predictions_path: Path,
    history_path: Path,
    attention_path: Path,
    report_path: Path,
    diagnostics: list[Path],
    figures: list[Path],
    provenance_path: Path,
    optional_runs_available: list[str],
) -> None:
    payload = {
        "manifest_type": "sprint7_gat_gatv2_run_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "batch_id": batch_id,
        "config_path": _relative(config_path),
        "reference_run_id": REFERENCE_RUN_ID,
        "headline_run_ids": list(HEADLINE_RUN_IDS),
        "optional_runs_available": optional_runs_available,
        "optional_runs_executed": [run["predeclared_id"] for run in runs if run["predeclared_id"] not in [REFERENCE_RUN_ID, *HEADLINE_RUN_IDS]],
        "runs": runs,
        "results_path": _relative(results_path),
        "predictions_path": _relative(predictions_path),
        "training_history_path": _relative(history_path),
        "attention_summary_path": _relative(attention_path),
        "report_path": _relative(report_path),
        "graph_artifact_provenance_path": _relative(provenance_path),
        "diagnostic_tables": [_relative(item) for item in diagnostics],
        "figures": [_relative(item) for item in figures],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_info(device: str) -> dict[str, object]:
    import torch

    try:
        import torch_geometric as pyg

        pyg_version: str = pyg.__version__
    except ImportError:
        pyg_version = "unavailable"
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": device,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_geometric_version": pyg_version,
        "torch_version": torch.__version__,
    }


def _artifact_file_hashes(artifact_dir: Path) -> list[dict[str, object]]:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Graph A artifact directory not found: {artifact_dir}")
    rows = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(artifact_dir).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    if not rows:
        raise ValueError(f"No artifact files found under {artifact_dir}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_id(config: Mapping[str, Any]) -> str:
    configured = config.get("run_id")
    if configured and str(configured).strip():
        return str(configured)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    return f"sprint7_gat_gatv2_seed{int(config.get('seed', 42))}_{ts}"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True), encoding="utf-8")


def _validate_consolidated_run_ids(results: pd.DataFrame) -> None:
    if "run_id" not in results.columns:
        raise ValueError("Consolidated Sprint 7 results must include run_id")
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Consolidated Sprint 7 results require unique run_id values")


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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
