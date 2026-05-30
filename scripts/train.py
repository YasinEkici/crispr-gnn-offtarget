from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.evaluation.diagnostics import write_gcn_diagnostics, write_gcn_report, write_logistic_regression_diagnostics, write_model_diagnostics  # noqa: E402
from crispr_gnn.evaluation.plots import write_baseline_plots, write_gcn_plots, write_mlp_plots, write_sequence_plots, write_xgboost_plots  # noqa: E402
from crispr_gnn.features.tabular import FEATURE_SET_ORDER  # noqa: E402
from crispr_gnn.training.baselines import BaselineRunConfig, MLPRunConfig, XGBoostRunConfig, run_dummy_and_logistic_baselines, run_tabular_mlp_baselines, run_xgboost_baselines  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--debug", action="store_true", help="Run a lightweight smoke training path.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    experiment_name = config.get("experiment_name", Path(args.config).stem)
    task = config.get("task", "placeholder")
    if args.max_epochs is not None:
        if task in {"sprint2_mlp", "sprint2_sequence", "sprint2_sequence_late_fusion"}:
            key = "mlp" if task == "sprint2_mlp" else "sequence"
            config.setdefault(key, {})["max_epochs"] = args.max_epochs
        else:
            config.setdefault("training", {})["max_epochs"] = args.max_epochs
    max_epochs = _display_max_epochs(config, task)

    print(f"Experiment: {experiment_name}")
    print(f"Task: {task}")
    print(f"Debug mode: {args.debug}")
    print(f"Max epochs: {max_epochs}")

    if task == "sprint2_dummy_logistic":
        return run_sprint2_dummy_logistic(config)
    if task == "sprint2_xgboost":
        return run_sprint2_xgboost(config)
    if task == "sprint2_mlp":
        return run_sprint2_mlp(config)
    if task == "sprint2_sequence":
        return run_sprint2_sequence(config)
    if task == "sprint2_sequence_late_fusion":
        return run_sprint2_sequence_late_fusion(config)
    if task == "sprint4_gcn":
        return run_sprint4_gcn(config)

    print("Training placeholder: this config does not yet map to an implemented task.")
    return 0 if args.debug else 1


def _display_max_epochs(config: dict[str, object], task: object) -> object:
    if task == "sprint2_mlp":
        mlp_config = config.get("mlp", {})
        if isinstance(mlp_config, dict):
            return mlp_config.get("max_epochs", "n/a")
    if task in {"sprint2_sequence", "sprint2_sequence_late_fusion"}:
        sequence_config = config.get("sequence", {})
        if isinstance(sequence_config, dict):
            return sequence_config.get("max_epochs", "n/a")
    training_config = config.get("training", {})
    if isinstance(training_config, dict):
        return training_config.get("max_epochs", "n/a")
    return "n/a"


def run_sprint2_dummy_logistic(config: dict[str, object]) -> int:
    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint2"))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    split = load_split_manifest(split_path)
    df = pd.read_parquet(raw_path)
    assigned = assign_measured_splits(df, split)
    feature_sets = list(config.get("feature_sets", FEATURE_SET_ORDER))
    logistic_config = config.get("logistic_regression", {})
    if not isinstance(logistic_config, dict):
        raise ValueError("logistic_regression config must be a mapping")
    baseline_config = BaselineRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(config.get("seed", split.config.seed)),
        logistic_max_iter=int(logistic_config.get("max_iter", 2_000)),
        logistic_class_weight=logistic_config.get("class_weight"),
    )

    results, predictions = run_dummy_and_logistic_baselines(
        assigned=assigned,
        feature_sets=feature_sets,
        config=baseline_config,
    )

    write_results_table(results, results_path)
    figure_paths = write_baseline_plots(results, predictions, figures_dir)
    diagnostic_tables, diagnostic_figures = write_logistic_regression_diagnostics(assigned, predictions, diagnostics_dir)

    print(f"Results written: {results_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def run_sprint2_xgboost(config: dict[str, object]) -> int:
    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint2"))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    split = load_split_manifest(split_path)
    df = pd.read_parquet(raw_path)
    assigned = assign_measured_splits(df, split)
    feature_sets = list(config.get("feature_sets", FEATURE_SET_ORDER))
    xgb_config = config.get("xgboost", {})
    if not isinstance(xgb_config, dict):
        raise ValueError("xgboost config must be a mapping")
    baseline_config = XGBoostRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(config.get("seed", split.config.seed)),
        n_estimators=int(xgb_config.get("n_estimators", 400)),
        max_depth=int(xgb_config.get("max_depth", 3)),
        learning_rate=float(xgb_config.get("learning_rate", 0.05)),
        subsample=float(xgb_config.get("subsample", 0.9)),
        colsample_bytree=float(xgb_config.get("colsample_bytree", 0.9)),
        min_child_weight=float(xgb_config.get("min_child_weight", 5.0)),
        reg_alpha=float(xgb_config.get("reg_alpha", 0.0)),
        reg_lambda=float(xgb_config.get("reg_lambda", 1.0)),
        early_stopping_rounds=xgb_config.get("early_stopping_rounds", 30),
        eval_metric=str(xgb_config.get("eval_metric", "aucpr")),
        tree_method=str(xgb_config.get("tree_method", "hist")),
        n_jobs=int(xgb_config.get("n_jobs", 4)),
    )

    results, predictions, feature_importance, feature_audit = run_xgboost_baselines(
        assigned=assigned,
        feature_sets=feature_sets,
        config=baseline_config,
        include_balanced=bool(config.get("include_balanced_train_weights", True)),
    )

    write_results_table(results, results_path)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_path = diagnostics_dir / "xgboost_feature_importance.csv"
    feature_audit_path = diagnostics_dir / "xgboost_feature_column_audit.csv"
    feature_importance.to_csv(feature_importance_path, index=False)
    feature_audit.to_csv(feature_audit_path, index=False)
    if feature_audit["is_forbidden"].any():
        forbidden = feature_audit.loc[feature_audit["is_forbidden"], "feature"].unique().tolist()
        raise ValueError(f"Forbidden columns found in XGBoost feature audit: {forbidden}")
    figure_paths = write_xgboost_plots(results, predictions, figures_dir)
    diagnostic_tables: list[Path] = []
    diagnostic_figures: list[Path] = []
    for model_name, display_name in [
        ("xgboost_unweighted", "XGBoost unweighted"),
        ("xgboost_balanced_train_weights", "XGBoost balanced train weights"),
    ]:
        if model_name in set(results["model_name"]):
            tables, figures = write_model_diagnostics(
                assigned,
                predictions,
                diagnostics_dir,
                model_name=model_name,
                artifact_prefix=model_name,
                display_name=display_name,
            )
            diagnostic_tables.extend(tables)
            diagnostic_figures.extend(figures)

    print(f"Results upserted: {results_path.relative_to(ROOT)}")
    print(f"Feature importance written: {feature_importance_path.relative_to(ROOT)}")
    print(f"Feature audit written: {feature_audit_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def run_sprint2_mlp(config: dict[str, object]) -> int:
    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint2"))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    split = load_split_manifest(split_path)
    df = pd.read_parquet(raw_path)
    assigned = assign_measured_splits(df, split)
    feature_sets = list(config.get("feature_sets", FEATURE_SET_ORDER))
    mlp_config = config.get("mlp", {})
    if not isinstance(mlp_config, dict):
        raise ValueError("mlp config must be a mapping")
    baseline_config = MLPRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(config.get("seed", split.config.seed)),
        hidden_dims=tuple(int(value) for value in mlp_config.get("hidden_dims", [128, 64])),
        learning_rate=float(mlp_config.get("learning_rate", 1e-3)),
        alpha=float(mlp_config.get("alpha", 1e-4)),
        batch_size=int(mlp_config.get("batch_size", 512)),
        max_epochs=int(mlp_config.get("max_epochs", 200)),
        min_epochs=int(mlp_config.get("min_epochs", 20)),
        patience=int(mlp_config.get("patience", 25)),
    )

    results, predictions, training_summary, feature_audit = run_tabular_mlp_baselines(
        assigned=assigned,
        feature_sets=feature_sets,
        config=baseline_config,
        include_balanced=bool(config.get("include_balanced_train_weights", True)),
        balanced_feature_sets=list(config.get("balanced_feature_sets", ["F3", "F4"])),
    )

    write_results_table(results, results_path)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    training_summary_path = diagnostics_dir / "tabular_mlp_training_summary.csv"
    feature_audit_path = diagnostics_dir / "tabular_mlp_feature_column_audit.csv"
    training_summary.to_csv(training_summary_path, index=False)
    feature_audit.to_csv(feature_audit_path, index=False)
    if feature_audit["is_forbidden"].any():
        forbidden = feature_audit.loc[feature_audit["is_forbidden"], "feature"].unique().tolist()
        raise ValueError(f"Forbidden columns found in tabular MLP feature audit: {forbidden}")
    figure_paths = write_mlp_plots(results, predictions, figures_dir)
    diagnostic_tables: list[Path] = []
    diagnostic_figures: list[Path] = []
    for model_name, display_name in [
        ("tabular_mlp_unweighted", "Tabular MLP unweighted"),
        ("tabular_mlp_balanced_train_weights", "Tabular MLP balanced train weights"),
    ]:
        if model_name in set(results["model_name"]):
            tables, figures = write_model_diagnostics(
                assigned,
                predictions,
                diagnostics_dir,
                model_name=model_name,
                artifact_prefix=model_name,
                display_name=display_name,
            )
            diagnostic_tables.extend(tables)
            diagnostic_figures.extend(figures)

    print(f"Results upserted: {results_path.relative_to(ROOT)}")
    print(f"Training summary written: {training_summary_path.relative_to(ROOT)}")
    print(f"Feature audit written: {feature_audit_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def run_sprint2_sequence(config: dict[str, object]) -> int:
    from crispr_gnn.training.sequence import SequenceRunConfig, run_sequence_baselines

    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint2"))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    split = load_split_manifest(split_path)
    df = pd.read_parquet(raw_path)
    assigned = assign_measured_splits(df, split)
    sequence_config = config.get("sequence", {})
    if not isinstance(sequence_config, dict):
        raise ValueError("sequence config must be a mapping")
    baseline_config = SequenceRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(config.get("seed", split.config.seed)),
        max_length=int(sequence_config.get("max_length", 23)),
        batch_size=int(sequence_config.get("batch_size", 512)),
        max_epochs=int(sequence_config.get("max_epochs", 120)),
        min_epochs=int(sequence_config.get("min_epochs", 15)),
        patience=int(sequence_config.get("patience", 20)),
        learning_rate=float(sequence_config.get("learning_rate", 1e-3)),
        weight_decay=float(sequence_config.get("weight_decay", 1e-4)),
        cnn_channels=int(sequence_config.get("cnn_channels", 64)),
        lstm_hidden_dim=int(sequence_config.get("lstm_hidden_dim", 64)),
        lstm_layers=int(sequence_config.get("lstm_layers", 1)),
        dropout=float(sequence_config.get("dropout", 0.2)),
        device=str(sequence_config.get("device", "cpu")),
        num_threads=int(sequence_config.get("num_threads", 1)),
    )

    results, predictions, training_summary, input_audit = run_sequence_baselines(
        assigned=assigned,
        config=baseline_config,
        models=list(config.get("models", ["sequence_cnn", "sequence_bilstm"])),
        include_balanced=bool(config.get("include_balanced_train_weights", False)),
    )

    write_results_table(results, results_path)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    training_summary_path = diagnostics_dir / "sequence_training_summary.csv"
    input_audit_path = diagnostics_dir / "sequence_input_audit.csv"
    training_summary.to_csv(training_summary_path, index=False)
    input_audit.to_csv(input_audit_path, index=False)
    if input_audit["is_forbidden"].any():
        forbidden = input_audit.loc[input_audit["is_forbidden"], "source_column"].unique().tolist()
        raise ValueError(f"Forbidden columns found in sequence input audit: {forbidden}")
    figure_paths = write_sequence_plots(results, predictions, figures_dir)
    diagnostic_tables: list[Path] = []
    diagnostic_figures: list[Path] = []
    for model_name in sorted(results["model_name"].unique().tolist()):
        display_name = model_name.replace("_", " ").title()
        tables, figures = write_model_diagnostics(
            assigned,
            predictions,
            diagnostics_dir,
            model_name=model_name,
            artifact_prefix=model_name,
            display_name=display_name,
        )
        diagnostic_tables.extend(tables)
        diagnostic_figures.extend(figures)

    print(f"Results upserted: {results_path.relative_to(ROOT)}")
    print(f"Training summary written: {training_summary_path.relative_to(ROOT)}")
    print(f"Input audit written: {input_audit_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def run_sprint2_sequence_late_fusion(config: dict[str, object]) -> int:
    from crispr_gnn.training.sequence import SequenceRunConfig, run_sequence_cnn_late_fusion_baselines

    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint2"))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    split = load_split_manifest(split_path)
    df = pd.read_parquet(raw_path)
    assigned = assign_measured_splits(df, split)
    sequence_config = config.get("sequence", {})
    if not isinstance(sequence_config, dict):
        raise ValueError("sequence config must be a mapping")
    baseline_config = SequenceRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(config.get("seed", split.config.seed)),
        max_length=int(sequence_config.get("max_length", 23)),
        batch_size=int(sequence_config.get("batch_size", 1024)),
        max_epochs=int(sequence_config.get("max_epochs", 50)),
        min_epochs=int(sequence_config.get("min_epochs", 8)),
        patience=int(sequence_config.get("patience", 8)),
        learning_rate=float(sequence_config.get("learning_rate", 1e-3)),
        weight_decay=float(sequence_config.get("weight_decay", 1e-4)),
        cnn_channels=int(sequence_config.get("cnn_channels", 32)),
        dropout=float(sequence_config.get("dropout", 0.2)),
        tabular_projection_dim=int(sequence_config.get("tabular_projection_dim", 32)),
        device=str(sequence_config.get("device", "cpu")),
        num_threads=int(sequence_config.get("num_threads", 4)),
    )

    results, predictions, training_summary, input_audit = run_sequence_cnn_late_fusion_baselines(
        assigned=assigned,
        config=baseline_config,
        tabular_feature_sets=list(config.get("tabular_feature_sets", ["F3", "F4"])),
    )

    write_results_table(results, results_path)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    training_summary_path = diagnostics_dir / "sequence_late_fusion_training_summary.csv"
    input_audit_path = diagnostics_dir / "sequence_late_fusion_input_audit.csv"
    training_summary.to_csv(training_summary_path, index=False)
    input_audit.to_csv(input_audit_path, index=False)
    if input_audit["is_forbidden"].any():
        forbidden = input_audit.loc[input_audit["is_forbidden"], "source_column"].unique().tolist()
        raise ValueError(f"Forbidden columns found in sequence late-fusion input audit: {forbidden}")
    figure_paths = write_sequence_plots(results, predictions, figures_dir, artifact_prefix="sequence_late_fusion")
    diagnostic_tables: list[Path] = []
    diagnostic_figures: list[Path] = []
    for model_name in sorted(results["model_name"].unique().tolist()):
        display_name = model_name.replace("_", " ").title()
        tables, figures = write_model_diagnostics(
            assigned,
            predictions,
            diagnostics_dir,
            model_name=model_name,
            artifact_prefix=model_name,
            display_name=display_name,
        )
        diagnostic_tables.extend(tables)
        diagnostic_figures.extend(figures)

    print(f"Results upserted: {results_path.relative_to(ROOT)}")
    print(f"Training summary written: {training_summary_path.relative_to(ROOT)}")
    print(f"Input audit written: {input_audit_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def run_sprint4_gcn(config: dict[str, object]) -> int:
    from crispr_gnn.graph.graph_schemas import GRAPH_A
    from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader, validate_gcn_headline_config
    from crispr_gnn.training.gcn import gcn_run_config_from_mapping, train_graph_a_gcn

    results_path = ROOT / str(config.get("results_path", "outputs/results/gcn_results.csv"))
    diagnostics_dir = ROOT / str(config.get("diagnostics_dir", "outputs/diagnostics/sprint4"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint4"))
    report_path = ROOT / str(config.get("report_path", "outputs/reports/gcn_report.md"))
    runs_base = ROOT / str(config.get("runs_dir", "outputs/runs"))

    validate_gcn_headline_config(config)
    run_config = gcn_run_config_from_mapping(config)

    run_id = _sprint4_run_id(config, run_config)
    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    runtime_path = run_dir / "runtime.json"
    runtime_path.write_text(
        json.dumps(_sprint4_runtime_info(str(run_config.device)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    graph_dir = ROOT / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_A)
    checkpoint_path = run_dir / "model.pt"

    results, predictions, training_history = train_graph_a_gcn(
        materialized, run_config, checkpoint_path=checkpoint_path
    )

    write_results_table(results, results_path)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = diagnostics_dir / "gcn_graph_a_predictions.csv"
    training_history_path = diagnostics_dir / "gcn_graph_a_training_history.csv"
    predictions.to_csv(predictions_path, index=False)
    training_history.to_csv(training_history_path, index=False)
    training_history.to_csv(run_dir / "training_history.csv", index=False)

    diagnostic_tables = write_gcn_diagnostics(results, predictions, diagnostics_dir)
    figure_paths = write_gcn_plots(results, predictions, training_history, figures_dir, graph_view=materialized.view("test"))
    report = write_gcn_report(results, diagnostic_tables, figure_paths, report_path, run_label=run_id)

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir.relative_to(ROOT)}")
    print(f"Resolved config: {resolved_config_path.relative_to(ROOT)}")
    print(f"Runtime info: {runtime_path.relative_to(ROOT)}")
    print(f"Results upserted: {results_path.relative_to(ROOT)}")
    print(f"Checkpoint: {checkpoint_path.relative_to(ROOT)}")
    print(f"Predictions written: {predictions_path.relative_to(ROOT)}")
    print(f"Training history written: {training_history_path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    print(f"Report written: {report.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def _sprint4_run_id(config: dict[str, object], run_config: object) -> str:
    configured = config.get("run_id")
    if configured and str(configured).strip():
        return str(configured)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    sprint = getattr(run_config, "sprint", "sprint4")
    schema = getattr(run_config, "graph_schema", "graph_a")
    model = getattr(run_config, "model_name", "gcn")
    seed = getattr(run_config, "seed", 42)
    return f"{sprint}_{schema}_{model}_seed{seed}_{ts}"


def _sprint4_runtime_info(device: str) -> dict[str, object]:
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


def write_results_table(results: pd.DataFrame, path: Path) -> None:
    key_columns = ["sprint", "label_scheme", "split_id", "seed", "training_regime", "model_name", "feature_set"]
    if "graph_schema" in results.columns:
        key_columns.append("graph_schema")
    output = results.copy()
    if path.exists():
        existing = pd.read_csv(path)
        if set(key_columns).issubset(existing.columns):
            replacement_keys = set(map(tuple, output[key_columns].itertuples(index=False, name=None)))
            keep_mask = ~existing[key_columns].apply(tuple, axis=1).isin(replacement_keys)
            output = pd.concat([existing.loc[keep_mask], output], ignore_index=True)
        else:
            output = pd.concat([existing, output], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
