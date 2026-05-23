from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.evaluation.diagnostics import write_logistic_regression_diagnostics, write_model_diagnostics  # noqa: E402
from crispr_gnn.evaluation.plots import write_baseline_plots, write_xgboost_plots  # noqa: E402
from crispr_gnn.features.tabular import FEATURE_SET_ORDER  # noqa: E402
from crispr_gnn.training.baselines import BaselineRunConfig, XGBoostRunConfig, run_dummy_and_logistic_baselines, run_xgboost_baselines  # noqa: E402
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
    max_epochs = args.max_epochs or config.get("training", {}).get("max_epochs", 1)
    task = config.get("task", "placeholder")

    print(f"Experiment: {experiment_name}")
    print(f"Task: {task}")
    print(f"Debug mode: {args.debug}")
    print(f"Max epochs: {max_epochs}")

    if task == "sprint2_dummy_logistic":
        return run_sprint2_dummy_logistic(config)
    if task == "sprint2_xgboost":
        return run_sprint2_xgboost(config)

    print("Training placeholder: this config does not yet map to an implemented task.")
    return 0 if args.debug else 1


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

    results, predictions = run_xgboost_baselines(
        assigned=assigned,
        feature_sets=feature_sets,
        config=baseline_config,
        include_balanced=bool(config.get("include_balanced_train_weights", True)),
    )

    write_results_table(results, results_path)
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
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    for path in diagnostic_tables:
        print(f"Diagnostic table written: {path.relative_to(ROOT)}")
    for path in diagnostic_figures:
        print(f"Diagnostic figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


def write_results_table(results: pd.DataFrame, path: Path) -> None:
    key_columns = ["sprint", "label_scheme", "split_id", "seed", "training_regime", "model_name", "feature_set"]
    output = results.copy()
    if path.exists():
        existing = pd.read_csv(path)
        if set(key_columns).issubset(existing.columns):
            replacement_keys = set(map(tuple, output[key_columns].itertuples(index=False, name=None)))
            keep_mask = ~existing[key_columns].apply(tuple, axis=1).isin(replacement_keys)
            output = pd.concat([existing.loc[keep_mask], output], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
