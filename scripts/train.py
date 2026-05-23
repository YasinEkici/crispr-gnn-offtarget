from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.evaluation.plots import write_baseline_plots  # noqa: E402
from crispr_gnn.features.tabular import FEATURE_SET_ORDER  # noqa: E402
from crispr_gnn.training.baselines import BaselineRunConfig, run_dummy_and_logistic_baselines  # noqa: E402
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

    print("Training placeholder: this config does not yet map to an implemented task.")
    return 0 if args.debug else 1


def run_sprint2_dummy_logistic(config: dict[str, object]) -> int:
    data_config = load_yaml(str(config["data_config"]))
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / str(config.get("split_manifest", "outputs/splits/sprint2_guides.json"))
    results_path = ROOT / str(config.get("results_path", "outputs/results/baseline_results.csv"))
    figures_dir = ROOT / str(config.get("figures_dir", "outputs/figures/sprint2"))
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

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    figure_paths = write_baseline_plots(results, predictions, figures_dir)

    print(f"Results written: {results_path.relative_to(ROOT)}")
    for path in figure_paths:
        print(f"Figure written: {path.relative_to(ROOT)}")
    print(results[["model_name", "feature_set", "test_auprc", "test_auroc", "test_f1", "test_mcc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
