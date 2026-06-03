from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.evaluation.diagnostics import write_gcn_diagnostics, write_gcn_report  # noqa: E402
from crispr_gnn.evaluation.plots import write_gcn_plots  # noqa: E402
from crispr_gnn.features.sprint5 import SPRINT5_FEATURE_SET_ORDER, normalize_sprint5_feature_set  # noqa: E402
from crispr_gnn.graph.graph_schemas import GRAPH_A  # noqa: E402
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader, validate_gcn_headline_config  # noqa: E402
from crispr_gnn.training.gcn import gcn_run_config_from_mapping, train_graph_a_gcn  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 5 Graph A feature-ablation GCN sweep.")
    parser.add_argument("--config", default="configs/sweeps/sprint5_graph_a_feature_ablation.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for smoke/debug runs.")
    parser.add_argument("--feature-set", action="append", default=None, help="Run only this Sprint 5 feature set. May be repeated.")
    parser.add_argument("--run-id", default=None, help="Override the output run batch ID.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    validate_gcn_headline_config(config)
    if config.get("sprint") != "sprint5":
        raise ValueError("Sprint 5 ablation runner requires sprint: sprint5")
    if config.get("graph", {}).get("schema") != GRAPH_A:
        raise ValueError("Sprint 5 primary ablation runner is Graph A only")

    run_batch_id = args.run_id or _batch_id(config)
    feature_sets = _feature_sets(config, args.feature_set)
    graph_dir = ROOT / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_A)
    _validate_sprint5_feature_tables(materialized.manifest, feature_sets)

    output_base = ROOT / str(config.get("outputs", {}).get("output_dir", "outputs/sprint5/graph_a_feature_ablation"))
    output_dir = output_base / run_batch_id
    diagnostics_dir = output_dir / "diagnostics_sprint5_graph_a"
    figures_dir = output_dir / "figures_sprint5_graph_a"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        run_config_mapping = _config_for_feature_set(config, feature_set)
        if args.max_epochs is not None:
            run_config_mapping.setdefault("training", {})["max_epochs"] = args.max_epochs
            run_config_mapping.setdefault("training", {})["min_epochs"] = 1
            run_config_mapping.setdefault("training", {})["patience"] = min(2, int(args.max_epochs))
        run_config = gcn_run_config_from_mapping(run_config_mapping)
        run_id = f"{run_batch_id}_{feature_set.lower()}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = run_dir / "resolved_config.yaml"
        resolved_config_path.write_text(
            yaml.safe_dump(run_config_mapping, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        runtime_path = run_dir / "runtime.json"
        runtime_path.write_text(
            json.dumps(_runtime_info(str(run_config.device)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_path = run_dir / "model.pt"
        results, predictions, history = train_graph_a_gcn(
            materialized,
            run_config,
            checkpoint_path=checkpoint_path,
        )
        results.insert(0, "run_id", run_id)
        predictions.insert(0, "run_id", run_id)
        history.insert(0, "run_id", run_id)
        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)
        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        print(
            f"{feature_set}: test_auprc={float(results.iloc[0]['test_auprc']):.6f}, "
            f"test_macro_f1={float(results.iloc[0]['test_macro_f1']):.6f}, "
            f"tn={int(results.iloc[0]['test_tn'])}"
        )

    result_table = pd.concat(all_results, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    history_table = pd.concat(all_history, ignore_index=True)
    results_path = output_dir / "sprint5_graph_a_feature_ablation_results.csv"
    predictions_path = diagnostics_dir / "sprint5_graph_a_predictions.csv"
    history_path = diagnostics_dir / "sprint5_graph_a_training_history.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)

    diagnostic_tables = write_gcn_diagnostics(result_table, prediction_table, diagnostics_dir, schema_label="sprint5_graph_a")
    figure_paths = write_gcn_plots(
        result_table,
        prediction_table,
        history_table,
        figures_dir,
        schema_label="sprint5_graph_a",
        graph_view=materialized.view("test"),
    )
    report_path = output_dir / "sprint5_graph_a_feature_ablation_report.md"
    report = write_gcn_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        report_path,
        run_label=run_batch_id,
        root=ROOT,
        title="Sprint 5 Graph A Feature Ablation Report",
        evidence_label="Sprint 5",
    )
    _write_provenance(output_dir / "graph_artifact_provenance.json", materialized.manifest, feature_sets)

    print(f"Run batch: {run_batch_id}")
    print(f"Output directory: {output_dir.relative_to(ROOT)}")
    print(f"Results: {results_path.relative_to(ROOT)}")
    print(f"Predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Training history: {history_path.relative_to(ROOT)}")
    print(f"Report: {report.relative_to(ROOT)}")
    print(result_table[["feature_set", "test_auprc", "test_auroc", "test_f1", "test_macro_f1", "test_mcc", "test_specificity", "test_tn", "test_fp", "test_fn", "test_tp"]].to_string(index=False))
    return 0


def _batch_id(config: dict[str, Any]) -> str:
    configured = config.get("run_id")
    if configured and str(configured).strip():
        return str(configured)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    return f"sprint5_graph_a_gcn_seed{int(config.get('seed', 42))}_{ts}"


def _feature_sets(config: dict[str, Any], selected: list[str] | None) -> list[str]:
    if selected:
        return [normalize_sprint5_feature_set(value) for value in selected]
    configured = config.get("feature_sets", SPRINT5_FEATURE_SET_ORDER)
    return [normalize_sprint5_feature_set(str(value)) for value in configured]


def _config_for_feature_set(config: dict[str, Any], feature_set: str) -> dict[str, Any]:
    run_config = deepcopy(config)
    run_config["features"] = {
        "edge_feature_sets": [feature_set.lower()],
        "feature_set": feature_set,
    }
    run_config["model"] = dict(run_config.get("model", {}))
    run_config["model"]["name"] = "gcn_graph_a_sprint5"
    return run_config


def _validate_sprint5_feature_tables(manifest: dict[str, Any], feature_sets: list[str]) -> None:
    available = set(manifest.get("feature_tables", {}))
    missing = sorted(set(feature_sets).difference(available))
    if missing:
        raise ValueError(
            "Sprint 5 feature tables are missing from Graph A artifacts. "
            f"Run scripts/build_sprint5_graph_a_features.py first. Missing={missing}"
        )
    metadata = manifest.get("metadata", {}).get("sprint5_feature_ablation", {})
    if metadata.get("topology") != "Graph A unchanged":
        raise ValueError("Sprint 5 Graph A artifact metadata is missing fixed-topology provenance")


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


def _write_provenance(path: Path, manifest: dict[str, Any], feature_sets: list[str]) -> None:
    payload = {
        "graph_name": manifest.get("graph_name"),
        "split_id": manifest.get("split_id"),
        "label_scheme": manifest.get("label_scheme"),
        "visibility_policy": manifest.get("metadata", {}).get("visibility_policy"),
        "sprint5_feature_sets": feature_sets,
        "feature_table_columns": {name: manifest.get("feature_tables", {}).get(name) for name in feature_sets},
        "preprocessing": manifest.get("preprocessing", {}).get("sprint5_feature_ablation", {}),
        "metadata": manifest.get("metadata", {}).get("sprint5_feature_ablation", {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
