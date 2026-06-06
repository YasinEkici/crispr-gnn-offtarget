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

from crispr_gnn.evaluation.diagnostics import write_sprint6_imbalance_diagnostics  # noqa: E402
from crispr_gnn.evaluation.plots import write_sprint6_imbalance_plots  # noqa: E402
from crispr_gnn.graph.graph_schemas import GRAPH_A  # noqa: E402
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader, validate_gcn_headline_config  # noqa: E402
from crispr_gnn.training.gcn import gcn_run_config_from_mapping, train_graph_a_gcn  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


HEADLINE_RUN_IDS = (
    "S6R0_wbce",
    "S6R1_bce_unw",
    "S6R2_focal_g2_a25",
    "S6R3_focal_g1_a25",
    "S6R4_focal_g2_a50",
    "S6R5_dice",
    "S6R6_tversky_a70_b30",
    "S6R7_balanced_sampling",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 6 Graph A S5F2 loss-comparison sweep.")
    parser.add_argument("--config", default="configs/sweeps/sprint6_loss_comparison.yaml")
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
    run_sprint6_loss_comparison(
        config_path=ROOT / args.config,
        batch_id=args.run_id,
        max_epochs=args.max_epochs,
        selected_run_ids=args.run,
        include_optional_runs=args.include_optional_runs,
    )
    return 0


def run_sprint6_loss_comparison(
    *,
    config_path: str | Path,
    batch_id: str | None = None,
    max_epochs: int | None = None,
    selected_run_ids: list[str] | None = None,
    include_optional_runs: bool = False,
) -> Path:
    config = load_yaml(config_path)
    _validate_sprint6_base_config(config)
    run_batch_id = batch_id or _batch_id(config)
    run_specs = _selected_run_specs(config, selected_run_ids, include_optional_runs=include_optional_runs)

    graph_dir = ROOT / str(config["data"]["graph_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_dir).load(GRAPH_A)
    _validate_sprint6_graph_artifacts(materialized.manifest)

    output_dir = ROOT / str(config.get("outputs", {}).get("output_dir", "outputs/sprint6/loss_comparison"))
    diagnostics_dir = output_dir / "diagnostics_sprint6"
    figures_dir = output_dir / "figures_sprint6"
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, object]] = []
    for order, run_spec in enumerate(run_specs):
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
        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)
        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        manifest_runs.append(
            {
                "run_id": run_id,
                "predeclared_id": run_spec["id"],
                "loss": run_spec["loss"],
                "loss_params": run_spec.get("params", {}),
                "sampling": run_spec.get("sampling"),
                "role": run_spec.get("role"),
                "metrics_path": _relative(run_dir / "metrics.csv"),
                "training_history_path": _relative(run_dir / "training_history.csv"),
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
    prediction_table = pd.concat(all_predictions, ignore_index=True)
    history_table = pd.concat(all_history, ignore_index=True)
    _validate_consolidated_run_ids(result_table)

    results_path = output_dir / "sprint6_loss_comparison_results.csv"
    predictions_path = diagnostics_dir / "sprint6_loss_comparison_predictions.csv"
    history_path = diagnostics_dir / "sprint6_loss_comparison_training_history.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)

    diagnostic_tables = write_sprint6_imbalance_diagnostics(result_table, prediction_table, diagnostics_dir)
    figure_paths = write_sprint6_imbalance_plots(result_table, prediction_table, history_table, figures_dir)
    report_path = _write_sprint6_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        output_dir / "sprint6_loss_comparison_report.md",
        batch_id=run_batch_id,
    )
    provenance_path = output_dir / "graph_artifact_provenance.json"
    _write_graph_provenance(
        provenance_path,
        graph_dir=graph_dir,
        manifest=materialized.manifest,
        run_specs=run_specs,
    )
    manifest_path = output_dir / "sprint6_loss_comparison_run_manifest.json"
    _write_run_manifest(
        manifest_path,
        config_path=Path(config_path),
        batch_id=run_batch_id,
        runs=manifest_runs,
        results_path=results_path,
        predictions_path=predictions_path,
        history_path=history_path,
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
                "loss",
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


def _validate_sprint6_base_config(config: Mapping[str, Any]) -> None:
    validate_gcn_headline_config(config)
    if config.get("sprint") != "sprint6":
        raise ValueError("Sprint 6 loss-comparison runner requires sprint: sprint6")
    if config.get("task") != "sprint6_loss_comparison":
        raise ValueError("Sprint 6 loss-comparison runner requires task: sprint6_loss_comparison")
    if config.get("graph", {}).get("schema") != GRAPH_A:
        raise ValueError("Sprint 6 loss comparison is frozen to Graph A")
    features = config.get("features", {})
    if features.get("feature_set") != "S5F2_energy" or features.get("edge_feature_sets") != ["s5f2_energy"]:
        raise ValueError("Sprint 6 loss comparison is frozen to S5F2_energy")
    evaluation = config.get("evaluation", {})
    if evaluation.get("regime") != "measured_only":
        raise ValueError("Sprint 6 headline loss comparison must remain measured-only")
    if evaluation.get("threshold_policy") != "validation_max_f1":
        raise ValueError("Sprint 6 threshold policy must be validation_max_f1")
    configured_ids = [str(run.get("id")) for run in config.get("runs", [])]
    if configured_ids[: len(HEADLINE_RUN_IDS)] != list(HEADLINE_RUN_IDS):
        raise ValueError("Sprint 6 headline run list must preserve the frozen S6R0-S6R7 order")


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
        if run_id in headline:
            selected.append(headline[run_id])
        elif run_id in optional:
            if not include_optional_runs:
                raise ValueError(
                    f"Optional Sprint 6 run '{run_id}' requires --include-optional-runs. "
                    "Optional runs are out of scope for the default Slice 2 runner."
                )
            selected.append(optional[run_id])
        else:
            allowed = sorted([*headline, *optional])
            raise ValueError(f"Unknown Sprint 6 run ID '{run_id}'. Allowed IDs: {allowed}")
    return selected


def _config_for_run(
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    max_epochs: int | None,
) -> dict[str, Any]:
    run_config = deepcopy(dict(config))
    training = dict(run_config.get("training", {}))
    training["loss"] = str(run_spec["loss"])
    training["loss_params"] = dict(run_spec.get("params", {}))
    if "sampling" in run_spec:
        training["sampling"] = dict(run_spec["sampling"])
    else:
        training.pop("sampling", None)
    if max_epochs is not None:
        training["max_epochs"] = int(max_epochs)
        training["min_epochs"] = 1
        training["patience"] = min(2, int(max_epochs))
        training["device"] = "cpu"
        training["use_compile"] = False
        training["use_amp"] = False
    run_config["training"] = training
    run_config["sprint6_run"] = {
        "id": run_spec["id"],
        "role": run_spec.get("role"),
        "loss": run_spec["loss"],
        "loss_params": dict(run_spec.get("params", {})),
        "sampling": dict(run_spec["sampling"]) if "sampling" in run_spec else None,
    }
    return run_config


def _validate_sprint6_graph_artifacts(manifest: Mapping[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_A:
        raise ValueError("Sprint 6 requires Graph A materialized artifacts")
    feature_tables = manifest.get("feature_tables", {})
    if "S5F2_energy" not in feature_tables and "s5f2_energy" not in feature_tables:
        raise ValueError("Sprint 6 requires the Sprint 5 S5F2_energy feature table")
    metadata = manifest.get("metadata", {})
    sprint5_metadata = metadata.get("sprint5_feature_ablation", {})
    if sprint5_metadata and sprint5_metadata.get("topology") != "Graph A unchanged":
        raise ValueError("Sprint 6 requires Sprint 5 fixed-topology Graph A provenance")


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
        "loss_params": json.dumps(run_spec.get("params", {}), sort_keys=True),
        "sampling": json.dumps(run_spec.get("sampling"), sort_keys=True),
        "role": run_spec.get("role"),
        "prior_sprint4_graph_a_test_auprc": config.get("prior_context", {})
        .get("sprint4_graph_a", {})
        .get("test_auprc"),
        "prior_sprint4_graph_a_test_mcc": config.get("prior_context", {})
        .get("sprint4_graph_a", {})
        .get("test_mcc"),
        "prior_sprint5_s5f2_test_auprc": config.get("prior_context", {})
        .get("sprint5_graph_a_s5f2_energy", {})
        .get("test_auprc"),
        "prior_sprint5_s5f2_test_mcc": config.get("prior_context", {})
        .get("sprint5_graph_a_s5f2_energy", {})
        .get("test_mcc"),
        "prior_test_positive_prevalence": config.get("prior_context", {}).get("test_positive_prevalence"),
    }
    for offset, (column, value) in enumerate(result_payload.items()):
        results.insert(offset, column, value)
    predictions.insert(0, "run_id", run_id)
    predictions.insert(1, "predeclared_run_id", run_spec["id"])
    predictions.insert(2, "loss", run_spec["loss"])
    history.insert(0, "run_id", run_id)
    history.insert(1, "predeclared_run_id", run_spec["id"])
    history.insert(2, "loss", run_spec["loss"])


def _write_sprint6_report(
    results: pd.DataFrame,
    diagnostic_tables: list[Path],
    figure_paths: list[Path],
    path: Path,
    *,
    batch_id: str,
) -> Path:
    rows = results.sort_values("run_order").copy()
    first = rows.iloc[0]
    summary_columns = [
        "predeclared_run_id",
        "loss",
        "loss_params",
        "sampling",
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
    baseline_row = rows.loc[rows["predeclared_run_id"] == "S6R0_wbce"]
    baseline_auprc = float(baseline_row.iloc[0]["test_auprc"]) if not baseline_row.empty else float(first["test_auprc"])
    summary["delta_auprc_vs_S6R0"] = summary["test_auprc"].astype(float) - baseline_auprc
    report = f"""# Sprint 6 Loss Comparison Report

Run batch: `{batch_id}`

## Contract

- Label scheme: `{first['label_scheme']}`.
- Split ID: `{first['split_id']}`.
- Graph schema: `{first['graph_schema']}`.
- Feature set: `{first['feature_set']}`.
- Training regime: measured-only headline; no `measured=0` putative rows.
- Checkpoint selection: validation AUPRC only.
- Threshold selection: validation max-F1 only.
- Primary metric: AUPRC. Threshold metrics are secondary interpretation outputs.
- Test positive prevalence: `{float(first['prior_test_positive_prevalence']):.6f}`.
- Required reference: `{first['baseline_reference']}` test AUPRC `{float(first['baseline_test_auprc']):.6f}`.
- Sprint 5 Graph A `S5F2_energy` reference test AUPRC `{float(first['prior_sprint5_s5f2_test_auprc']):.6f}`.

## Result Summary

{_markdown_table(summary)}

## Interpretation Boundaries

Sprint 6 varies only the loss function or measured-only training-time sampling.
AUPRC remains the primary comparison. Specificity, TNR, MCC, and macro F1 are
reported to diagnose threshold behavior and negative-class recognition, but
improvements in those secondary metrics must not be described as AUPRC gains.

If threshold collapse persists across losses, the interpretation must include
the architecture caveat: in the current `GraphAEdgeGCN`, candidate-edge features
such as `S5F2_energy` are concatenated at the edge classifier and do not enter
GCN message passing. Collapse therefore cannot be attributed to the loss alone.

## Artifact Index

Diagnostic tables:
{chr(10).join(f'- `{_relative(path)}`' for path in diagnostic_tables)}

Figures:
{chr(10).join(f'- `{_relative(path)}`' for path in figure_paths)}
"""
    path.write_text(report, encoding="utf-8")
    return path


def _write_graph_provenance(
    path: Path,
    *,
    graph_dir: Path,
    manifest: Mapping[str, Any],
    run_specs: list[Mapping[str, Any]],
) -> None:
    graph_a_dir = graph_dir / GRAPH_A
    payload = {
        "provenance_type": "sprint6_graph_a_s5f2_energy_artifacts_sha256",
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
                "loss": run["loss"],
                "params": run.get("params", {}),
                "sampling": run.get("sampling"),
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
    report_path: Path,
    diagnostics: list[Path],
    figures: list[Path],
    provenance_path: Path,
    optional_runs_available: list[str],
) -> None:
    payload = {
        "manifest_type": "sprint6_loss_comparison_run_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "batch_id": batch_id,
        "config_path": _relative(config_path),
        "headline_run_ids": list(HEADLINE_RUN_IDS),
        "optional_runs_available": optional_runs_available,
        "optional_runs_executed": [run["predeclared_id"] for run in runs if run["predeclared_id"] not in HEADLINE_RUN_IDS],
        "runs": runs,
        "results_path": _relative(results_path),
        "predictions_path": _relative(predictions_path),
        "training_history_path": _relative(history_path),
        "report_path": _relative(report_path),
        "graph_artifact_provenance_path": _relative(provenance_path),
        "diagnostic_tables": [_relative(path) for path in diagnostics],
        "figures": [_relative(path) for path in figures],
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
    return f"sprint6_loss_comparison_seed{int(config.get('seed', 42))}_{ts}"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True), encoding="utf-8")


def _validate_consolidated_run_ids(results: pd.DataFrame) -> None:
    if "run_id" not in results.columns:
        raise ValueError("Consolidated Sprint 6 results must include run_id")
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Consolidated Sprint 6 results require unique run_id values")


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
