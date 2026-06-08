from __future__ import annotations

import argparse
import datetime
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

from crispr_gnn.graph.graph_schemas import GRAPH_C  # noqa: E402
from crispr_gnn.graph.pyg_dataset import LABEL_SCHEME, SPLIT_ID, VISIBILITY_POLICY, Sprint3HeteroDataLoader  # noqa: E402
from crispr_gnn.models.gat import graph_c_attention_edge_tensors  # noqa: E402
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE, graph_c_edge_feature_attrs  # noqa: E402
from crispr_gnn.training.gcn import collect_graph_attention_summary, gcn_run_config_from_mapping, train_graph_c_gcn  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


REFERENCE_RUN_IDS = (
    "S7D_REF_XGB_F4",
    "S7D_REF_GRAPH_A_GCN",
    "S7D_REF_GRAPH_A_GATV2",
    "S7D_REF_GRAPH_C_GCN",
    "S7D_REF_FULL_GRAPH_C_GATV2",
)
HEADLINE_RUN_IDS = (
    "S7D_R1_no_context_edges",
    "S7D_R2_edge_blind_attention",
    "S7D_R3_mask_target_context_features",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 7D Graph C GATv2 mechanism ablations.")
    parser.add_argument("--config", default="configs/sweeps/sprint7d_graphc_gatv2_mechanism_ablation.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for smoke/debug runs.")
    parser.add_argument("--run", action="append", default=None, help="Run only this predeclared run ID. May repeat.")
    parser.add_argument("--run-id", default=None, help="Batch ID recorded in manifest and per-run IDs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_sprint7d_graphc_gatv2_mechanism_ablation(
        config_path=ROOT / args.config,
        batch_id=args.run_id,
        max_epochs=args.max_epochs,
        selected_run_ids=args.run,
    )
    return 0


def run_sprint7d_graphc_gatv2_mechanism_ablation(
    *,
    config_path: str | Path,
    batch_id: str | None = None,
    max_epochs: int | None = None,
    selected_run_ids: list[str] | None = None,
) -> Path:
    config = load_yaml(config_path)
    _validate_base_config(config)
    batch = batch_id or _batch_id(config)
    run_specs = _selected_run_specs(config, selected_run_ids)
    include_references = selected_run_ids is None

    graph_c_dir = ROOT / str(config["data"]["graph_c_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_c_dir).load(GRAPH_C)
    _validate_graph_c_artifacts(materialized.manifest)

    output_dir = ROOT / str(config.get("outputs", {}).get("output_dir", "outputs/sprint7d"))
    diagnostics_dir = output_dir / "diagnostics"
    figures_dir = output_dir / "figures"
    runs_dir = output_dir / "runs"
    for directory in [output_dir, diagnostics_dir, figures_dir, runs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    provenance_path = output_dir / "graph_artifact_provenance.json"
    _write_graph_provenance(
        provenance_path,
        graph_c_dir=graph_c_dir,
        graph_c_manifest=materialized.manifest,
        run_specs=run_specs,
    )

    all_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    all_attention: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    manifest_runs: list[dict[str, object]] = []

    if include_references:
        references = _reference_rows(config, batch_id=batch)
        all_results.append(pd.DataFrame(references))
        for reference in references:
            manifest_runs.append(
                {
                    "run_id": reference["run_id"],
                    "predeclared_id": reference["predeclared_run_id"],
                    "role": reference.get("role"),
                    "source": reference.get("source"),
                    "checkpoint_path": None,
                }
            )

    start_order = len(REFERENCE_RUN_IDS) if include_references else 0
    for offset, run_spec in enumerate(run_specs, start=1):
        run_config_mapping = _config_for_run(config, run_spec, max_epochs=max_epochs)
        run_config = gcn_run_config_from_mapping(run_config_mapping)
        run_id = f"{batch}_{run_spec['id']}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(run_dir / "resolved_config.yaml", run_config_mapping)
        (run_dir / "runtime.json").write_text(
            json.dumps(_runtime_info(str(run_config.device)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint_path = run_dir / "model.pt"
        results, predictions, history = train_graph_c_gcn(materialized, run_config, checkpoint_path=checkpoint_path)
        order = start_order + offset
        _insert_run_metadata(results, predictions, history, run_id=run_id, run_spec=run_spec, order=order)

        attention = collect_graph_attention_summary(
            materialized,
            run_config,
            checkpoint_path=checkpoint_path,
            split="test",
        )
        attention.insert(0, "run_id", run_id)
        attention.insert(1, "predeclared_run_id", run_spec["id"])
        attention.insert(2, "graph_schema", run_config.graph_schema)
        audit = _component_audit(materialized, run_config, run_id=run_id, run_spec=run_spec)

        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)
        attention.to_csv(run_dir / "attention_summary.csv", index=False)
        audit.to_csv(run_dir / "component_audit.csv", index=False)
        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        all_attention.append(attention)
        all_audits.append(audit)
        manifest_runs.append(
            {
                "run_id": run_id,
                "predeclared_id": run_spec["id"],
                "graph_schema": GRAPH_C,
                "architecture": run_spec["architecture"],
                "role": run_spec.get("role"),
                "ablation": dict(run_spec.get("ablation", {})),
                "metrics_path": _relative(run_dir / "metrics.csv"),
                "training_history_path": _relative(run_dir / "training_history.csv"),
                "attention_summary_path": _relative(run_dir / "attention_summary.csv"),
                "component_audit_path": _relative(run_dir / "component_audit.csv"),
                "checkpoint_path": _relative(checkpoint_path),
                "resolved_config_path": _relative(run_dir / "resolved_config.yaml"),
            }
        )
        row = results.iloc[0]
        print(
            f"{run_spec['id']}: test_auprc={float(row['test_auprc']):.6f}, "
            f"test_mcc={float(row['test_mcc']):.6f}, tn={int(row['test_tn'])}, fp={int(row['test_fp'])}"
        )

    result_table = pd.concat(all_results, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    history_table = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    attention_table = pd.concat(all_attention, ignore_index=True) if all_attention else pd.DataFrame()
    audit_table = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()
    _validate_consolidated_run_ids(result_table)

    results_path = output_dir / "graphc_gatv2_mechanism_ablation.csv"
    predictions_path = diagnostics_dir / "graphc_gatv2_mechanism_predictions.csv"
    history_path = diagnostics_dir / "graphc_gatv2_mechanism_training_history.csv"
    attention_path = diagnostics_dir / "graphc_gatv2_mechanism_attention_summary.csv"
    audit_path = diagnostics_dir / "graphc_gatv2_component_ablation_audit.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)
    attention_table.to_csv(attention_path, index=False)
    audit_table.to_csv(audit_path, index=False)

    diagnostic_tables = _write_diagnostics(result_table, prediction_table, attention_table, audit_table, diagnostics_dir)
    figure_paths = _write_figures(result_table, prediction_table, history_table, attention_table, figures_dir)
    report_path = _write_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        output_dir / "graphc_gatv2_mechanism_ablation_report.md",
        batch_id=batch,
    )
    manifest_path = output_dir / "graphc_gatv2_mechanism_ablation_run_manifest.json"
    _write_run_manifest(
        manifest_path,
        config_path=Path(config_path),
        batch_id=batch,
        runs=manifest_runs,
        results_path=results_path,
        predictions_path=predictions_path,
        history_path=history_path,
        attention_path=attention_path,
        audit_path=audit_path,
        report_path=report_path,
        diagnostics=diagnostic_tables,
        figures=figure_paths,
        provenance_path=provenance_path,
    )

    print(f"Run batch: {batch}")
    print(f"Output directory: {_relative(output_dir)}")
    print(f"Results: {_relative(results_path)}")
    print(f"Report: {_relative(report_path)}")
    print(
        result_table[
            [
                "predeclared_run_id",
                "graph_schema",
                "architecture",
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


def _validate_base_config(config: Mapping[str, Any]) -> None:
    if config.get("sprint") != "sprint7d":
        raise ValueError("Sprint 7D runner requires sprint: sprint7d")
    if config.get("task") != "sprint7d_graphc_gatv2_mechanism_ablation":
        raise ValueError("Sprint 7D runner requires task: sprint7d_graphc_gatv2_mechanism_ablation")
    data = config.get("data", {})
    graph = config.get("graph", {})
    evaluation = config.get("evaluation", {})
    if data.get("label_scheme") != LABEL_SCHEME or data.get("split_id") != SPLIT_ID:
        raise ValueError("Sprint 7D must preserve Scheme A and sprint2_main_seed42")
    if graph.get("visibility_policy") != VISIBILITY_POLICY:
        raise ValueError("Sprint 7D must preserve strict-inductive visibility")
    if evaluation.get("protocol") != "headline_guide_level" or evaluation.get("no_test_tuning") is not True:
        raise ValueError("Sprint 7D must preserve headline guide-level no-test-tuning evaluation")
    features = config.get("features", {})
    if features.get("feature_set") != "S5F2_energy" or features.get("edge_feature_sets") != ["s5f2_energy"]:
        raise ValueError("Sprint 7D is frozen to S5F2_energy")
    training = config.get("training", {})
    if str(training.get("loss", "")).lower() != "weighted_bce":
        raise ValueError("Sprint 7D must keep weighted_bce")
    if training.get("loss_params", {}).get("pos_weight") != "auto":
        raise ValueError("Sprint 7D weighted BCE must keep pos_weight: auto")
    configured_ids = [str(run.get("id")) for run in config.get("runs", [])]
    if configured_ids != list(HEADLINE_RUN_IDS):
        raise ValueError("Sprint 7D headline run list/order drift")
    for run in config.get("runs", []):
        if run.get("architecture") != "gatv2":
            raise ValueError("Sprint 7D trains Graph C GATv2 ablations only")
        flags = dict(run.get("ablation", {}))
        enabled = [key for key, value in flags.items() if bool(value)]
        if len(enabled) != 1:
            raise ValueError(f"Sprint 7D run {run.get('id')} must enable exactly one ablation flag")


def _selected_run_specs(config: Mapping[str, Any], selected_run_ids: list[str] | None) -> list[dict[str, Any]]:
    headline = {str(run["id"]): dict(run) for run in config.get("runs", [])}
    if selected_run_ids is None:
        return [headline[run_id] for run_id in HEADLINE_RUN_IDS]
    selected = []
    for run_id in selected_run_ids:
        if run_id in REFERENCE_RUN_IDS:
            continue
        if run_id not in headline:
            raise ValueError(f"Unknown Sprint 7D run ID '{run_id}'. Allowed IDs: {sorted([*REFERENCE_RUN_IDS, *headline])}")
        selected.append(headline[run_id])
    return selected


def _config_for_run(
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    max_epochs: int | None,
) -> dict[str, Any]:
    run_config = deepcopy(dict(config))
    run_config["data"] = dict(run_config.get("data", {}))
    run_config["data"]["graph_artifact_dir"] = run_config["data"]["graph_c_artifact_dir"]
    run_config["graph"] = {
        "schema": GRAPH_C,
        "materialization": "heterodata",
        "visibility_policy": VISIBILITY_POLICY,
    }
    run_config["model"] = dict(run_config.get("model", {}))
    run_config["model"]["name"] = str(run_spec["model_name"])
    run_config["model"]["architecture"] = "gatv2"
    run_config["model"]["target_node_representation"] = "target_observation_context_encoder"
    ablation = dict(run_spec.get("ablation", {}))
    attention = dict(run_config["model"].get("attention", {}))
    attention["drop_context_similarity_edges"] = bool(ablation.get("drop_context_similarity_edges", False))
    attention["edge_blind_candidate_attention"] = bool(ablation.get("edge_blind_candidate_attention", False))
    run_config["model"]["attention"] = attention
    run_config["model"]["mask_target_observation_features"] = bool(
        ablation.get("mask_target_observation_features", False)
    )
    run_config["features"] = {
        "edge_feature_sets": ["s5f2_energy"],
        "feature_set": "S5F2_energy",
    }
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
    run_config["sprint7d_run"] = {
        "id": run_spec["id"],
        "role": run_spec.get("role"),
        "graph_schema": GRAPH_C,
        "architecture": "gatv2",
        "feature_set": "S5F2_energy",
        "loss": "weighted_bce",
        "loss_params": {"pos_weight": "auto"},
        "ablation": ablation,
    }
    return run_config


def _validate_graph_c_artifacts(manifest: Mapping[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_C:
        raise ValueError("Sprint 7D Graph C artifacts have wrong schema")
    feature_tables = manifest.get("feature_tables", {})
    if int(feature_tables.get("S5F2_energy", 0)) != 268:
        raise ValueError("Sprint 7D Graph C requires Sprint 5B S5F2_energy feature table")
    if "target_observation_features" not in feature_tables:
        raise ValueError("Sprint 7D Graph C requires target_observation_features")


def _component_audit(materialized: Any, config: Any, *, run_id: str, run_spec: Mapping[str, Any]) -> pd.DataFrame:
    attrs = graph_c_edge_feature_attrs(config.edge_feature_sets)
    rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        view = materialized.view(split)
        candidate_store = view[GRAPH_C_EDGE_TYPE]
        context_store = view[GRAPH_C_CONTEXT_EDGE_TYPE]
        candidate_attr = candidate_store[attrs[0]]
        attention_edge_index, attention_edge_attr = graph_c_attention_edge_tensors(
            view,
            edge_feature_attrs=attrs,
            include_context_edges=not config.drop_context_similarity_edges,
            zero_candidate_attention_attr=config.edge_blind_candidate_attention,
        )
        candidate_edges = int(candidate_store.edge_index.shape[1])
        context_edges = int(context_store.edge_index.shape[1])
        candidate_attention_rows = candidate_edges * 2
        rows.append(
            {
                "run_id": run_id,
                "predeclared_run_id": run_spec["id"],
                "split": split,
                "graph_schema": GRAPH_C,
                "drop_context_similarity_edges": bool(config.drop_context_similarity_edges),
                "edge_blind_candidate_attention": bool(config.edge_blind_candidate_attention),
                "mask_target_observation_features": bool(config.mask_target_observation_features),
                "candidate_edges": candidate_edges,
                "context_edges_available": context_edges,
                "context_edges_used": 0 if config.drop_context_similarity_edges else context_edges,
                "attention_edges": int(attention_edge_index.shape[1]),
                "attention_edge_attr_dim": int(attention_edge_attr.shape[1]),
                "candidate_attention_attr_abs_sum": float(attention_edge_attr[:candidate_attention_rows].abs().sum()),
                "context_attention_attr_abs_sum": float(attention_edge_attr[candidate_attention_rows:].abs().sum()),
                "classifier_candidate_edge_attr_dim": int(candidate_attr.shape[1]),
                "classifier_candidate_edge_attr_abs_sum": float(candidate_attr.abs().sum()),
                "target_observation_feature_dim": int(view["target_observation"].x.shape[1]),
                "target_feature_abs_sum_before_mask": float(view["target_observation"].x.abs().sum()),
                "target_feature_abs_sum_after_mask": (
                    0.0
                    if config.mask_target_observation_features
                    else float(view["target_observation"].x.abs().sum())
                ),
            }
        )
    return pd.DataFrame(rows)


def _reference_rows(config: Mapping[str, Any], *, batch_id: str) -> list[dict[str, object]]:
    rows = [
        _reference_row(
            "S7D_REF_XGB_F4",
            config.get("prior_context", {}).get("xgboost_f4", {}),
            batch_id=batch_id,
            source_label="Sprint 2 XGBoost F4 baseline",
            graph_schema="tabular_reference",
            architecture="xgboost",
            model_name="xgboost_unweighted_f4_reference",
            feature_set="F4",
            loss="unweighted",
        ),
        _reference_row(
            "S7D_REF_GRAPH_A_GCN",
            _reference_source(config, "graph_a_gcn_s5f2", "sprint7_results_csv", "S7R0_gcn_reference"),
            batch_id=batch_id,
            source_label="Sprint 6/7 Graph A GCN S5F2 weighted-BCE carry-forward",
            graph_schema="graph_a_minimal_physical_target",
            architecture="gcn",
            model_name="gcn_graph_a_sprint6_reference",
        ),
        _reference_row(
            "S7D_REF_GRAPH_A_GATV2",
            _reference_source(config, "graph_a_gatv2_s5f2", "sprint7_results_csv", "S7R2_gatv2_edge_aware"),
            batch_id=batch_id,
            source_label="Sprint 7 Graph A GATv2 carry-forward",
            graph_schema="graph_a_minimal_physical_target",
            architecture="gatv2",
            model_name="gatv2_graph_a_sprint7_reference",
        ),
        _reference_row(
            "S7D_REF_GRAPH_C_GCN",
            _reference_source(config, "graph_c_gcn_s5f2", "sprint5b_graph_c_results_csv", None),
            batch_id=batch_id,
            source_label="Sprint 5B Graph C GCN S5F2 carry-forward",
            graph_schema=GRAPH_C,
            architecture="gcn",
            model_name="gcn_graph_c_sprint5b_reference",
        ),
        _reference_row(
            "S7D_REF_FULL_GRAPH_C_GATV2",
            _reference_source(config, "full_graph_c_gatv2_s5f2", "sprint7b_results_csv", "S7B_R3_graph_c_gatv2_s5f2"),
            batch_id=batch_id,
            source_label="Sprint 7B full Graph C GATv2 carry-forward",
            graph_schema=GRAPH_C,
            architecture="gatv2",
            model_name="gatv2_graph_c_sprint7b_full_reference",
        ),
    ]
    for order, row in enumerate(rows):
        row["run_order"] = order
    return rows


def _reference_source(
    config: Mapping[str, Any],
    prior_key: str,
    source_csv_key: str,
    predeclared_id: str | None,
) -> dict[str, object]:
    prior = dict(config.get("prior_context", {}).get(prior_key, {}))
    disk = _load_reference_from_disk(config, source_csv_key, predeclared_id)
    return {**prior, **disk}


def _load_reference_from_disk(config: Mapping[str, Any], source_csv_key: str, predeclared_id: str | None) -> dict[str, object]:
    path = ROOT / str(config.get("references", {}).get(source_csv_key, ""))
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if predeclared_id is not None and "predeclared_run_id" in table.columns:
        rows = table.loc[table["predeclared_run_id"].astype(str) == predeclared_id]
        if not rows.empty:
            return rows.iloc[0].to_dict()
    if len(table) == 1:
        return table.iloc[0].to_dict()
    return {}


def _reference_row(
    predeclared_id: str,
    source: Mapping[str, Any],
    *,
    batch_id: str,
    source_label: str,
    graph_schema: str,
    architecture: str,
    model_name: str,
    feature_set: str = "S5F2_energy",
    loss: str = "weighted_bce",
) -> dict[str, object]:
    return {
        "run_id": f"{batch_id}_{predeclared_id}",
        "predeclared_run_id": predeclared_id,
        "role": "carry_forward_reference_no_retrain",
        "source": source.get("source", source_label),
        "sprint": "sprint7d",
        "label_scheme": LABEL_SCHEME,
        "split_id": SPLIT_ID,
        "seed": 42,
        "training_regime": "measured_only",
        "model_name": model_name,
        "architecture": architecture,
        "feature_set": feature_set,
        "graph_schema": graph_schema,
        "visibility_policy": VISIBILITY_POLICY,
        "target_node_representation": _reference_target_representation(graph_schema, architecture),
        "loss": loss,
        "checkpoint_policy": "validation_auprc" if architecture != "xgboost" else None,
        "checkpoint_selection_split": "validation" if architecture != "xgboost" else None,
        "threshold_policy": "validation_max_f1" if architecture != "xgboost" else None,
        "threshold_selection_split": "validation" if architecture != "xgboost" else None,
        "edge_feature_sets": "s5f2_energy" if feature_set == "S5F2_energy" else None,
        "edge_feature_columns": int(source.get("edge_feature_columns", 268 if feature_set == "S5F2_energy" else 135)),
        "edge_aware_attention": source.get("edge_aware_attention") if architecture == "gatv2" else None,
        "attention_heads": source.get("attention_heads") if architecture == "gatv2" else None,
        "attention_concat": source.get("attention_concat") if architecture == "gatv2" else None,
        "attention_dropout": source.get("attention_dropout") if architecture == "gatv2" else None,
        "self_loop_edge_fill": source.get("self_loop_edge_fill") if architecture == "gatv2" else None,
        "gatv2_share_weights": source.get("gatv2_share_weights") if architecture == "gatv2" else None,
        "drop_context_similarity_edges": None,
        "edge_blind_candidate_attention": None,
        "mask_target_observation_features": None,
        "test_positive_rate": float(source.get("test_positive_rate", 0.900705)),
        "test_auprc": float(source.get("test_auprc")),
        "test_auroc": float(source.get("test_auroc")),
        "test_macro_f1": _optional_float(source.get("test_macro_f1")),
        "test_mcc": float(source.get("test_mcc")),
        "test_specificity": _optional_float(source.get("test_specificity")),
        "test_sensitivity": _optional_float(source.get("test_sensitivity")),
        "test_tn": int(source.get("test_tn")),
        "test_fp": int(source.get("test_fp")),
        "test_fn": int(source.get("test_fn")),
        "test_tp": int(source.get("test_tp")),
        "notes": "Carry-forward matched-contract reference; not retrained in Sprint 7D runner.",
    }


def _insert_run_metadata(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    *,
    run_id: str,
    run_spec: Mapping[str, Any],
    order: int,
) -> None:
    ablation = dict(run_spec.get("ablation", {}))
    payload = {
        "run_id": run_id,
        "run_order": order,
        "predeclared_run_id": run_spec["id"],
        "role": run_spec.get("role"),
        "controlled_variable": "graph_c_gatv2_component_ablation",
        "drop_context_similarity_edges": bool(ablation.get("drop_context_similarity_edges", False)),
        "edge_blind_candidate_attention": bool(ablation.get("edge_blind_candidate_attention", False)),
        "mask_target_observation_features": bool(ablation.get("mask_target_observation_features", False)),
    }
    for offset, (column, value) in enumerate(payload.items()):
        _insert_or_assign(results, offset, column, value)
    _insert_or_assign(predictions, 0, "run_id", run_id)
    _insert_or_assign(predictions, 1, "predeclared_run_id", run_spec["id"])
    _insert_or_assign(predictions, 2, "role", run_spec.get("role"))
    _insert_or_assign(history, 0, "run_id", run_id)
    _insert_or_assign(history, 1, "predeclared_run_id", run_spec["id"])
    _insert_or_assign(history, 2, "role", run_spec.get("role"))


def _insert_or_assign(df: pd.DataFrame, loc: int, column: str, value: object) -> None:
    if column in df.columns:
        df[column] = value
    else:
        df.insert(loc, column, value)


def _write_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    attention: pd.DataFrame,
    audit: pd.DataFrame,
    diagnostics_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    threshold_columns = [
        "run_id",
        "predeclared_run_id",
        "graph_schema",
        "architecture",
        "drop_context_similarity_edges",
        "edge_blind_candidate_attention",
        "mask_target_observation_features",
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
    threshold_path = diagnostics_dir / "graphc_gatv2_mechanism_threshold_metrics.csv"
    results[[column for column in threshold_columns if column in results.columns]].to_csv(threshold_path, index=False)
    paths.append(threshold_path)

    deltas = results.copy()
    full_auprc = _reference_metric(deltas, "S7D_REF_FULL_GRAPH_C_GATV2", "test_auprc")
    full_mcc = _reference_metric(deltas, "S7D_REF_FULL_GRAPH_C_GATV2", "test_mcc")
    graph_c_gcn_auprc = _reference_metric(deltas, "S7D_REF_GRAPH_C_GCN", "test_auprc")
    xgb_auprc = _reference_metric(deltas, "S7D_REF_XGB_F4", "test_auprc")
    deltas["delta_auprc_vs_full_graph_c_gatv2"] = deltas["test_auprc"].astype(float) - full_auprc
    deltas["delta_mcc_vs_full_graph_c_gatv2"] = deltas["test_mcc"].astype(float) - full_mcc
    deltas["delta_auprc_vs_graph_c_gcn"] = deltas["test_auprc"].astype(float) - graph_c_gcn_auprc
    deltas["delta_auprc_vs_xgboost_f4"] = deltas["test_auprc"].astype(float) - xgb_auprc
    delta_path = diagnostics_dir / "graphc_gatv2_mechanism_deltas.csv"
    deltas[
        [
            "run_id",
            "predeclared_run_id",
            "test_auprc",
            "delta_auprc_vs_full_graph_c_gatv2",
            "delta_mcc_vs_full_graph_c_gatv2",
            "delta_auprc_vs_graph_c_gcn",
            "delta_auprc_vs_xgboost_f4",
            "test_mcc",
            "test_specificity",
            "test_tn",
            "test_fp",
        ]
    ].to_csv(delta_path, index=False)
    paths.append(delta_path)

    audit_path = diagnostics_dir / "graphc_gatv2_component_ablation_audit.csv"
    audit.to_csv(audit_path, index=False)
    paths.append(audit_path)

    attention_contract_path = diagnostics_dir / "graphc_gatv2_mechanism_attention_contract_summary.csv"
    if attention.empty:
        pd.DataFrame(
            [{"note": "No attention rows were produced.", "interpretation": "attention is interpretation-only"}]
        ).to_csv(attention_contract_path, index=False)
    else:
        attention.to_csv(attention_contract_path, index=False)
    paths.append(attention_contract_path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        per_guide_path = diagnostics_dir / "graphc_gatv2_mechanism_per_guide_score_summary.csv"
        _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "graph_schema", "architecture", "grna_target_id"],
        ).to_csv(per_guide_path, index=False)
        paths.append(per_guide_path)

        deciles_path = diagnostics_dir / "graphc_gatv2_mechanism_score_deciles.csv"
        _score_deciles(test_predictions).to_csv(deciles_path, index=False)
        paths.append(deciles_path)
    return paths


def _write_figures(
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

    path = figures_dir / "graphc_gatv2_mechanism_auprc_comparison.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, rows["test_auprc"].astype(float))
    ax.axhline(0.992522, color="black", linestyle="--", linewidth=1, label="XGBoost F4")
    ax.axhline(0.900705, color="gray", linestyle=":", linewidth=1, label="test prevalence")
    ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "graphc_gatv2_mechanism_threshold_metrics.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    rows.set_index("predeclared_run_id")[["test_mcc", "test_specificity", "test_macro_f1"]].astype(float).plot(
        kind="bar",
        ax=ax,
    )
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        path = figures_dir / "graphc_gatv2_mechanism_pr_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for _run_id, group in test_predictions.groupby("run_id", sort=False):
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

        path = figures_dir / "graphc_gatv2_mechanism_roc_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for _run_id, group in test_predictions.groupby("run_id", sort=False):
            fpr, tpr, _thresholds = roc_curve(group["label"], group["score"])
            ax.plot(fpr, tpr, label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "graphc_gatv2_mechanism_score_distributions.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        for _run_id, group in test_predictions.groupby("run_id", sort=False):
            ax.hist(group["score"], bins=20, alpha=0.45, label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Rows")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if not history.empty:
        path = figures_dir / "graphc_gatv2_mechanism_training_curves.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        for _run_id, group in history.groupby("run_id", sort=False):
            ax.plot(group["epoch"], group["val_auprc"], marker="o", label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation AUPRC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if not attention.empty:
        path = figures_dir / "graphc_gatv2_mechanism_attention_by_edge_kind.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot = (
            attention.groupby(["predeclared_run_id", "edge_kind"], dropna=False)["attention_mean"]
            .mean()
            .unstack("edge_kind")
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean attention weight")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_report(
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
        "graph_schema",
        "architecture",
        "drop_context_similarity_edges",
        "edge_blind_candidate_attention",
        "mask_target_observation_features",
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
    report = f"""# Sprint 7D Graph C GATv2 Mechanism Ablation Report

Run batch: `{batch_id}`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Architecture policy: Graph C GATv2 only for newly trained runs; no new losses, samplers, encoders, or graph schemas.
- Controlled question: which existing Graph C GATv2 component is necessary for the Sprint 7B rare-negative operating-point behavior.
- Primary metric: AUPRC. MCC, specificity, TN/FP, and macro F1 are secondary operating-point diagnostics.
- Attention summaries are interpretation-only model artifacts, not biological causal evidence.

## Result Summary

{_markdown_table(summary)}

## Ablation Definitions

- `S7D_R1_no_context_edges`: drops existing `context_similar_to` target-observation topology; direct target-observation features and S5F2 edge-aware attention remain active.
- `S7D_R2_edge_blind_attention`: zeroes candidate `S5F2_energy` only inside GATv2 attention/message passing; the final edge classifier still receives S5F2.
- `S7D_R3_mask_target_context_features`: masks direct target-observation node features; context-derived topology and S5F2 edge-aware attention remain active.

## Interpretation Boundaries

- Graph C is not topology-only; it combines target-observation node semantics, context-derived topology, and candidate-edge features.
- A single-seed ablation can support mechanism hypotheses, not statistical superiority claims.
- No architecture, threshold, loss, sampler, or rerun choice may be selected from test diagnostics.

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
    for run_id, group in predictions.groupby("run_id", sort=False):
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
                    "graph_schema": decile_group["graph_schema"].iloc[0],
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
    graph_c_dir: Path,
    graph_c_manifest: Mapping[str, Any],
    run_specs: list[Mapping[str, Any]],
) -> None:
    payload = {
        "provenance_type": "sprint7d_graphc_gatv2_mechanism_artifacts_sha256",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "graph_c_artifact_dir": _relative(graph_c_dir),
        "graph_c": _graph_manifest_summary(graph_c_manifest),
        "run_specs": [
            {
                "id": run["id"],
                "graph_schema": GRAPH_C,
                "architecture": run["architecture"],
                "role": run.get("role"),
                "ablation": dict(run.get("ablation", {})),
            }
            for run in run_specs
        ],
        "files": {GRAPH_C: _artifact_file_hashes(graph_c_dir / GRAPH_C)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _graph_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {
        "graph_name": manifest.get("graph_name"),
        "split_id": manifest.get("split_id"),
        "label_scheme": manifest.get("label_scheme"),
        "visibility_policy": manifest.get("metadata", {}).get("visibility_policy"),
        "feature_tables": manifest.get("feature_tables", {}),
        "metadata": manifest.get("metadata", {}),
        "preprocessing": manifest.get("preprocessing", {}),
    }


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
    audit_path: Path,
    report_path: Path,
    diagnostics: list[Path],
    figures: list[Path],
    provenance_path: Path,
) -> None:
    payload = {
        "manifest_type": "sprint7d_graphc_gatv2_mechanism_ablation_run_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "batch_id": batch_id,
        "config_path": _relative(config_path),
        "reference_run_ids": list(REFERENCE_RUN_IDS),
        "headline_run_ids": list(HEADLINE_RUN_IDS),
        "runs": runs,
        "results_path": _relative(results_path),
        "predictions_path": _relative(predictions_path),
        "training_history_path": _relative(history_path),
        "attention_summary_path": _relative(attention_path),
        "component_audit_path": _relative(audit_path),
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
        raise FileNotFoundError(f"Graph artifact directory not found: {artifact_dir}")
    rows = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(artifact_dir).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    if not rows:
        raise ValueError(f"No artifact files found under {artifact_dir}")
    return rows


def _sha256(path: Path) -> str:
    import hashlib

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
    return f"sprint7d_graphc_gatv2_mechanism_seed{int(config.get('seed', 42))}_{ts}"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True), encoding="utf-8")


def _validate_consolidated_run_ids(results: pd.DataFrame) -> None:
    if "run_id" not in results.columns:
        raise ValueError("Consolidated Sprint 7D results must include run_id")
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Consolidated Sprint 7D results require unique run_id values")


def _reference_metric(results: pd.DataFrame, predeclared_run_id: str, column: str) -> float:
    rows = results.loc[results["predeclared_run_id"] == predeclared_run_id]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][column])


def _reference_target_representation(graph_schema: str, architecture: str) -> str | None:
    if architecture == "xgboost":
        return None
    if graph_schema == GRAPH_C:
        return "target_observation_context_encoder"
    return "zero_type_feature"


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
