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

from crispr_gnn.features.target_context import (  # noqa: E402
    TARGET_CONTEXT_EXPECTED_COUNTS,
    target_context_family_counts,
    validate_target_context_feature_names,
)
from crispr_gnn.graph.graph_schemas import GRAPH_C  # noqa: E402
from crispr_gnn.graph.pyg_dataset import LABEL_SCHEME, SPLIT_ID, VISIBILITY_POLICY, Sprint3HeteroDataLoader  # noqa: E402
from crispr_gnn.models.gat import graph_c_attention_edge_tensors  # noqa: E402
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE, graph_c_edge_feature_attrs  # noqa: E402
from crispr_gnn.training.gcn import (  # noqa: E402
    collect_context_edge_interaction_summary,
    collect_graph_attention_summary,
    collect_target_context_encoder_summary,
    gcn_run_config_from_mapping,
    train_graph_c_gcn,
)
from crispr_gnn.utils.config import load_yaml  # noqa: E402


REFERENCE_RUN_IDS = (
    "S8A_REF_XGB_F4",
    "S8A_REF_GRAPH_A_GCN",
    "S8A_REF_GRAPH_C_GCN",
    "S8A_REF_FULL_GRAPH_C_GATV2",
    "S8A_REF_NO_CTX_EDGE_GATV2",
    "S8A_REF_S7F_R2",
    "S8A_REF_S7F_R3",
)
HEADLINE_RUN_IDS = (
    "S8A_R0_base_reference",
    "S8A_R1_family_gated_v2",
    "S8A_R2_context_edge_film",
    "S8A_R3_gated_plus_film",
    "S8A_R4_regularized_exp_branch",
)
CANONICAL_RUN_FLAGS = {
    "S8A_R0_base_reference": {
        "target_context_encoder_type": "family_aware_experimental_emphasis",
        "family_gate": False,
        "gate_reduction": 4,
        "experimental_branch_bottleneck": None,
        "experimental_branch_feature_dropout": 0.0,
        "context_edge_interaction": "none",
        "interaction_edge_dim": 64,
    },
    "S8A_R1_family_gated_v2": {
        "target_context_encoder_type": "family_aware_experimental_emphasis",
        "family_gate": True,
        "gate_reduction": 4,
        "experimental_branch_bottleneck": None,
        "experimental_branch_feature_dropout": 0.0,
        "context_edge_interaction": "none",
        "interaction_edge_dim": 64,
    },
    "S8A_R2_context_edge_film": {
        "target_context_encoder_type": "family_aware_experimental_emphasis",
        "family_gate": False,
        "gate_reduction": 4,
        "experimental_branch_bottleneck": None,
        "experimental_branch_feature_dropout": 0.0,
        "context_edge_interaction": "film",
        "interaction_edge_dim": 64,
    },
    "S8A_R3_gated_plus_film": {
        "target_context_encoder_type": "family_aware_experimental_emphasis",
        "family_gate": True,
        "gate_reduction": 4,
        "experimental_branch_bottleneck": None,
        "experimental_branch_feature_dropout": 0.0,
        "context_edge_interaction": "film",
        "interaction_edge_dim": 64,
    },
    "S8A_R4_regularized_exp_branch": {
        "target_context_encoder_type": "family_aware_experimental_emphasis",
        "family_gate": False,
        "gate_reduction": 4,
        "experimental_branch_bottleneck": 4,
        "experimental_branch_feature_dropout": 0.3,
        "context_edge_interaction": "none",
        "interaction_edge_dim": 64,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sprint 8A Graph C GATv2 target-context + context-edge interaction comparison."
    )
    parser.add_argument("--config", default="configs/sweeps/sprint8a_target_context_interaction.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for smoke/debug runs.")
    parser.add_argument("--run", action="append", default=None, help="Run only this predeclared run ID. May repeat.")
    parser.add_argument("--run-id", default=None, help="Batch ID recorded in manifest and per-run IDs.")
    parser.add_argument("--seed", type=int, default=None, help="Override the top-level training seed.")
    parser.add_argument("--output-dir", default=None, help="Override outputs.output_dir for isolated reruns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_sprint8a_target_context_interaction(
        config_path=ROOT / args.config,
        batch_id=args.run_id,
        max_epochs=args.max_epochs,
        selected_run_ids=args.run,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
    )
    return 0


def run_sprint8a_target_context_interaction(
    *,
    config_path: str | Path,
    batch_id: str | None = None,
    max_epochs: int | None = None,
    selected_run_ids: list[str] | None = None,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    if seed_override is not None:
        config = dict(config)
        config["seed"] = int(seed_override)
    _validate_base_config(config)
    batch = batch_id or _batch_id(config)
    run_specs = _selected_run_specs(config, selected_run_ids)
    include_references = selected_run_ids is None

    graph_c_dir = ROOT / str(config["data"]["graph_c_artifact_dir"])
    materialized = Sprint3HeteroDataLoader(graph_c_dir).load(GRAPH_C)
    _validate_graph_c_artifacts(materialized.manifest)

    output_dir = _resolve_output_dir(output_dir_override, config, default="outputs/sprint8a")
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
    all_encoder_summary: list[pd.DataFrame] = []
    all_interaction_summary: list[pd.DataFrame] = []
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
            materialized, run_config, checkpoint_path=checkpoint_path, split="test"
        )
        attention.insert(0, "run_id", run_id)
        attention.insert(1, "predeclared_run_id", run_spec["id"])
        attention.insert(2, "graph_schema", run_config.graph_schema)

        encoder_summary = collect_target_context_encoder_summary(
            materialized, run_config, checkpoint_path=checkpoint_path, split="test"
        )
        encoder_summary.insert(0, "run_id", run_id)
        encoder_summary.insert(1, "predeclared_run_id", run_spec["id"])

        interaction_summary = collect_context_edge_interaction_summary(
            materialized, run_config, checkpoint_path=checkpoint_path, split="test"
        )
        if not interaction_summary.empty:
            interaction_summary.insert(0, "run_id", run_id)
            interaction_summary.insert(1, "predeclared_run_id", run_spec["id"])
            all_interaction_summary.append(interaction_summary)

        audit = _interaction_audit(
            materialized, run_config, run_id=run_id, run_spec=run_spec, encoder_summary=encoder_summary
        )

        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)
        attention.to_csv(run_dir / "attention_summary.csv", index=False)
        encoder_summary.to_csv(run_dir / "target_context_encoder_summary.csv", index=False)
        audit.to_csv(run_dir / "target_context_interaction_audit.csv", index=False)
        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        all_attention.append(attention)
        all_encoder_summary.append(encoder_summary)
        all_audits.append(audit)
        manifest_runs.append(
            {
                "run_id": run_id,
                "predeclared_id": run_spec["id"],
                "graph_schema": GRAPH_C,
                "architecture": run_spec["architecture"],
                "role": run_spec.get("role"),
                **_run_spec_flags(run_spec),
                "metrics_path": _relative(run_dir / "metrics.csv"),
                "training_history_path": _relative(run_dir / "training_history.csv"),
                "attention_summary_path": _relative(run_dir / "attention_summary.csv"),
                "target_context_encoder_summary_path": _relative(run_dir / "target_context_encoder_summary.csv"),
                "target_context_interaction_audit_path": _relative(run_dir / "target_context_interaction_audit.csv"),
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
    encoder_summary_table = pd.concat(all_encoder_summary, ignore_index=True) if all_encoder_summary else pd.DataFrame()
    interaction_table = (
        pd.concat(all_interaction_summary, ignore_index=True) if all_interaction_summary else pd.DataFrame()
    )
    audit_table = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()
    _validate_consolidated_run_ids(result_table)

    results_path = output_dir / "target_context_interaction_comparison.csv"
    predictions_path = diagnostics_dir / "target_context_interaction_predictions.csv"
    history_path = diagnostics_dir / "target_context_interaction_training_history.csv"
    attention_path = diagnostics_dir / "target_context_interaction_attention_summary.csv"
    audit_path = diagnostics_dir / "target_context_interaction_audit.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)
    attention_table.to_csv(attention_path, index=False)
    audit_table.to_csv(audit_path, index=False)

    diagnostic_tables = _write_diagnostics(
        result_table,
        prediction_table,
        encoder_summary_table,
        interaction_table,
        diagnostics_dir,
    )
    diagnostic_tables = [audit_path, *diagnostic_tables]
    figure_paths = _write_figures(
        result_table,
        prediction_table,
        history_table,
        encoder_summary_table,
        figures_dir,
    )
    report_path = _write_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        output_dir / "target_context_interaction_report.md",
        batch_id=batch,
    )
    manifest_path = output_dir / "target_context_interaction_run_manifest.json"
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
    return output_dir


def _validate_base_config(config: Mapping[str, Any]) -> None:
    if config.get("sprint") != "sprint8a":
        raise ValueError("Sprint 8A runner requires sprint: sprint8a")
    if config.get("task") != "sprint8a_target_context_interaction":
        raise ValueError("Sprint 8A runner requires task: sprint8a_target_context_interaction")
    data = config.get("data", {})
    graph = config.get("graph", {})
    evaluation = config.get("evaluation", {})
    if data.get("label_scheme") != LABEL_SCHEME or data.get("split_id") != SPLIT_ID:
        raise ValueError("Sprint 8A must preserve Scheme A and sprint2_main_seed42")
    if graph.get("visibility_policy") != VISIBILITY_POLICY:
        raise ValueError("Sprint 8A must preserve strict-inductive visibility")
    if evaluation.get("protocol") != "headline_guide_level" or evaluation.get("no_test_tuning") is not True:
        raise ValueError("Sprint 8A must preserve headline guide-level no-test-tuning evaluation")
    features = config.get("features", {})
    if features.get("feature_set") != "S5F2_energy" or features.get("edge_feature_sets") != ["s5f2_energy"]:
        raise ValueError("Sprint 8A is frozen to S5F2_energy")
    training = config.get("training", {})
    if str(training.get("loss", "")).lower() != "weighted_bce":
        raise ValueError("Sprint 8A must keep weighted_bce")
    if training.get("loss_params", {}).get("pos_weight") != "auto":
        raise ValueError("Sprint 8A weighted BCE must keep pos_weight: auto")
    configured_ids = [str(run.get("id")) for run in config.get("runs", [])]
    if configured_ids != list(HEADLINE_RUN_IDS):
        raise ValueError("Sprint 8A headline run list/order drift (expected the 5 canonical R0-R4 runs)")
    for run in config.get("runs", []):
        if run.get("architecture") != "gatv2":
            raise ValueError("Sprint 8A trains Graph C GATv2 runs only")
        encoder = run.get("target_context_encoder")
        if not isinstance(encoder, Mapping) or not encoder.get("type"):
            raise ValueError(f"Sprint 8A run {run.get('id')} must declare target_context_encoder.type")
        _validate_canonical_run_flags(run)


def _selected_run_specs(config: Mapping[str, Any], selected_run_ids: list[str] | None) -> list[dict[str, Any]]:
    headline = {str(run["id"]): dict(run) for run in config.get("runs", [])}
    if selected_run_ids is None:
        return [headline[run_id] for run_id in HEADLINE_RUN_IDS]
    selected = []
    for run_id in selected_run_ids:
        if run_id in REFERENCE_RUN_IDS:
            continue
        if run_id not in headline:
            allowed = sorted([*REFERENCE_RUN_IDS, *headline])
            raise ValueError(f"Unknown Sprint 8A run ID '{run_id}'. Allowed IDs: {allowed}")
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
    run_config["model"]["target_context_encoder"] = dict(run_spec["target_context_encoder"])
    run_config["model"]["target_observation_mask_families"] = []
    run_config["model"]["context_edge_interaction"] = str(run_spec.get("context_edge_interaction", "none"))
    run_config["model"]["interaction_edge_dim"] = int(run_spec.get("interaction_edge_dim", 64))
    attention = dict(run_config["model"].get("attention", {}))
    attention["drop_context_similarity_edges"] = True
    attention["edge_blind_candidate_attention"] = False
    attention["edge_aware"] = True
    run_config["model"]["attention"] = attention
    run_config["model"]["mask_target_observation_features"] = False
    run_config["features"] = {"edge_feature_sets": ["s5f2_energy"], "feature_set": "S5F2_energy"}
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
    run_config["sprint8a_run"] = {
        "id": run_spec["id"],
        "role": run_spec.get("role"),
        "graph_schema": GRAPH_C,
        "architecture": "gatv2",
        "feature_set": "S5F2_energy",
        "loss": "weighted_bce",
        "loss_params": {"pos_weight": "auto"},
        **_run_spec_flags(run_spec),
        "drop_context_similarity_edges": True,
    }
    return run_config


def _validate_graph_c_artifacts(manifest: Mapping[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_C:
        raise ValueError("Sprint 8A Graph C artifacts have wrong schema")
    feature_tables = manifest.get("feature_tables", {})
    if int(feature_tables.get("S5F2_energy", 0)) != 268:
        raise ValueError("Sprint 8A Graph C requires Sprint 5B S5F2_energy feature table")
    if int(feature_tables.get("target_observation_features", 0)) != sum(TARGET_CONTEXT_EXPECTED_COUNTS.values()):
        raise ValueError("Sprint 8A Graph C requires 212 target_observation_features")


def _run_spec_flags(run_spec: Mapping[str, Any]) -> dict[str, object]:
    encoder = run_spec.get("target_context_encoder", {}) or {}
    experimental_branch = encoder.get("experimental_branch", {}) or {}
    return {
        "target_context_encoder_type": str(encoder.get("type")),
        "family_gate": bool(encoder.get("family_gate", False)),
        "gate_reduction": int(encoder.get("gate_reduction", 4)),
        "experimental_branch_bottleneck": experimental_branch.get("bottleneck"),
        "experimental_branch_feature_dropout": float(experimental_branch.get("feature_dropout", 0.0)),
        "context_edge_interaction": str(run_spec.get("context_edge_interaction", "none")),
        "interaction_edge_dim": int(run_spec.get("interaction_edge_dim", 64)),
    }


def _validate_canonical_run_flags(run_spec: Mapping[str, Any]) -> None:
    run_id = str(run_spec.get("id"))
    expected = CANONICAL_RUN_FLAGS.get(run_id)
    if expected is None:
        raise ValueError(f"Unexpected Sprint 8A canonical run ID: {run_id}")
    encoder = run_spec.get("target_context_encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError(f"Sprint 8A run {run_id} must declare target_context_encoder")
    experimental_branch = encoder.get("experimental_branch")
    if not isinstance(experimental_branch, Mapping):
        raise ValueError(f"Sprint 8A run {run_id} must explicitly declare target_context_encoder.experimental_branch")
    required_encoder_keys = {"type", "family_gate", "gate_reduction", "experimental_branch"}
    missing_encoder = required_encoder_keys.difference(encoder)
    if missing_encoder:
        raise ValueError(f"Sprint 8A run {run_id} missing encoder keys: {sorted(missing_encoder)}")
    missing_branch = {"bottleneck", "feature_dropout"}.difference(experimental_branch)
    if missing_branch:
        raise ValueError(f"Sprint 8A run {run_id} missing experimental_branch keys: {sorted(missing_branch)}")
    missing_run = {"context_edge_interaction", "interaction_edge_dim"}.difference(run_spec)
    if missing_run:
        raise ValueError(f"Sprint 8A run {run_id} missing run keys: {sorted(missing_run)}")
    observed = _run_spec_flags(run_spec)
    if observed != expected:
        raise ValueError(
            f"Sprint 8A run {run_id} flag drift: expected {expected}, observed {observed}"
        )


def _interaction_audit(
    materialized: Any,
    config: Any,
    *,
    run_id: str,
    run_spec: Mapping[str, Any],
    encoder_summary: pd.DataFrame,
) -> pd.DataFrame:
    attrs = graph_c_edge_feature_attrs(config.edge_feature_sets)
    flags = _run_spec_flags(run_spec)
    rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        view = materialized.view(split)
        feature_names = list(getattr(view["target_observation"], "feature_names", []))
        validate_target_context_feature_names(feature_names)
        family_counts = target_context_family_counts(feature_names)
        target_x = view["target_observation"].x
        edge_store = view[GRAPH_C_EDGE_TYPE]
        context_store = view[GRAPH_C_CONTEXT_EDGE_TYPE]
        candidate_attr = edge_store[attrs[0]]
        attention_edge_index, attention_edge_attr = graph_c_attention_edge_tensors(
            view, edge_feature_attrs=attrs, include_context_edges=False, zero_candidate_attention_attr=False
        )
        candidate_edges = int(edge_store.edge_index.shape[1])
        candidate_attention_rows = candidate_edges * 2
        rows.append(
            {
                "run_id": run_id,
                "predeclared_run_id": run_spec["id"],
                "split": split,
                "graph_schema": GRAPH_C,
                "target_context_encoder_type": flags["target_context_encoder_type"],
                "family_gate": flags["family_gate"],
                "experimental_branch_bottleneck": flags["experimental_branch_bottleneck"],
                "experimental_branch_feature_dropout": flags["experimental_branch_feature_dropout"],
                "context_edge_interaction": flags["context_edge_interaction"],
                "interaction_edge_dim": flags["interaction_edge_dim"],
                "target_observation_feature_dim": int(target_x.shape[1]),
                "target_sequence_columns": int(family_counts["target_sequence_one_hot"]),
                "experimental_epigenetic_columns": int(family_counts["experimental_epigenetic"]),
                "computed_nucleosome_aggregate_columns": int(family_counts["computed_nucleosome_aggregates"]),
                "computed_nucleosome_missingness_columns": int(family_counts["computed_nucleosome_missingness"]),
                "target_observation_abs_sum": float(target_x.abs().sum()),
                "drop_context_similarity_edges": bool(config.drop_context_similarity_edges),
                "context_edges_available": int(context_store.edge_index.shape[1]),
                "context_edges_used": 0,
                "attention_edges": int(attention_edge_index.shape[1]),
                "attention_edge_attr_dim": int(attention_edge_attr.shape[1]),
                "candidate_attention_attr_abs_sum": float(attention_edge_attr[:candidate_attention_rows].abs().sum()),
                "context_attention_attr_abs_sum": float(attention_edge_attr[candidate_attention_rows:].abs().sum()),
                "classifier_candidate_edge_attr_dim": int(candidate_attr.shape[1]),
                "classifier_candidate_edge_attr_abs_sum": float(candidate_attr.abs().sum()),
                "encoder_summary_rows": int(len(encoder_summary)),
            }
        )
    return pd.DataFrame(rows)


def _reference_rows(config: Mapping[str, Any], *, batch_id: str) -> list[dict[str, object]]:
    prior = config.get("prior_context", {})
    specs = [
        ("S8A_REF_XGB_F4", "xgboost_f4", "tabular_reference", "xgboost", "xgboost_unweighted_f4_reference", "F4", "unweighted"),
        ("S8A_REF_GRAPH_A_GCN", "graph_a_gcn_s5f2", "graph_a_minimal_physical_target", "gcn", "gcn_graph_a_reference", "S5F2_energy", "weighted_bce"),
        ("S8A_REF_GRAPH_C_GCN", "graph_c_gcn_s5f2", GRAPH_C, "gcn", "gcn_graph_c_sprint5b_reference", "S5F2_energy", "weighted_bce"),
        ("S8A_REF_FULL_GRAPH_C_GATV2", "full_graph_c_gatv2_s5f2", GRAPH_C, "gatv2", "gatv2_graph_c_sprint7b_full_reference", "S5F2_energy", "weighted_bce"),
        ("S8A_REF_NO_CTX_EDGE_GATV2", "no_context_edge_graph_c_gatv2_s5f2", GRAPH_C, "gatv2", "gatv2_graph_c_sprint7d_no_context_reference", "S5F2_energy", "weighted_bce"),
        ("S8A_REF_S7F_R2", "s7f_r2", GRAPH_C, "gatv2", "gatv2_graph_c_sprint7f_r2_reference", "S5F2_energy", "weighted_bce"),
        ("S8A_REF_S7F_R3", "s7f_r3", GRAPH_C, "gatv2", "gatv2_graph_c_sprint7f_r3_reference", "S5F2_energy", "weighted_bce"),
    ]
    rows = [
        _reference_row(
            predeclared_id,
            dict(prior.get(prior_key, {})),
            batch_id=batch_id,
            graph_schema=graph_schema,
            architecture=architecture,
            model_name=model_name,
            feature_set=feature_set,
            loss=loss,
        )
        for predeclared_id, prior_key, graph_schema, architecture, model_name, feature_set, loss in specs
    ]
    for order, row in enumerate(rows):
        row["run_order"] = order
    return rows


def _reference_row(
    predeclared_id: str,
    source: Mapping[str, Any],
    *,
    batch_id: str,
    graph_schema: str,
    architecture: str,
    model_name: str,
    feature_set: str,
    loss: str,
) -> dict[str, object]:
    return {
        "run_id": f"{batch_id}_{predeclared_id}",
        "predeclared_run_id": predeclared_id,
        "role": "carry_forward_reference_no_retrain",
        "source": source.get("source", predeclared_id),
        "sprint": "sprint8a",
        "label_scheme": LABEL_SCHEME,
        "split_id": SPLIT_ID,
        "seed": 42,
        "training_regime": "measured_only",
        "model_name": model_name,
        "architecture": architecture,
        "feature_set": feature_set,
        "graph_schema": graph_schema,
        "visibility_policy": VISIBILITY_POLICY if architecture != "xgboost" else None,
        "target_node_representation": _reference_target_representation(graph_schema, architecture),
        "loss": loss,
        "checkpoint_policy": "validation_auprc" if architecture != "xgboost" else None,
        "threshold_policy": "validation_max_f1" if architecture != "xgboost" else None,
        "edge_feature_columns": int(source.get("edge_feature_columns", 268 if feature_set == "S5F2_energy" else 135)),
        "target_context_encoder_type": source.get("target_context_encoder_type"),
        "family_gate": None,
        "context_edge_interaction": None,
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
        "parameter_count": _optional_int(source.get("parameter_count")),
        "notes": "Carry-forward matched-contract reference; not retrained in Sprint 8A runner.",
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
    flags = _run_spec_flags(run_spec)
    payload = {
        "run_id": run_id,
        "run_order": order,
        "predeclared_run_id": run_spec["id"],
        "role": run_spec.get("role"),
        "controlled_variable": "target_context_interaction_delta",
        "drop_context_similarity_edges": True,
        "edge_blind_candidate_attention": False,
        **flags,
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
        df.insert(min(loc, df.shape[1]), column, value)


def _write_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    encoder_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    diagnostics_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    threshold_columns = [
        "run_id",
        "predeclared_run_id",
        "graph_schema",
        "architecture",
        "target_context_encoder_type",
        "family_gate",
        "context_edge_interaction",
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
    threshold_path = diagnostics_dir / "target_context_interaction_threshold_metrics.csv"
    results[[column for column in threshold_columns if column in results.columns]].to_csv(threshold_path, index=False)
    paths.append(threshold_path)

    deltas = results.copy()
    base_auprc = _reference_metric(deltas, "S8A_REF_S7F_R3", "test_auprc")
    base_mcc = _reference_metric(deltas, "S8A_REF_S7F_R3", "test_mcc")
    r2_mcc = _reference_metric(deltas, "S8A_REF_S7F_R2", "test_mcc")
    graph_a_auprc = _reference_metric(deltas, "S8A_REF_GRAPH_A_GCN", "test_auprc")
    xgb_auprc = _reference_metric(deltas, "S8A_REF_XGB_F4", "test_auprc")
    deltas["delta_auprc_vs_s7f_r3_base"] = deltas["test_auprc"].astype(float) - base_auprc
    deltas["delta_mcc_vs_s7f_r3_base"] = deltas["test_mcc"].astype(float) - base_mcc
    deltas["delta_mcc_vs_s7f_r2"] = deltas["test_mcc"].astype(float) - r2_mcc
    deltas["delta_auprc_vs_graph_a_gcn"] = deltas["test_auprc"].astype(float) - graph_a_auprc
    deltas["delta_auprc_vs_xgboost_f4"] = deltas["test_auprc"].astype(float) - xgb_auprc
    delta_path = diagnostics_dir / "target_context_interaction_deltas.csv"
    delta_columns = [
        "run_id",
        "predeclared_run_id",
        "target_context_encoder_type",
        "family_gate",
        "context_edge_interaction",
        "test_auprc",
        "delta_auprc_vs_s7f_r3_base",
        "delta_mcc_vs_s7f_r3_base",
        "delta_mcc_vs_s7f_r2",
        "delta_auprc_vs_graph_a_gcn",
        "delta_auprc_vs_xgboost_f4",
        "test_mcc",
        "test_specificity",
        "test_tn",
        "test_fp",
    ]
    deltas[[column for column in delta_columns if column in deltas.columns]].to_csv(delta_path, index=False)
    paths.append(delta_path)

    gate_path = diagnostics_dir / "target_context_interaction_branch_gate_summary.csv"
    gate_columns = [
        "run_id",
        "predeclared_run_id",
        "target_context_encoder_type",
        "target_context_family",
        "branch_dim",
        "activation_l2_mean",
        "family_gate_enabled",
        "family_gate_weight_mean",
    ]
    if encoder_summary.empty:
        pd.DataFrame([{"note": "No target-context encoder summary rows were produced."}]).to_csv(gate_path, index=False)
    else:
        encoder_summary[[column for column in gate_columns if column in encoder_summary.columns]].to_csv(
            gate_path, index=False
        )
    paths.append(gate_path)

    film_path = diagnostics_dir / "target_context_interaction_film_summary.csv"
    if interaction_summary.empty:
        pd.DataFrame(
            [{"note": "No context-edge interaction rows were produced (all runs used context_edge_interaction=none)."}]
        ).to_csv(film_path, index=False)
    else:
        interaction_summary.to_csv(film_path, index=False)
    paths.append(film_path)

    param_path = diagnostics_dir / "target_context_interaction_parameter_counts.csv"
    param_columns = [
        "run_id",
        "predeclared_run_id",
        "target_context_encoder_type",
        "family_gate",
        "context_edge_interaction",
        "interaction_edge_dim",
        "parameter_count",
        "test_auprc",
    ]
    param_rows = results[[column for column in param_columns if column in results.columns]].copy()
    if "parameter_count" in param_rows.columns:
        param_rows = param_rows.loc[param_rows["parameter_count"].notna()]
    param_rows.to_csv(param_path, index=False)
    paths.append(param_path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        per_guide_path = diagnostics_dir / "target_context_interaction_per_guide_score_summary.csv"
        _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "graph_schema", "architecture", "grna_target_id"],
        ).to_csv(per_guide_path, index=False)
        paths.append(per_guide_path)

        deciles_path = diagnostics_dir / "target_context_interaction_score_deciles.csv"
        _score_deciles(test_predictions).to_csv(deciles_path, index=False)
        paths.append(deciles_path)
    return paths


def _write_figures(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    encoder_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    paths: list[Path] = []
    rows = results.sort_values("run_order")
    labels = rows["predeclared_run_id"].astype(str).tolist()

    path = figures_dir / "target_context_interaction_auprc_comparison.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, rows["test_auprc"].astype(float))
    ax.axhline(0.992522, color="black", linestyle="--", linewidth=1, label="XGBoost F4")
    ax.axhline(0.900705, color="gray", linestyle=":", linewidth=1, label="test prevalence")
    ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "target_context_interaction_threshold_metrics.png"
    fig, ax = plt.subplots(figsize=(12, 4))
    rows.set_index("predeclared_run_id")[["test_mcc", "test_specificity", "test_macro_f1"]].astype(float).plot(
        kind="bar", ax=ax
    )
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        path = figures_dir / "target_context_interaction_pr_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for _run_id, group in test_predictions.groupby("run_id", sort=False):
            precision, recall, _ = precision_recall_curve(group["label"], group["score"])
            ax.plot(recall, precision, label=str(group["predeclared_run_id"].iloc[0]))
        ax.axhline(0.900705, color="gray", linestyle="--", linewidth=1, label="prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "target_context_interaction_roc_curves.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        for _run_id, group in test_predictions.groupby("run_id", sort=False):
            fpr, tpr, _ = roc_curve(group["label"], group["score"])
            ax.plot(fpr, tpr, label=str(group["predeclared_run_id"].iloc[0]))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "target_context_interaction_score_distributions.png"
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

    path = figures_dir / "target_context_interaction_training_curves.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    if not history.empty:
        for _run_id, group in history.groupby("run_id", sort=False):
            ax.plot(group["epoch"], group["val_auprc"], marker="o", label=str(group["predeclared_run_id"].iloc[0]))
        ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUPRC")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "target_context_interaction_gate_weights.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    gated = pd.DataFrame()
    if not encoder_summary.empty and "family_gate_weight_mean" in encoder_summary.columns:
        gated = encoder_summary.copy()
        if "family_gate_enabled" in gated.columns:
            gated = gated.loc[gated["family_gate_enabled"].astype(bool)]
    if not gated.empty:
        pivot = (
            gated.groupby(["predeclared_run_id", "target_context_family"], dropna=False)["family_gate_weight_mean"]
            .mean()
            .unstack("target_context_family")
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean learned family gate weight")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.set_title("No gated runs (family_gate disabled in all runs)")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "target_context_interaction_parameter_counts.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    param_rows = rows.copy()
    if "parameter_count" in param_rows.columns:
        param_rows = param_rows.loc[param_rows["parameter_count"].notna()]
    if not param_rows.empty and "parameter_count" in param_rows.columns:
        ax.bar(param_rows["predeclared_run_id"].astype(str), param_rows["parameter_count"].astype(float))
        ax.set_ylabel("Total model parameter count")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.set_title("No trained-run parameter counts available")
        ax.set_axis_off()
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
        "target_context_encoder_type",
        "family_gate",
        "context_edge_interaction",
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
    report = f"""# Sprint 8A Target-Context & Context-Edge Interaction Report

Run batch: `{batch_id}`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold; no test-driven selection.
- Feature/loss policy: `S5F2_energy` candidate-edge features and Sprint 6 winner `weighted_bce` with `pos_weight: auto`.
- Architecture policy: Graph C GATv2 only; `context_similar_to` edges dropped; candidate S5F2 active in GATv2 attention and the classifier. The frozen GATv2 attention/message passing is unchanged; the context-edge interaction is head-only.
- Canonical base: Sprint 7F R3 family-aware experimental-emphasis encoder (`S8A_REF_S7F_R3`).
- Selection: validation AUPRC primary (validation MCC / macro F1 tie-break). Test AUPRC is the primary reported test metric; MCC/specificity/macro F1 are secondary threshold diagnostics.
- Family-gate weights, FiLM scale/shift, attention summaries, and parameter counts are interpretation-only model artifacts, not biological causal evidence.

## Result Summary

{_markdown_table(summary)}

## Canonical Run Definitions

- `S8A_R0_base_reference`: Sprint 7F R3 encoder, no gate, no interaction (reproduces the base path).
- `S8A_R1_family_gated_v2`: + SENET-style learned family gate over the four target-context branches.
- `S8A_R2_context_edge_film`: + head-only FiLM interaction between the target-context embedding and S5F2 edge features.
- `S8A_R3_gated_plus_film`: family gate + FiLM (the R1+R2 combination).
- `S8A_R4_regularized_exp_branch`: + bottleneck and feature-dropout on the experimental-epigenetic branch.

## Interpretation Boundaries

- Compare trained rows primarily against `S8A_REF_S7F_R3` (the base) on validation AUPRC; report all runs.
- AUPRC remains the primary metric; rare-negative gains in MCC/specificity are threshold diagnostics, not AUPRC gains.
- Report `parameter_count` next to performance; a gain that tracks added parameters is interpreted cautiously (capacity vs structure).
- A single-seed comparison supports mechanism hypotheses, not statistical superiority; robustness is Sprint 9.
- No architecture, threshold, loss, encoder, interaction, or rerun choice may be selected from test diagnostics.

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
                    "score_decile": int(decile),
                    "rows": int(len(decile_group)),
                    "positives": int(decile_group["label"].sum()),
                    "positive_rate": float(decile_group["label"].mean()),
                    "mean_score": float(decile_group["score"].mean()),
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
        "provenance_type": "sprint8a_target_context_interaction_artifacts_sha256",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "graph_c_artifact_dir": _relative(graph_c_dir),
        "graph_c": _graph_manifest_summary(graph_c_manifest),
        "run_specs": [
            {
                "id": run["id"],
                "graph_schema": GRAPH_C,
                "architecture": run["architecture"],
                "role": run.get("role"),
                **_run_spec_flags(run),
                "drop_context_similarity_edges": True,
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
        "manifest_type": "sprint8a_target_context_interaction_run_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "batch_id": batch_id,
        "config_path": _relative(config_path),
        "reference_run_ids": list(REFERENCE_RUN_IDS),
        "headline_run_ids": list(HEADLINE_RUN_IDS),
        "selection_metric": "validation_auprc",
        "runs": runs,
        "results_path": _relative(results_path),
        "predictions_path": _relative(predictions_path),
        "training_history_path": _relative(history_path),
        "attention_summary_path": _relative(attention_path),
        "target_context_interaction_audit_path": _relative(audit_path),
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
        rows.append(
            {"path": path.relative_to(artifact_dir).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
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
    return f"sprint8a_target_context_interaction_seed{int(config.get('seed', 42))}_{ts}"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True), encoding="utf-8")


def _validate_consolidated_run_ids(results: pd.DataFrame) -> None:
    if "run_id" not in results.columns:
        raise ValueError("Consolidated Sprint 8A results must include run_id")
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Consolidated Sprint 8A results require unique run_id values")


def _reference_metric(results: pd.DataFrame, predeclared_run_id: str, column: str) -> float:
    rows = results.loc[results["predeclared_run_id"] == predeclared_run_id]
    if rows.empty or column not in rows.columns:
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


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_output_dir(
    output_dir_override: str | Path | None,
    config: Mapping[str, Any],
    *,
    default: str,
) -> Path:
    configured = output_dir_override or config.get("outputs", {}).get("output_dir", default)
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
