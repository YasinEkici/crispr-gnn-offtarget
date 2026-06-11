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

from crispr_gnn.graph.graph_schemas import GRAPH_C  # noqa: E402
from crispr_gnn.graph.pyg_dataset import LABEL_SCHEME, SPLIT_ID, VISIBILITY_POLICY, Sprint3HeteroDataLoader  # noqa: E402
from crispr_gnn.models.gcn import GRAPH_C_EDGE_TYPE  # noqa: E402
from crispr_gnn.models.sequence_context_encoder import build_s1_pair_for_edges, sequence_input_audit  # noqa: E402
from crispr_gnn.training.gcn import (  # noqa: E402
    collect_context_edge_interaction_summary,
    collect_graph_attention_summary,
    collect_target_context_encoder_summary,
    gcn_run_config_from_mapping,
    train_graph_c_gcn,
)
from crispr_gnn.utils.config import load_yaml  # noqa: E402


REFERENCE_RUN_IDS = ("S8B_R0_reference",)
HEADLINE_RUN_IDS = ("S8B_R1_sequence_only", "S8B_R2_sequence_plus_context")
CANONICAL_RUN_FLAGS = {
    "S8B_R1_sequence_only": {
        "sequence_context_mode": "sequence_only",
        "context_edge_interaction": "none",
        "target_context_encoder_type": None,
    },
    "S8B_R2_sequence_plus_context": {
        "sequence_context_mode": "late_fusion",
        "context_edge_interaction": "film",
        "target_context_encoder_type": "family_aware_experimental_emphasis",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sprint 8B sequence-context encoder comparison.")
    parser.add_argument("--config", default="configs/sweeps/sprint8b_sequence_context.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for smoke/debug runs.")
    parser.add_argument("--run", action="append", default=None, help="Run only this predeclared run ID. May repeat.")
    parser.add_argument("--run-id", default=None, help="Batch ID recorded in manifest and per-run IDs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_sprint8b_sequence_context(
        config_path=ROOT / args.config,
        batch_id=args.run_id,
        max_epochs=args.max_epochs,
        selected_run_ids=args.run,
    )
    return 0


def run_sprint8b_sequence_context(
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

    output_dir = ROOT / str(config.get("outputs", {}).get("output_dir", "outputs/sprint8b"))
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
    all_sequence_audits: list[pd.DataFrame] = []
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

        sequence_audit = _sequence_input_audit(materialized, run_config, run_id=run_id, run_spec=run_spec)
        sequence_audit.to_csv(run_dir / "sequence_input_audit.csv", index=False)
        history.to_csv(run_dir / "training_history.csv", index=False)
        results.to_csv(run_dir / "metrics.csv", index=False)

        attention_path = None
        encoder_summary_path = None
        interaction_summary_path = None
        if run_config.sequence_context_mode != "sequence_only":
            attention = collect_graph_attention_summary(
                materialized, run_config, checkpoint_path=checkpoint_path, split="test"
            )
            attention.insert(0, "run_id", run_id)
            attention.insert(1, "predeclared_run_id", run_spec["id"])
            attention.to_csv(run_dir / "attention_summary.csv", index=False)
            all_attention.append(attention)
            attention_path = run_dir / "attention_summary.csv"

            encoder_summary = collect_target_context_encoder_summary(
                materialized, run_config, checkpoint_path=checkpoint_path, split="test"
            )
            encoder_summary.insert(0, "run_id", run_id)
            encoder_summary.insert(1, "predeclared_run_id", run_spec["id"])
            encoder_summary.to_csv(run_dir / "target_context_encoder_summary.csv", index=False)
            all_encoder_summary.append(encoder_summary)
            encoder_summary_path = run_dir / "target_context_encoder_summary.csv"

            interaction_summary = collect_context_edge_interaction_summary(
                materialized, run_config, checkpoint_path=checkpoint_path, split="test"
            )
            if not interaction_summary.empty:
                interaction_summary.insert(0, "run_id", run_id)
                interaction_summary.insert(1, "predeclared_run_id", run_spec["id"])
                interaction_summary.to_csv(run_dir / "context_edge_interaction_summary.csv", index=False)
                all_interaction_summary.append(interaction_summary)
                interaction_summary_path = run_dir / "context_edge_interaction_summary.csv"

        all_results.append(results)
        all_predictions.append(predictions)
        all_history.append(history)
        all_sequence_audits.append(sequence_audit)
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
                "sequence_input_audit_path": _relative(run_dir / "sequence_input_audit.csv"),
                "attention_summary_path": _relative(attention_path) if attention_path else None,
                "target_context_encoder_summary_path": (
                    _relative(encoder_summary_path) if encoder_summary_path else None
                ),
                "context_edge_interaction_summary_path": (
                    _relative(interaction_summary_path) if interaction_summary_path else None
                ),
                "checkpoint_path": _relative(checkpoint_path),
                "resolved_config_path": _relative(run_dir / "resolved_config.yaml"),
            }
        )
        row = results.iloc[0]
        print(
            f"{run_spec['id']}: test_auprc={float(row['test_auprc']):.6f}, "
            f"test_mcc={float(row['test_mcc']):.6f}, sequence_mode={run_config.sequence_context_mode}"
        )

    result_table = pd.concat(all_results, ignore_index=True)
    prediction_table = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    history_table = pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()
    sequence_audit_table = pd.concat(all_sequence_audits, ignore_index=True) if all_sequence_audits else pd.DataFrame()
    attention_table = pd.concat(all_attention, ignore_index=True) if all_attention else pd.DataFrame()
    encoder_summary_table = pd.concat(all_encoder_summary, ignore_index=True) if all_encoder_summary else pd.DataFrame()
    interaction_table = (
        pd.concat(all_interaction_summary, ignore_index=True) if all_interaction_summary else pd.DataFrame()
    )
    _validate_consolidated_run_ids(result_table)

    results_path = output_dir / "sequence_context_comparison.csv"
    predictions_path = diagnostics_dir / "sequence_context_predictions.csv"
    history_path = diagnostics_dir / "sequence_context_training_history.csv"
    sequence_audit_path = diagnostics_dir / "sequence_context_sequence_input_audit.csv"
    result_table.to_csv(results_path, index=False)
    prediction_table.to_csv(predictions_path, index=False)
    history_table.to_csv(history_path, index=False)
    sequence_audit_table.to_csv(sequence_audit_path, index=False)

    diagnostic_tables = _write_diagnostics(
        result_table,
        prediction_table,
        sequence_audit_table,
        attention_table,
        encoder_summary_table,
        interaction_table,
        diagnostics_dir,
    )
    figure_paths = _write_figures(result_table, prediction_table, history_table, figures_dir)
    report_path = _write_report(
        result_table,
        diagnostic_tables,
        figure_paths,
        output_dir / "sequence_context_report.md",
        batch_id=batch,
    )
    manifest_path = output_dir / "sequence_context_run_manifest.json"
    _write_run_manifest(
        manifest_path,
        config_path=Path(config_path),
        batch_id=batch,
        runs=manifest_runs,
        results_path=results_path,
        predictions_path=predictions_path,
        history_path=history_path,
        sequence_audit_path=sequence_audit_path,
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
    if config.get("sprint") != "sprint8b":
        raise ValueError("Sprint 8B runner requires sprint: sprint8b")
    if config.get("task") != "sprint8b_sequence_context":
        raise ValueError("Sprint 8B runner requires task: sprint8b_sequence_context")
    data = config.get("data", {})
    graph = config.get("graph", {})
    evaluation = config.get("evaluation", {})
    if data.get("label_scheme") != LABEL_SCHEME or data.get("split_id") != SPLIT_ID:
        raise ValueError("Sprint 8B must preserve Scheme A and sprint2_main_seed42")
    if graph.get("visibility_policy") != VISIBILITY_POLICY:
        raise ValueError("Sprint 8B must preserve strict-inductive visibility")
    if evaluation.get("protocol") != "headline_guide_level" or evaluation.get("no_test_tuning") is not True:
        raise ValueError("Sprint 8B must preserve headline guide-level no-test-tuning evaluation")
    if evaluation.get("pretrained_weights_allowed") is not False:
        raise ValueError("Sprint 8B same-contract rows must not use external pretrained weights")
    features = config.get("features", {})
    if features.get("feature_set") != "S5F2_energy" or features.get("edge_feature_sets") != ["s5f2_energy"]:
        raise ValueError("Sprint 8B late-fusion anchor is frozen to S5F2_energy")
    training = config.get("training", {})
    if str(training.get("loss", "")).lower() != "weighted_bce":
        raise ValueError("Sprint 8B must keep weighted_bce")
    if training.get("loss_params", {}).get("pos_weight") != "auto":
        raise ValueError("Sprint 8B weighted BCE must keep pos_weight: auto")
    configured_ids = [str(run.get("id")) for run in config.get("runs", [])]
    if configured_ids != list(HEADLINE_RUN_IDS):
        raise ValueError("Sprint 8B headline run list/order drift")
    for run in config.get("runs", []):
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
            raise ValueError(f"Unknown Sprint 8B run ID '{run_id}'. Allowed IDs: {allowed}")
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
    run_config["graph"] = {"schema": GRAPH_C, "materialization": "heterodata", "visibility_policy": VISIBILITY_POLICY}
    run_config["model"] = dict(run_config.get("model", {}))
    run_config["model"]["name"] = str(run_spec["model_name"])
    run_config["model"]["architecture"] = "gatv2"
    run_config["model"]["target_node_representation"] = "target_observation_context_encoder"
    run_config["model"]["target_context_encoder"] = dict(run_spec["target_context_encoder"])
    run_config["model"]["target_observation_mask_families"] = []
    run_config["model"]["context_edge_interaction"] = str(run_spec.get("context_edge_interaction", "none"))
    run_config["model"]["interaction_edge_dim"] = int(run_spec.get("interaction_edge_dim", 64))
    run_config["model"]["sequence_context_encoder"] = dict(run_spec["sequence_context_encoder"])
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
    run_config["sprint8b_run"] = {
        "id": run_spec["id"],
        "role": run_spec.get("role"),
        "graph_schema": GRAPH_C,
        "architecture": "gatv2",
        "feature_set": "S5F2_energy",
        "loss": "weighted_bce",
        "loss_params": {"pos_weight": "auto"},
        **_run_spec_flags(run_spec),
        "from_scratch": True,
        "external_pretrained_weights": False,
        "drop_context_similarity_edges": True,
    }
    return run_config


def _validate_graph_c_artifacts(manifest: Mapping[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_C:
        raise ValueError("Sprint 8B Graph C artifacts have wrong schema")
    feature_tables = manifest.get("feature_tables", {})
    if int(feature_tables.get("S5F2_energy", 0)) != 268:
        raise ValueError("Sprint 8B Graph C requires Sprint 5B S5F2_energy feature table")
    if int(feature_tables.get("target_observation_features", 0)) != 212:
        raise ValueError("Sprint 8B Graph C requires 212 target_observation_features")


def _run_spec_flags(run_spec: Mapping[str, Any]) -> dict[str, object]:
    encoder = run_spec.get("target_context_encoder", {}) or {}
    sequence = run_spec.get("sequence_context_encoder", {}) or {}
    mode = str(sequence.get("mode", "none"))
    return {
        "target_context_encoder_type": (
            None if mode == "sequence_only" else str(encoder.get("type", "family_aware_experimental_emphasis"))
        ),
        "context_edge_interaction": str(run_spec.get("context_edge_interaction", "none")),
        "interaction_edge_dim": int(run_spec.get("interaction_edge_dim", 64)),
        "sequence_context_mode": mode,
        "sequence_embed_dim": int(sequence.get("embed_dim", 64)),
        "sequence_conv_channels": int(sequence.get("conv_channels", 32)),
        "sequence_conv_kernel": int(sequence.get("conv_kernel", 3)),
        "sequence_lstm_hidden": int(sequence.get("lstm_hidden", 32)),
        "sequence_dropout": float(sequence.get("dropout", 0.2)),
    }


def _validate_canonical_run_flags(run_spec: Mapping[str, Any]) -> None:
    run_id = str(run_spec.get("id"))
    expected = CANONICAL_RUN_FLAGS.get(run_id)
    if expected is None:
        raise ValueError(f"Unexpected Sprint 8B run ID: {run_id}")
    observed = _run_spec_flags(run_spec)
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"Sprint 8B canonical matrix drift for {run_id}: {key}={observed.get(key)!r}")
    if run_spec.get("architecture") != "gatv2":
        raise ValueError("Sprint 8B trains Graph C GATv2-dispatched runs only")


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
    for frame in [results, predictions, history]:
        frame.insert(0, "run_id", run_id)
        frame.insert(1, "predeclared_run_id", run_spec["id"])
        frame.insert(2, "run_order", order)
    for key, value in flags.items():
        results[key] = value
    results["role"] = run_spec.get("role")
    results["from_scratch"] = True
    results["external_pretrained_weights"] = False
    predictions["sequence_context_mode"] = flags["sequence_context_mode"]


def _sequence_input_audit(
    materialized: Any,
    config: Any,
    *,
    run_id: str,
    run_spec: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        view = materialized.view(split)
        guide_names = list(getattr(view["sgRNA"], "feature_names", []))
        target_names = list(getattr(view["target_observation"], "feature_names", []))
        audit = sequence_input_audit(guide_feature_names=guide_names, target_feature_names=target_names)
        s1 = build_s1_pair_for_edges(
            guide_node_x=view["sgRNA"].x,
            guide_feature_names=guide_names,
            target_node_x=view["target_observation"].x,
            target_feature_names=target_names,
            edge_index=view[GRAPH_C_EDGE_TYPE].edge_index,
        )
        rows.append(
            {
                "run_id": run_id,
                "predeclared_run_id": run_spec["id"],
                "split": split,
                "graph_schema": GRAPH_C,
                "sequence_context_mode": config.sequence_context_mode,
                "representation": audit["representation"],
                "positions": audit["positions"],
                "channels": audit["channels"],
                "guide_onehot_columns": audit["guide_onehot_columns"],
                "target_onehot_columns": audit["target_onehot_columns"],
                "mismatch_channel": audit["mismatch_channel"],
                "candidate_edges": int(s1.shape[0]),
                "s1_shape": "x".join(str(value) for value in s1.shape),
                "s1_abs_sum": float(s1.abs().sum()),
                "policy": audit["policy"],
                "source": audit["source"],
                "external_pretrained_weights": False,
                "no_reproduction_claim": True,
            }
        )
    return pd.DataFrame(rows)


def _reference_rows(config: Mapping[str, Any], *, batch_id: str) -> list[dict[str, object]]:
    prior = config.get("prior_context", {})
    r0 = prior.get("s8a_r2_context_edge_film", {})
    prevalence = prior.get("test_positive_prevalence", 0.900705)
    return [
        {
            "run_id": f"{batch_id}_S8B_R0_reference",
            "predeclared_run_id": "S8B_R0_reference",
            "run_order": 0,
            "role": "carry_forward_s8a_r2_no_retrain",
            "source": r0.get("source"),
            "source_batch": r0.get("source_batch"),
            "sprint": "sprint8b",
            "graph_schema": GRAPH_C,
            "architecture": "gatv2",
            "feature_set": "S5F2_energy",
            "loss": "weighted_bce",
            "target_context_encoder_type": "family_aware_experimental_emphasis",
            "context_edge_interaction": "film",
            "sequence_context_mode": "reference_no_sequence_encoder",
            "sequence_embed_dim": None,
            "from_scratch": False,
            "external_pretrained_weights": False,
            "test_positive_rate": prevalence,
            "best_val_auprc": _optional_float(r0.get("best_val_auprc")),
            "test_auprc": _optional_float(r0.get("test_auprc")),
            "test_auroc": _optional_float(r0.get("test_auroc")),
            "test_macro_f1": _optional_float(r0.get("test_macro_f1")),
            "test_mcc": _optional_float(r0.get("test_mcc")),
            "test_specificity": _optional_float(r0.get("test_specificity")),
            "test_sensitivity": _optional_float(r0.get("test_sensitivity")),
            "test_tn": _optional_int(r0.get("test_tn")),
            "test_fp": _optional_int(r0.get("test_fp")),
            "test_fn": _optional_int(r0.get("test_fn")),
            "test_tp": _optional_int(r0.get("test_tp")),
            "parameter_count": _optional_int(r0.get("parameter_count")),
            "active_parameter_count": None,
        }
    ]


def _write_diagnostics(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    sequence_audit: pd.DataFrame,
    attention: pd.DataFrame,
    encoder_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    diagnostics_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    threshold_path = diagnostics_dir / "sequence_context_threshold_metrics.csv"
    threshold_columns = [
        "run_id",
        "predeclared_run_id",
        "sequence_context_mode",
        "test_positive_rate",
        "test_f1",
        "test_macro_f1",
        "test_mcc",
        "test_specificity",
        "test_sensitivity",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
    ]
    results[[column for column in threshold_columns if column in results.columns]].to_csv(threshold_path, index=False)
    paths.append(threshold_path)

    deltas = results.copy()
    r0_auprc = _reference_metric(deltas, "S8B_R0_reference", "test_auprc")
    r0_mcc = _reference_metric(deltas, "S8B_R0_reference", "test_mcc")
    xgb_auprc = float(results.attrs.get("xgb_auprc", 0.992522))
    deltas["delta_auprc_vs_s8a_r2_reference"] = deltas["test_auprc"].astype(float) - r0_auprc
    deltas["delta_mcc_vs_s8a_r2_reference"] = deltas["test_mcc"].astype(float) - r0_mcc
    deltas["delta_auprc_vs_xgboost_f4"] = deltas["test_auprc"].astype(float) - xgb_auprc
    delta_path = diagnostics_dir / "sequence_context_deltas.csv"
    delta_columns = [
        "run_id",
        "predeclared_run_id",
        "sequence_context_mode",
        "test_auprc",
        "delta_auprc_vs_s8a_r2_reference",
        "delta_mcc_vs_s8a_r2_reference",
        "delta_auprc_vs_xgboost_f4",
        "test_mcc",
        "test_specificity",
        "test_tn",
        "test_fp",
    ]
    deltas[[column for column in delta_columns if column in deltas.columns]].to_csv(delta_path, index=False)
    paths.append(delta_path)

    sequence_audit_path = diagnostics_dir / "sequence_context_sequence_input_audit.csv"
    sequence_audit.to_csv(sequence_audit_path, index=False)
    paths.append(sequence_audit_path)

    param_path = diagnostics_dir / "sequence_context_parameter_counts.csv"
    param_columns = [
        "run_id",
        "predeclared_run_id",
        "sequence_context_mode",
        "sequence_embed_dim",
        "parameter_count",
        "active_parameter_count",
        "test_auprc",
    ]
    results[[column for column in param_columns if column in results.columns]].to_csv(param_path, index=False)
    paths.append(param_path)

    attention_path = diagnostics_dir / "sequence_context_attention_summary.csv"
    if attention.empty:
        pd.DataFrame([{"note": "No attention rows were produced for sequence-only runs."}]).to_csv(
            attention_path, index=False
        )
    else:
        attention.to_csv(attention_path, index=False)
    paths.append(attention_path)

    encoder_path = diagnostics_dir / "sequence_context_target_context_encoder_summary.csv"
    if encoder_summary.empty:
        pd.DataFrame([{"note": "No target-context encoder summary rows were produced."}]).to_csv(
            encoder_path, index=False
        )
    else:
        encoder_summary.to_csv(encoder_path, index=False)
    paths.append(encoder_path)

    interaction_path = diagnostics_dir / "sequence_context_film_summary.csv"
    if interaction_summary.empty:
        pd.DataFrame([{"note": "No context-edge interaction rows were produced."}]).to_csv(
            interaction_path, index=False
        )
    else:
        interaction_summary.to_csv(interaction_path, index=False)
    paths.append(interaction_path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        per_guide_path = diagnostics_dir / "sequence_context_per_guide_score_summary.csv"
        _per_group_score_summary(
            test_predictions,
            group_columns=["run_id", "predeclared_run_id", "graph_schema", "architecture", "grna_target_id"],
        ).to_csv(per_guide_path, index=False)
        paths.append(per_guide_path)

        deciles_path = diagnostics_dir / "sequence_context_score_deciles.csv"
        _score_deciles(test_predictions).to_csv(deciles_path, index=False)
        paths.append(deciles_path)
    return paths


def _write_figures(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    paths: list[Path] = []
    rows = results.sort_values("run_order")
    labels = rows["predeclared_run_id"].astype(str).tolist()

    path = figures_dir / "sequence_context_auprc_comparison.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, rows["test_auprc"].astype(float))
    ax.axhline(0.992522, color="black", linestyle="--", linewidth=1, label="XGBoost F4")
    ax.axhline(0.900705, color="gray", linestyle=":", linewidth=1, label="test prevalence")
    ax.set_ylabel("Test AUPRC")
    ax.set_ylim(0.85, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "sequence_context_threshold_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    rows.set_index("predeclared_run_id")[["test_mcc", "test_specificity", "test_macro_f1"]].astype(float).plot(
        kind="bar", ax=ax
    )
    ax.set_ylabel("Metric")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if not predictions.empty:
        test_predictions = predictions.loc[predictions["split"] == "test"].copy()
        path = figures_dir / "sequence_context_pr_curves.png"
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

        path = figures_dir / "sequence_context_roc_curves.png"
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

        path = figures_dir / "sequence_context_score_distributions.png"
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

    path = figures_dir / "sequence_context_training_curves.png"
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

    path = figures_dir / "sequence_context_parameter_counts.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    param_rows = rows.loc[rows["parameter_count"].notna()].copy()
    if not param_rows.empty:
        ax.bar(param_rows["predeclared_run_id"].astype(str), param_rows["parameter_count"].astype(float), alpha=0.6)
        if "active_parameter_count" in param_rows.columns:
            active = param_rows.loc[param_rows["active_parameter_count"].notna()]
            if not active.empty:
                ax.scatter(active["predeclared_run_id"].astype(str), active["active_parameter_count"].astype(float), color="black")
        ax.set_ylabel("Parameters (bars nominal; points active)")
        ax.tick_params(axis="x", rotation=20)
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
        "sequence_context_mode",
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
        "parameter_count",
        "active_parameter_count",
    ]
    summary = rows[[column for column in summary_columns if column in rows.columns]].copy()
    report = f"""# Sprint 8B Sequence-Context Encoder Report

Run batch: `{batch_id}`

## Contract

- Label/split/evaluation: frozen Scheme A, `sprint2_main_seed42`, measured-only headline, validation-only checkpoint and threshold; no test-driven selection.
- Run matrix: `S8B_R0_reference` is the Sprint 8A R2 carry-forward row; only `S8B_R1_sequence_only` and `S8B_R2_sequence_plus_context` are trained.
- Sequence policy: S1 guide/target pair reconstructed from Graph C one-hot node features; no raw-data join and no energy/epigenetic/context leakage into the sequence branch.
- Architecture policy: CRISPR-Net/CRISPR-IP-adapted Conv+BiLSTM, trained from scratch; no externally pretrained CRISPR/genomic weights; no reproduction claim.
- Selection: validation AUPRC primary. Test AUPRC is the primary reported test metric; MCC/specificity/macro F1 are secondary threshold diagnostics.
- Parameter reporting: nominal and active parameter counts are reported separately because Sprint 8A interaction mode can instantiate inactive classifier parameters.

## Result Summary

{_markdown_table(summary)}

## Interpretation Boundaries

- `S8B_R1_sequence_only` tests pure sequence-context value under the 8A harness.
- `S8B_R2_sequence_plus_context` tests whether S1 sequence context adds over the Sprint 8A R2 target-context + FiLM candidate.
- A single seed supports directional mechanism evidence only; robustness is Sprint 9.
- No architecture, threshold, loss, encoder, or rerun choice may be selected from test diagnostics.

## Artifact Index

Diagnostic tables:
{chr(10).join(f'- `{_relative(item)}`' for item in diagnostic_tables)}

Figures:
{chr(10).join(f'- `{_relative(item)}`' for item in figure_paths)}
"""
    path.write_text(report, encoding="utf-8")
    return path


def _write_graph_provenance(
    path: Path,
    *,
    graph_c_dir: Path,
    graph_c_manifest: Mapping[str, Any],
    run_specs: list[Mapping[str, Any]],
) -> None:
    payload = {
        "provenance_type": "sprint8b_sequence_context_artifacts_sha256",
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
                "from_scratch": True,
                "external_pretrained_weights": False,
            }
            for run in run_specs
        ],
        "files": {GRAPH_C: _artifact_file_hashes(graph_c_dir / GRAPH_C)},
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
    sequence_audit_path: Path,
    report_path: Path,
    diagnostics: list[Path],
    figures: list[Path],
    provenance_path: Path,
) -> None:
    payload = {
        "manifest_type": "sprint8b_sequence_context_run_manifest",
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
        "sequence_input_audit_path": _relative(sequence_audit_path),
        "report_path": _relative(report_path),
        "graph_artifact_provenance_path": _relative(provenance_path),
        "diagnostic_tables": [_relative(item) for item in diagnostics],
        "figures": [_relative(item) for item in figures],
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
    return f"sprint8b_sequence_context_seed{int(config.get('seed', 42))}_{ts}"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=True), encoding="utf-8")


def _validate_consolidated_run_ids(results: pd.DataFrame) -> None:
    if "run_id" not in results.columns:
        raise ValueError("Consolidated Sprint 8B results must include run_id")
    if results["run_id"].astype(str).duplicated().any():
        raise ValueError("Consolidated Sprint 8B results require unique run_id values")


def _reference_metric(results: pd.DataFrame, predeclared_run_id: str, column: str) -> float:
    rows = results.loc[results["predeclared_run_id"] == predeclared_run_id]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


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


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
