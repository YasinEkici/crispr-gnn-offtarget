import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_sprint7c_graphc_gatv2_explanation as runner


def test_sprint7c_analysis_writes_output_contract(tmp_path, monkeypatch) -> None:
    _write_required_artifacts(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    output_dir = runner.run_sprint7c_graphc_explanation(output_dir=tmp_path / "outputs" / "sprint7c", write_figures=False)

    required = [
        output_dir / "sprint7c_identity_alignment_audit.md",
        output_dir / "sprint7c_graphc_gatv2_explanation_report.md",
        output_dir / "sprint7c_analysis_manifest.json",
        output_dir / "diagnostics" / "sprint7c_prediction_alignment_audit.csv",
        output_dir / "diagnostics" / "sprint7c_metric_recomputation.csv",
        output_dir / "diagnostics" / "sprint7c_threshold_transfer.csv",
        output_dir / "diagnostics" / "sprint7c_error_transitions.csv",
        output_dir / "diagnostics" / "sprint7c_per_guide_error_gain.csv",
        output_dir / "diagnostics" / "sprint7c_attention_edge_kind_summary.csv",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    transitions = pd.read_csv(output_dir / "diagnostics" / "sprint7c_error_transitions.csv")
    assert set(transitions["transition"]) == {"FP_to_TN", "TN_to_FP", "TP_to_FN", "TP_to_TP"}
    assert int((transitions["transition"] == "FP_to_TN").sum()) == 1
    assert int((transitions["transition"] == "TN_to_FP").sum()) == 1

    metrics = pd.read_csv(output_dir / "diagnostics" / "sprint7c_metric_recomputation.csv")
    graph_c_test = metrics.loc[
        (metrics["analysis_model_id"] == "graph_c_gatv2_s7b") & (metrics["split"] == "test")
    ].iloc[0]
    assert bool(graph_c_test["threshold_available"]) is True
    assert int(graph_c_test["tn"]) == 1
    assert int(graph_c_test["fn"]) == 1

    report = (output_dir / "sprint7c_graphc_gatv2_explanation_report.md").read_text(encoding="utf-8")
    assert "Analysis-only sprint" in report
    assert "Net TN gain" in report
    assert "not topology" in report


def test_sprint7c_alignment_blocks_misaligned_transition() -> None:
    predictions = pd.DataFrame(
        [
            _prediction("left", "test", 0, 1, 0, 0.7),
            _prediction("left", "test", 1, 1, 1, 0.7),
            _prediction("right", "test", 0, 1, 0, 0.2),
            _prediction("right", "test", 2, 1, 1, 0.2),
        ]
    )
    alignment = runner._alignment_audit(predictions)
    assert runner._comparison_passed(alignment, "left", "right", split="test") is False


def _write_required_artifacts(root: Path) -> None:
    _write_predictions(
        root / "outputs/sprint6/loss_comparison/diagnostics_sprint6/sprint6_loss_comparison_predictions.csv",
        "S6R0_wbce",
        model_name="gcn_graph_a_sprint6",
        graph_schema="graph_a_minimal_physical_target",
    )
    _write_predictions(
        root / "outputs/sprint7/diagnostics/gat_predictions.csv",
        "S7R2_gatv2_edge_aware",
        model_name="gatv2_graph_a_sprint7",
        graph_schema="graph_a_minimal_physical_target",
    )
    _write_predictions(
        root / "outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_predictions.csv",
        None,
        model_name="gcn_graph_c_sprint5b_energy",
        graph_schema="graph_c_context_observation",
        include_predeclared=False,
        scores=[0.7, 0.02, 0.8, 0.9],
    )
    _write_sprint7b_predictions(root / "outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv")
    _write_thresholds(root)
    _write_attention(root / "outputs/sprint7b/diagnostics/gatv2_topology_attention_summary.csv")


def _write_predictions(
    path: Path,
    predeclared_run_id: str | None,
    *,
    model_name: str,
    graph_schema: str,
    include_predeclared: bool = True,
    scores: list[float] | None = None,
) -> None:
    rows = []
    scores = scores or [0.8, 0.1, 0.7, 0.6]
    for split in ["val", "test"]:
        for index, (label, score) in enumerate(zip([0, 0, 1, 1], scores, strict=True)):
            row = _prediction(model_name, split, index, 10 + index, label, score)
            row["model_name"] = model_name
            row["graph_schema"] = graph_schema
            row["feature_set"] = "S5F2_energy"
            row["genome"] = "hg19" if "graph_c" not in graph_schema else None
            if include_predeclared:
                row["predeclared_run_id"] = predeclared_run_id
            rows.append(row)
    _write_csv(path, pd.DataFrame(rows))


def _write_sprint7b_predictions(path: Path) -> None:
    rows = []
    specs = {
        "S7B_R1_graph_b_gcn_s5f2": ("gcn_graph_b", "graph_b_guide_similarity_control", [0.8, 0.1, 0.7, 0.6]),
        "S7B_R2_graph_b_gatv2_s5f2": ("gatv2_graph_b", "graph_b_guide_similarity_control", [0.6, 0.2, 0.8, 0.7]),
        "S7B_R3_graph_c_gatv2_s5f2": ("gatv2_graph_c", "graph_c_context_observation", [0.1, 0.2, 0.05, 0.9]),
    }
    for run_id, (model_name, graph_schema, scores) in specs.items():
        for split in ["val", "test"]:
            for index, (label, score) in enumerate(zip([0, 0, 1, 1], scores, strict=True)):
                row = _prediction(model_name, split, index, 10 + index, label, score)
                row["run_id"] = f"batch_{run_id}"
                row["predeclared_run_id"] = run_id
                row["model_name"] = model_name
                row["architecture"] = "gatv2" if "gatv2" in model_name else "gcn"
                row["graph_schema"] = graph_schema
                row["feature_set"] = "S5F2_energy"
                row["genome"] = "hg19" if "graph_b" in graph_schema else None
                rows.append(row)
    _write_csv(path, pd.DataFrame(rows))


def _prediction(model_id: str, split: str, row_index: int, guide: int, label: int, score: float) -> dict[str, object]:
    return {
        "analysis_model_id": model_id,
        "analysis_predeclared_run_id": model_id,
        "analysis_source_label": model_id,
        "split": split,
        "row_index": row_index,
        "grna_target_id": guide,
        "genome": "hg19",
        "label": label,
        "score": score,
    }


def _write_thresholds(root: Path) -> None:
    _write_csv(
        root / "outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_fixed_threshold_metrics.csv",
        pd.DataFrame([{"split": "test", "threshold": 0.05}, {"split": "val", "threshold": 0.05}]),
    )
    _write_csv(
        root / "outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_threshold_metrics.csv",
        pd.DataFrame([{"run_id": "batch_S6R0_wbce", "split": "test", "threshold": 0.5}]),
    )
    _write_csv(
        root / "outputs/sprint7/gat_comparison.csv",
        pd.DataFrame([{"predeclared_run_id": "S7R2_gatv2_edge_aware", "threshold": 0.5}]),
    )
    _write_csv(
        root / "outputs/sprint7b/gatv2_topology_comparison.csv",
        pd.DataFrame(
            [
                {"predeclared_run_id": "S7B_R1_graph_b_gcn_s5f2", "threshold": 0.5},
                {"predeclared_run_id": "S7B_R2_graph_b_gatv2_s5f2", "threshold": 0.5},
                {"predeclared_run_id": "S7B_R3_graph_c_gatv2_s5f2", "threshold": 0.15},
            ]
        ),
    )


def _write_attention(path: Path) -> None:
    _write_csv(
        path,
        pd.DataFrame(
            [
                {
                    "predeclared_run_id": "S7B_R3_graph_c_gatv2_s5f2",
                    "graph_schema": "graph_c_context_observation",
                    "architecture": "gatv2",
                    "split": "test",
                    "edge_kind": "context_similar_to",
                    "edge_count": 10,
                    "attention_mean": 0.2,
                    "attention_std": 0.1,
                    "attention_min": 0.0,
                    "attention_max": 0.5,
                }
            ]
        ),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
