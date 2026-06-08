import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_sprint7_gat_comparison as runner


def test_sprint7_runner_writes_output_contract_and_attention_provenance(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    graph_root = tmp_path / "graphs"
    graph_a_dir = graph_root / "graph_a_minimal_physical_target"
    graph_a_dir.mkdir(parents=True)
    (graph_a_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (graph_a_dir / "dummy.parquet").write_bytes(b"not-real-parquet-needed-for-hash-only")

    call_order: list[str] = []
    original_write_graph_provenance = runner._write_graph_provenance

    def write_graph_provenance(*args, **kwargs) -> None:
        call_order.append("provenance")
        original_write_graph_provenance(*args, **kwargs)

    def train_graph_a_gcn(*args, **kwargs):
        call_order.append("train")
        return _fake_train_graph_a_gcn(*args, **kwargs)

    def collect_graph_a_attention_summary(*args, **kwargs):
        call_order.append("attention")
        return _fake_attention()

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "_write_graph_provenance", write_graph_provenance)
    monkeypatch.setattr(runner, "train_graph_a_gcn", train_graph_a_gcn)
    monkeypatch.setattr(runner, "collect_graph_a_attention_summary", collect_graph_a_attention_summary)
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7_gat_comparison(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
        selected_run_ids=["S7R1_gat_edge_aware"],
    )

    assert call_order == ["provenance", "train", "attention"]
    assert output_dir == tmp_path / "outputs" / "sprint7"
    required = [
        output_dir / "gat_comparison.csv",
        output_dir / "gat_report.md",
        output_dir / "gat_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "gat_threshold_metrics.csv",
        output_dir / "diagnostics" / "attention_weight_summary.csv",
        output_dir / "diagnostics" / "attention_contract_summary.csv",
        output_dir / "figures" / "gat_model_auprc_comparison.png",
        output_dir / "figures" / "attention_weight_summary.png",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "resolved_config.yaml",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "runtime.json",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "training_history.csv",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "metrics.csv",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "attention_summary.csv",
        output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "model.pt",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "gat_comparison.csv")
    assert results["run_id"].is_unique
    assert set(results["predeclared_run_id"]) == {"S7R1_gat_edge_aware"}
    assert results.iloc[0]["architecture"] == "gat"
    assert bool(results.iloc[0]["edge_aware_attention"]) is True
    assert "run_id" in pd.read_csv(output_dir / "diagnostics" / "gat_predictions.csv").columns
    assert "candidate_forward" in set(pd.read_csv(output_dir / "diagnostics" / "attention_weight_summary.csv")["edge_kind"])

    resolved = yaml.safe_load(
        (output_dir / "runs" / "batch_S7R1_gat_edge_aware" / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["model"]["architecture"] == "gat"
    assert resolved["model"]["attention"]["edge_aware"] is True
    assert resolved["model"]["attention"]["self_loop_edge_fill"] == 0.0
    assert resolved["training"]["loss"] == "weighted_bce"
    assert resolved["training"]["loss_params"] == {"pos_weight": "auto"}

    manifest = json.loads((output_dir / "gat_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_runs_executed"] == []
    assert manifest["runs"][0]["run_id"] == "batch_S7R1_gat_edge_aware"
    assert manifest["runs"][0]["attention_summary_path"].endswith("attention_summary.csv")


def test_sprint7_runner_default_includes_sprint6_reference_and_two_headline_runs(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    graph_root = tmp_path / "graphs"
    graph_a_dir = graph_root / "graph_a_minimal_physical_target"
    graph_a_dir.mkdir(parents=True)
    (graph_a_dir / "dummy.parquet").write_bytes(b"not-real-parquet-needed-for-hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_a_gcn", _fake_train_graph_a_gcn)
    monkeypatch.setattr(runner, "collect_graph_a_attention_summary", lambda *args, **kwargs: _fake_attention())
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7_gat_comparison(config_path=config_path, batch_id="batch", max_epochs=1)

    results = pd.read_csv(output_dir / "gat_comparison.csv")
    assert set(results["predeclared_run_id"]) == {
        "S7R0_gcn_reference",
        "S7R1_gat_edge_aware",
        "S7R2_gatv2_edge_aware",
    }
    assert list(results.sort_values("run_order")["predeclared_run_id"]) == [
        "S7R0_gcn_reference",
        "S7R1_gat_edge_aware",
        "S7R2_gatv2_edge_aware",
    ]


def test_sprint7_runner_rejects_optional_edge_blind_run_without_explicit_opt_in(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = runner.load_yaml(config_path)

    with pytest.raises(ValueError, match="requires --include-optional-runs"):
        runner._selected_run_specs(config, ["S7R3_gat_edge_blind_control"], include_optional_runs=False)


def _write_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sweeps" / "sprint7_gat_gatv2.yaml")
    source["data"]["graph_artifact_dir"] = "graphs"
    source["outputs"]["output_dir"] = "outputs/sprint7"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint7_gat_gatv2.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
    return path


class _FakeMaterialized:
    graph_name = "graph_a_minimal_physical_target"
    manifest = {
        "graph_name": "graph_a_minimal_physical_target",
        "split_id": "sprint2_main_seed42",
        "label_scheme": "scheme_a",
        "feature_tables": {"S5F2_energy": 268},
        "metadata": {
            "visibility_policy": "strict_inductive_primary",
            "sprint5_feature_ablation": {"topology": "Graph A unchanged"},
        },
        "preprocessing": {"sprint5_feature_ablation": {"fit_scope": "train_only"}},
    }


class _FakeLoader:
    def __init__(self, _path: Path) -> None:
        self.path = _path

    def load(self, graph_name: str) -> _FakeMaterialized:
        assert graph_name == "graph_a_minimal_physical_target"
        return _FakeMaterialized()


def _fake_loader_factory(path: Path) -> _FakeLoader:
    return _FakeLoader(path)


def _fake_train_graph_a_gcn(_materialized, config, *, checkpoint_path: Path):
    checkpoint_path.write_bytes(b"fake-checkpoint")
    architecture_offset = {"gat": 0.001, "gatv2": 0.002}.get(config.architecture, 0.0)
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint7",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": config.model_name,
                "architecture": config.architecture,
                "feature_set": "S5F2_energy",
                "graph_schema": "graph_a_minimal_physical_target",
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": "zero_type_feature",
                "loss": "weighted_bce",
                "threshold": 0.5,
                "threshold_selection_split": "validation",
                "checkpoint_selection_split": "validation",
                "baseline_reference": "xgboost_unweighted / F4",
                "baseline_test_auprc": 0.992522,
                "baseline_test_auroc": 0.938416,
                "baseline_test_mcc": 0.345198,
                "test_positive_rate": 0.900705,
                "test_auprc": 0.976000 + architecture_offset,
                "test_auroc": 0.820000,
                "test_f1": 0.910000,
                "test_macro_f1": 0.720000,
                "test_mcc": 0.480000,
                "test_specificity": 0.290000,
                "test_sensitivity": 0.996000,
                "test_tn": 49,
                "test_fp": 120,
                "test_fn": 6,
                "test_tp": 1527,
                "edge_aware_attention": config.edge_aware_attention,
                "attention_heads": config.attention_heads,
                "attention_concat": config.attention_concat,
                "attention_dropout": config.attention_dropout,
                "self_loop_edge_fill": config.self_loop_edge_fill,
                "parameter_count": 12345,
            }
        ]
    )
    predictions = _fake_predictions(config.architecture)
    history = pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "graph_schema": "graph_a_minimal_physical_target",
                "feature_set": "S5F2_energy",
                "epoch": 1,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_auprc": 0.8,
                "lr": 0.001,
                "selection_split": "validation",
            }
        ]
    )
    return results, predictions, history


def _fake_predictions(architecture: str) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate(
            [(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.6), (1, 0.8), (1, 0.4)]
        ):
            rows.append(
                {
                    "model_name": f"{architecture}_model",
                    "architecture": architecture,
                    "graph_schema": "graph_a_minimal_physical_target",
                    "feature_set": "S5F2_energy",
                    "split": split,
                    "row_index": index,
                    "grna_target_id": f"guide_{index % 3}",
                    "genome": "hg19" if index < 3 else "mm10",
                    "label": label,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def _fake_attention() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "gat_graph_a_sprint7_edge_aware",
                "architecture": "gat",
                "split": "test",
                "layer": 0,
                "head": 0,
                "edge_kind": "candidate_forward",
                "edge_count": 3,
                "attention_mean": 0.4,
                "attention_std": 0.1,
                "attention_min": 0.2,
                "attention_max": 0.6,
            },
            {
                "model_name": "gat_graph_a_sprint7_edge_aware",
                "architecture": "gat",
                "split": "test",
                "layer": 0,
                "head": 0,
                "edge_kind": "self_loop",
                "edge_count": 4,
                "attention_mean": 0.2,
                "attention_std": 0.05,
                "attention_min": 0.1,
                "attention_max": 0.3,
            },
        ]
    )
