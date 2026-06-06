import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_sprint6_loss_comparison as runner


def test_sprint6_runner_writes_output_contract_and_resolved_loss_provenance(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    graph_root = tmp_path / "graphs"
    graph_a_dir = graph_root / "graph_a_minimal_physical_target"
    graph_a_dir.mkdir(parents=True)
    (graph_a_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (graph_a_dir / "dummy.parquet").write_bytes(b"not-real-parquet-needed-for-hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_a_gcn", _fake_train_graph_a_gcn)
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint6_loss_comparison(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
        selected_run_ids=["S6R0_wbce", "S6R7_balanced_sampling"],
    )

    assert output_dir == tmp_path / "outputs" / "sprint6" / "loss_comparison"
    required = [
        output_dir / "sprint6_loss_comparison_results.csv",
        output_dir / "sprint6_loss_comparison_report.md",
        output_dir / "sprint6_loss_comparison_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics_sprint6" / "imbalance_threshold_metrics.csv",
        output_dir / "figures_sprint6" / "imbalance_negative_retrieval_summary.png",
        output_dir / "runs" / "batch_S6R0_wbce" / "resolved_config.yaml",
        output_dir / "runs" / "batch_S6R0_wbce" / "runtime.json",
        output_dir / "runs" / "batch_S6R0_wbce" / "training_history.csv",
        output_dir / "runs" / "batch_S6R0_wbce" / "metrics.csv",
        output_dir / "runs" / "batch_S6R0_wbce" / "model.pt",
        output_dir / "runs" / "batch_S6R7_balanced_sampling" / "resolved_config.yaml",
        output_dir / "runs" / "batch_S6R7_balanced_sampling" / "model.pt",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "sprint6_loss_comparison_results.csv")
    assert results["run_id"].is_unique
    assert set(results["predeclared_run_id"]) == {"S6R0_wbce", "S6R7_balanced_sampling"}
    assert "run_id" in pd.read_csv(output_dir / "diagnostics_sprint6" / "sprint6_loss_comparison_predictions.csv").columns

    wbce_config = yaml.safe_load(
        (output_dir / "runs" / "batch_S6R0_wbce" / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    sampling_config = yaml.safe_load(
        (output_dir / "runs" / "batch_S6R7_balanced_sampling" / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert wbce_config["training"]["loss"] == "weighted_bce"
    assert wbce_config["training"]["loss_params"] == {"pos_weight": "auto"}
    assert "sampling" not in wbce_config["training"]
    assert sampling_config["training"]["loss"] == "bce_unweighted"
    assert sampling_config["training"]["loss_params"] == {"pos_weight": 1.0}
    assert sampling_config["training"]["sampling"]["strategy"] == "balanced_subsample"

    manifest = json.loads((output_dir / "sprint6_loss_comparison_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_runs_executed"] == []
    assert manifest["runs"][0]["run_id"] == "batch_S6R0_wbce"


def test_sprint6_runner_rejects_optional_runs_without_explicit_opt_in(tmp_path) -> None:
    config_path = _write_config(tmp_path)
    config = runner.load_yaml(config_path)

    with pytest.raises(ValueError, match="requires --include-optional-runs"):
        runner._selected_run_specs(config, ["S6R8_classbalanced_cui"], include_optional_runs=False)


def _write_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sweeps" / "sprint6_loss_comparison.yaml")
    source["data"]["graph_artifact_dir"] = "graphs"
    source["outputs"]["output_dir"] = "outputs/sprint6/loss_comparison"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint6_loss_comparison.yaml"
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
    loss_offset = {
        "weighted_bce": 0.0,
        "bce_unweighted": -0.01,
    }.get(config.loss, -0.02)
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint6",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": "gcn_graph_a_sprint6",
                "feature_set": "S5F2_energy",
                "graph_schema": "graph_a_minimal_physical_target",
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": "zero_type_feature",
                "loss": config.loss,
                "threshold": 0.5,
                "threshold_selection_split": "validation",
                "checkpoint_selection_split": "validation",
                "baseline_reference": "xgboost_unweighted / F4",
                "baseline_test_auprc": 0.992522,
                "baseline_test_auroc": 0.938416,
                "baseline_test_mcc": 0.345198,
                "test_positive_rate": 0.900705,
                "test_auprc": 0.970000 + loss_offset,
                "test_auroc": 0.810000,
                "test_f1": 0.910000,
                "test_macro_f1": 0.700000,
                "test_mcc": 0.400000,
                "test_specificity": 0.500000,
                "test_sensitivity": 0.950000,
                "test_tn": 1,
                "test_fp": 1,
                "test_fn": 1,
                "test_tp": 5,
            }
        ]
    )
    predictions = _fake_predictions()
    history = pd.DataFrame(
        [
            {
                "model_name": "gcn_graph_a_sprint6",
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


def _fake_predictions() -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate(
            [(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.6), (1, 0.8), (1, 0.4)]
        ):
            rows.append(
                {
                    "model_name": "gcn_graph_a_sprint6",
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
