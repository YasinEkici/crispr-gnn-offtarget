import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_sprint7b_gatv2_topology as runner


def test_sprint7b_runner_writes_output_contract(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    for schema in ["graph_b_guide_similarity_control", "graph_c_context_observation"]:
        artifact_dir = tmp_path / "graphs" / ("sprint7b" if "graph_b" in schema else "sprint5b") / schema
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (artifact_dir / "dummy.parquet").write_bytes(b"hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_b_gcn", _fake_train)
    monkeypatch.setattr(runner, "train_graph_c_gcn", _fake_train)
    monkeypatch.setattr(runner, "collect_graph_attention_summary", lambda *args, **kwargs: _fake_attention())
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7b_gatv2_topology(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
    )

    required = [
        output_dir / "gatv2_topology_comparison.csv",
        output_dir / "gatv2_topology_report.md",
        output_dir / "gatv2_topology_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "gatv2_topology_threshold_metrics.csv",
        output_dir / "diagnostics" / "gatv2_topology_attention_summary.csv",
        output_dir / "figures" / "gatv2_topology_auprc_comparison.png",
        output_dir / "figures" / "gatv2_topology_attention_by_edge_kind.png",
        output_dir / "runs" / "batch_S7B_R1_graph_b_gcn_s5f2" / "metrics.csv",
        output_dir / "runs" / "batch_S7B_R2_graph_b_gatv2_s5f2" / "attention_summary.csv",
        output_dir / "runs" / "batch_S7B_R3_graph_c_gatv2_s5f2" / "resolved_config.yaml",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "gatv2_topology_comparison.csv")
    assert set(results["predeclared_run_id"]) == {
        "S7B_REF_XGB_F4",
        "S7B_REF_GA_GCN",
        "S7B_REF_GA_GATV2",
        "S7B_REF_GC_GCN",
        "S7B_R1_graph_b_gcn_s5f2",
        "S7B_R2_graph_b_gatv2_s5f2",
        "S7B_R3_graph_c_gatv2_s5f2",
    }
    manifest = json.loads((output_dir / "gatv2_topology_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["headline_run_ids"] == list(runner.HEADLINE_RUN_IDS)
    assert len(manifest["runs"]) == 7

    resolved = yaml.safe_load(
        (output_dir / "runs" / "batch_S7B_R3_graph_c_gatv2_s5f2" / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert resolved["graph"]["schema"] == "graph_c_context_observation"
    assert resolved["model"]["architecture"] == "gatv2"
    assert resolved["training"]["loss"] == "weighted_bce"


def _write_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(Path(__file__).resolve().parents[1] / "configs" / "sweeps" / "sprint7b_gatv2_topology.yaml")
    source["data"]["graph_b_artifact_dir"] = "graphs/sprint7b"
    source["data"]["graph_c_artifact_dir"] = "graphs/sprint5b"
    source["outputs"]["output_dir"] = "outputs/sprint7b"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint7b_gatv2_topology.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
    return path


class _FakeMaterialized:
    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        if graph_name == "graph_b_guide_similarity_control":
            self.manifest = {
                "graph_name": graph_name,
                "split_id": "sprint2_main_seed42",
                "label_scheme": "scheme_a",
                "feature_tables": {"S5F2_energy": 268},
                "metadata": {"visibility_policy": "strict_inductive_primary"},
            }
        else:
            self.manifest = {
                "graph_name": graph_name,
                "split_id": "sprint2_main_seed42",
                "label_scheme": "scheme_a",
                "feature_tables": {"S5F2_energy": 268, "target_observation_features": 8},
                "metadata": {"visibility_policy": "strict_inductive_primary"},
            }


class _FakeLoader:
    def __init__(self, _path: Path) -> None:
        self.path = _path

    def load(self, graph_name: str) -> _FakeMaterialized:
        return _FakeMaterialized(graph_name)


def _fake_loader_factory(path: Path) -> _FakeLoader:
    return _FakeLoader(path)


def _fake_train(_materialized, config, *, checkpoint_path: Path):
    checkpoint_path.write_bytes(b"fake-checkpoint")
    is_graph_c = config.graph_schema == "graph_c_context_observation"
    architecture_offset = 0.002 if config.architecture == "gatv2" else 0.0
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint7b",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": config.model_name,
                "architecture": config.architecture,
                "feature_set": "S5F2_energy",
                "graph_schema": config.graph_schema,
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": config.target_node_representation,
                "loss": "weighted_bce",
                "test_positive_rate": 0.900705,
                "test_auprc": 0.970 + architecture_offset + (0.001 if is_graph_c else 0.0),
                "test_auroc": 0.820,
                "test_f1": 0.910,
                "test_macro_f1": 0.700,
                "test_mcc": 0.300,
                "test_specificity": 0.200,
                "test_sensitivity": 0.990,
                "test_tn": 34,
                "test_fp": 135,
                "test_fn": 12,
                "test_tp": 1521,
                "edge_aware_attention": config.edge_aware_attention if config.architecture == "gatv2" else None,
                "attention_heads": config.attention_heads if config.architecture == "gatv2" else None,
                "attention_concat": config.attention_concat if config.architecture == "gatv2" else None,
                "attention_dropout": config.attention_dropout,
                "self_loop_edge_fill": config.self_loop_edge_fill if config.architecture == "gatv2" else None,
            }
        ]
    )
    predictions = _fake_predictions(config)
    history = pd.DataFrame(
        [
            {
                "model_name": config.model_name,
                "graph_schema": config.graph_schema,
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


def _fake_predictions(config) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        for index, (label, score) in enumerate([(1, 0.9), (0, 0.2), (1, 0.7), (0, 0.6)]):
            rows.append(
                {
                    "model_name": config.model_name,
                    "architecture": config.architecture,
                    "graph_schema": config.graph_schema,
                    "feature_set": "S5F2_energy",
                    "split": split,
                    "row_index": index,
                    "grna_target_id": f"guide_{index % 2}",
                    "genome": "hg19",
                    "label": label,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def _fake_attention() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "gatv2",
                "architecture": "gatv2",
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
                "model_name": "gatv2",
                "architecture": "gatv2",
                "split": "test",
                "layer": 0,
                "head": 0,
                "edge_kind": "context_similar_to",
                "edge_count": 3,
                "attention_mean": 0.2,
                "attention_std": 0.05,
                "attention_min": 0.1,
                "attention_max": 0.3,
            },
        ]
    )
