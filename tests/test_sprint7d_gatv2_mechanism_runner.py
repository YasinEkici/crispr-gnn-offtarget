import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from scripts import run_sprint7d_graphc_gatv2_mechanism_ablation as runner


def test_sprint7d_runner_writes_output_contract(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    artifact_dir = tmp_path / "graphs" / "sprint5b" / "graph_c_context_observation"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "dummy.parquet").write_bytes(b"hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_c_gcn", _fake_train)
    monkeypatch.setattr(runner, "collect_graph_attention_summary", lambda *args, **kwargs: _fake_attention())
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7d_graphc_gatv2_mechanism_ablation(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
    )

    required = [
        output_dir / "graphc_gatv2_mechanism_ablation.csv",
        output_dir / "graphc_gatv2_mechanism_ablation_report.md",
        output_dir / "graphc_gatv2_mechanism_ablation_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "graphc_gatv2_mechanism_threshold_metrics.csv",
        output_dir / "diagnostics" / "graphc_gatv2_component_ablation_audit.csv",
        output_dir / "diagnostics" / "graphc_gatv2_mechanism_attention_summary.csv",
        output_dir / "figures" / "graphc_gatv2_mechanism_auprc_comparison.png",
        output_dir / "figures" / "graphc_gatv2_mechanism_attention_by_edge_kind.png",
        output_dir / "runs" / "batch_S7D_R1_no_context_edges" / "component_audit.csv",
        output_dir / "runs" / "batch_S7D_R2_edge_blind_attention" / "resolved_config.yaml",
        output_dir / "runs" / "batch_S7D_R3_mask_target_context_features" / "metrics.csv",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "graphc_gatv2_mechanism_ablation.csv")
    assert set(results["predeclared_run_id"]) == {
        "S7D_REF_XGB_F4",
        "S7D_REF_GRAPH_A_GCN",
        "S7D_REF_GRAPH_A_GATV2",
        "S7D_REF_GRAPH_C_GCN",
        "S7D_REF_FULL_GRAPH_C_GATV2",
        "S7D_R1_no_context_edges",
        "S7D_R2_edge_blind_attention",
        "S7D_R3_mask_target_context_features",
    }
    manifest = json.loads((output_dir / "graphc_gatv2_mechanism_ablation_run_manifest.json").read_text())
    assert manifest["headline_run_ids"] == list(runner.HEADLINE_RUN_IDS)
    assert len(manifest["runs"]) == 8

    resolved = yaml.safe_load(
        (output_dir / "runs" / "batch_S7D_R2_edge_blind_attention" / "resolved_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert resolved["graph"]["schema"] == "graph_c_context_observation"
    assert resolved["model"]["architecture"] == "gatv2"
    assert resolved["model"]["attention"]["edge_blind_candidate_attention"] is True
    assert resolved["model"]["attention"]["edge_aware"] is True
    assert resolved["training"]["loss"] == "weighted_bce"
    assert resolved["training"]["loss_params"] == {"pos_weight": "auto"}

    audit = pd.read_csv(output_dir / "diagnostics" / "graphc_gatv2_component_ablation_audit.csv")
    no_context = audit.loc[audit["predeclared_run_id"] == "S7D_R1_no_context_edges"]
    assert set(no_context["context_edges_used"]) == {0}
    edge_blind = audit.loc[audit["predeclared_run_id"] == "S7D_R2_edge_blind_attention"]
    assert set(edge_blind["candidate_attention_attr_abs_sum"]) == {0.0}
    assert edge_blind["classifier_candidate_edge_attr_abs_sum"].min() > 0.0
    masked = audit.loc[audit["predeclared_run_id"] == "S7D_R3_mask_target_context_features"]
    assert set(masked["target_feature_abs_sum_after_mask"]) == {0.0}


def _write_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "sweeps"
        / "sprint7d_graphc_gatv2_mechanism_ablation.yaml"
    )
    source["data"]["graph_c_artifact_dir"] = "graphs/sprint5b"
    source["outputs"]["output_dir"] = "outputs/sprint7d"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint7d_graphc_gatv2_mechanism_ablation.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
    return path


class _FakeMaterialized:
    graph_name = "graph_c_context_observation"
    manifest = {
        "graph_name": "graph_c_context_observation",
        "split_id": "sprint2_main_seed42",
        "label_scheme": "scheme_a",
        "feature_tables": {"S5F2_energy": 268, "target_observation_features": 8},
        "metadata": {"visibility_policy": "strict_inductive_primary"},
    }

    def view(self, split: str) -> HeteroData:
        data = HeteroData()
        data.graph_name = "graph_c_context_observation"
        data["sgRNA"].x = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        data["target_observation"].x = torch.tensor(
            [[1.0, 0.2, 0.3], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            dtype=torch.float32,
        )
        edge_store = data[GRAPH_C_EDGE_TYPE]
        edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
        edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        edge_store.supervision_mask = torch.tensor([True, True, True])
        edge_store.edge_attr_s5f2_energy = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=torch.float32,
        )
        data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        return data


class _FakeLoader:
    def __init__(self, _path: Path) -> None:
        self.path = _path

    def load(self, graph_name: str) -> _FakeMaterialized:
        assert graph_name == "graph_c_context_observation"
        return _FakeMaterialized()


def _fake_loader_factory(path: Path) -> _FakeLoader:
    return _FakeLoader(path)


def _fake_train(_materialized, config, *, checkpoint_path: Path):
    checkpoint_path.write_bytes(b"fake-checkpoint")
    offset = {
        "gatv2_graph_c_sprint7d_no_context_edges": 0.001,
        "gatv2_graph_c_sprint7d_edge_blind_attention": 0.002,
        "gatv2_graph_c_sprint7d_mask_target_context_features": 0.003,
    }[config.model_name]
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint7d",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": config.model_name,
                "architecture": "gatv2",
                "feature_set": "S5F2_energy",
                "graph_schema": "graph_c_context_observation",
                "visibility_policy": "strict_inductive_primary",
                "target_node_representation": "target_observation_context_encoder",
                "loss": "weighted_bce",
                "test_positive_rate": 0.900705,
                "test_auprc": 0.960 + offset,
                "test_auroc": 0.820,
                "test_f1": 0.910,
                "test_macro_f1": 0.700,
                "test_mcc": 0.300 + offset,
                "test_specificity": 0.200,
                "test_sensitivity": 0.990,
                "test_tn": 34,
                "test_fp": 135,
                "test_fn": 12,
                "test_tp": 1521,
                "edge_aware_attention": True,
                "attention_heads": 4,
                "attention_concat": True,
                "attention_dropout": 0.2,
                "self_loop_edge_fill": 0.0,
                "drop_context_similarity_edges": config.drop_context_similarity_edges,
                "edge_blind_candidate_attention": config.edge_blind_candidate_attention,
                "mask_target_observation_features": config.mask_target_observation_features,
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
                    "genome": None,
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
