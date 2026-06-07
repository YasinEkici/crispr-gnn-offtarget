import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.models.gat import GraphCEdgeGATv2
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from crispr_gnn.models.target_context_encoder import (
    FAMILY_AWARE_BRANCH_DIMS,
    FamilyAwareTargetContextEncoder,
    target_context_family_indices,
)
from crispr_gnn.training.gcn import gcn_run_config_from_mapping
from scripts import run_sprint7f_target_context_encoder as runner


def test_family_aware_target_context_encoder_forward_and_family_indexes() -> None:
    names = _target_context_feature_names()
    indexes = target_context_family_indices(names)
    assert {family: len(values) for family, values in indexes.items()} == {
        "target_sequence_one_hot": 115,
        "experimental_epigenetic": 6,
        "computed_nucleosome_aggregates": 78,
        "computed_nucleosome_missingness": 13,
    }

    encoder = FamilyAwareTargetContextEncoder(
        input_dim=212,
        hidden_dim=16,
        feature_names=names,
        branch_dims=FAMILY_AWARE_BRANCH_DIMS,
        dropout=0.0,
        encoder_type="family_aware",
    )
    output = encoder(torch.ones((3, 212), dtype=torch.float32))

    assert output.shape == (3, 16)
    summary = encoder.activation_summary(torch.ones((3, 212), dtype=torch.float32))
    assert len(summary) == 4
    assert {row["target_context_family"] for row in summary} == set(indexes)


def test_graph_c_gatv2_accepts_family_aware_target_context_encoder() -> None:
    data = _fake_graph_c_view()
    model = GraphCEdgeGATv2(
        sgrna_input_dim=4,
        target_observation_input_dim=212,
        edge_input_dim=2,
        hidden_dim=16,
        num_layers=1,
        heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_context_similarity_edges=True,
        target_context_encoder_type="family_aware",
        target_context_feature_names=_target_context_feature_names(),
    )

    logits, attention_records = model(data, edge_feature_attrs=["edge_attr_s5f2_energy"], return_attention=True)

    assert logits.shape == (3,)
    assert model.target_context_encoder_type == "family_aware"
    assert attention_records[0]["alpha"].shape[1] == 2
    assert model.target_context_encoder_activation_summary(data["target_observation"].x)[0][
        "target_context_encoder_parameter_count"
    ] > 0


def test_sprint7f_config_parses_target_context_encoder_type() -> None:
    config = runner.load_yaml(Path("configs/sweeps/sprint7f_target_context_encoder.yaml"))
    run_mapping = runner._config_for_run(config, config["runs"][1], max_epochs=1)
    parsed = gcn_run_config_from_mapping(run_mapping)

    assert parsed.graph_schema == "graph_c_context_observation"
    assert parsed.architecture == "gatv2"
    assert parsed.target_context_encoder_type == "family_aware"
    assert parsed.drop_context_similarity_edges is True
    assert parsed.edge_aware_attention is True


def test_sprint7f_runner_writes_output_contract(tmp_path, monkeypatch) -> None:
    config_path = _write_runner_config(tmp_path)
    artifact_parent = tmp_path / "graphs" / "sprint5b"
    artifact_dir = artifact_parent / "graph_c_context_observation"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "dummy.parquet").write_bytes(b"hash-only")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "Sprint3HeteroDataLoader", _fake_loader_factory)
    monkeypatch.setattr(runner, "train_graph_c_gcn", _fake_train)
    monkeypatch.setattr(runner, "collect_graph_attention_summary", lambda *args, **kwargs: _fake_attention())
    monkeypatch.setattr(runner, "collect_target_context_encoder_summary", lambda *args, **kwargs: _fake_encoder_summary(args[1]))
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7f_target_context_encoder(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
    )

    required = [
        output_dir / "target_context_encoder_comparison.csv",
        output_dir / "target_context_encoder_report.md",
        output_dir / "target_context_encoder_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "target_context_encoder_threshold_metrics.csv",
        output_dir / "diagnostics" / "target_context_encoder_deltas.csv",
        output_dir / "diagnostics" / "target_context_encoder_activation_summary.csv",
        output_dir / "diagnostics" / "target_context_encoder_audit.csv",
        output_dir / "figures" / "target_context_encoder_auprc_comparison.png",
        output_dir / "figures" / "target_context_encoder_branch_activation_norms.png",
        output_dir / "runs" / "batch_S7F_R2_family_aware_context_encoder" / "target_context_encoder_summary.csv",
        output_dir / "runs" / "batch_S7F_R3_family_aware_experimental_emphasis" / "resolved_config.yaml",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "target_context_encoder_comparison.csv")
    assert set(results["predeclared_run_id"]) == {*runner.REFERENCE_RUN_IDS, *runner.HEADLINE_RUN_IDS}
    assert set(results.loc[results["predeclared_run_id"].isin(runner.HEADLINE_RUN_IDS), "target_context_encoder_type"]) == {
        "unified_deep",
        "family_aware",
        "family_aware_experimental_emphasis",
    }
    manifest = json.loads((output_dir / "target_context_encoder_run_manifest.json").read_text())
    assert manifest["headline_run_ids"] == list(runner.HEADLINE_RUN_IDS)
    assert "target_context_encoder_activation_summary_path" in manifest

    resolved = yaml.safe_load(
        (
            output_dir
            / "runs"
            / "batch_S7F_R3_family_aware_experimental_emphasis"
            / "resolved_config.yaml"
        ).read_text(encoding="utf-8")
    )
    assert resolved["model"]["target_context_encoder"]["type"] == "family_aware_experimental_emphasis"
    assert resolved["model"]["attention"]["drop_context_similarity_edges"] is True

    audit = pd.read_csv(output_dir / "diagnostics" / "target_context_encoder_audit.csv")
    train_val_test = audit.loc[audit["split"].isin(["train", "val", "test"])]
    assert set(train_val_test["context_edges_used"]) == {0}
    assert train_val_test["candidate_attention_attr_abs_sum"].min() > 0.0
    assert train_val_test["classifier_candidate_edge_attr_abs_sum"].min() > 0.0


def _target_context_feature_names() -> list[str]:
    names = [
        f"feature__target_pos_{position:02d}_{base}"
        for position in range(23)
        for base in ("A", "C", "G", "T", "N")
    ]
    names.extend(f"feature__{name}" for name in EXPERIMENTAL_EPIGENETIC_FEATURES)
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        for suffix in ("mean", "std", "min", "max", "center", "pam_proximal_mean"):
            names.append(f"feature__{feature}_{suffix}")
        names.append(f"feature__{feature}_missing")
    return names


def _write_runner_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(Path("configs/sweeps/sprint7f_target_context_encoder.yaml"))
    source["data"]["graph_c_artifact_dir"] = "graphs/sprint5b"
    source["outputs"]["output_dir"] = "outputs/sprint7f"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint7f_target_context_encoder.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=True), encoding="utf-8")
    return path


class _FakeMaterialized:
    graph_name = "graph_c_context_observation"
    manifest = {
        "graph_name": "graph_c_context_observation",
        "split_id": "sprint2_main_seed42",
        "label_scheme": "scheme_a",
        "feature_tables": {"S5F2_energy": 268, "target_observation_features": 212},
        "metadata": {"visibility_policy": "strict_inductive_primary"},
    }

    def view(self, _split: str) -> HeteroData:
        return _fake_graph_c_view()


class _FakeLoader:
    def __init__(self, _path: Path) -> None:
        self.path = _path

    def load(self, graph_name: str) -> _FakeMaterialized:
        assert graph_name == "graph_c_context_observation"
        return _FakeMaterialized()


def _fake_loader_factory(path: Path) -> _FakeLoader:
    return _FakeLoader(path)


def _fake_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_c_context_observation"
    data["sgRNA"].x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    data["target_observation"].x = torch.arange(636, dtype=torch.float32).reshape(3, 212) + 1.0
    data["target_observation"].feature_names = _target_context_feature_names()
    edge_store = data[GRAPH_C_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s5f2_energy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data


def _fake_train(_materialized, config, *, checkpoint_path: Path):
    checkpoint_path.write_bytes(b"fake-checkpoint")
    offsets = {
        "unified_deep": 0.001,
        "family_aware": 0.002,
        "family_aware_experimental_emphasis": 0.003,
    }
    offset = offsets[config.target_context_encoder_type]
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint7f",
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
                "target_context_encoder_type": config.target_context_encoder_type,
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
            }
        ]
    )


def _fake_encoder_summary(config) -> pd.DataFrame:
    rows = []
    families = (
        ["all_target_observation_features"]
        if config.target_context_encoder_type == "unified_deep"
        else [
            "target_sequence_one_hot",
            "experimental_epigenetic",
            "computed_nucleosome_aggregates",
            "computed_nucleosome_missingness",
        ]
    )
    for family in families:
        rows.append(
            {
                "model_name": config.model_name,
                "architecture": "gatv2",
                "graph_schema": "graph_c_context_observation",
                "split": "test",
                "target_context_encoder_type": config.target_context_encoder_type,
                "target_context_family": family,
                "input_columns": 212 if family == "all_target_observation_features" else 1,
                "branch_dim": 128 if family == "all_target_observation_features" else 16,
                "activation_mean": 0.1,
                "activation_std": 0.01,
                "activation_l2_mean": 1.0,
                "target_context_encoder_parameter_count": 123,
            }
        )
    return pd.DataFrame(rows)
