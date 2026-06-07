import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.features.target_context import (
    target_context_family_counts,
    target_context_mask_indices,
    validate_target_context_feature_names,
)
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from scripts import analyze_sprint7e_target_context_features as profiler
from scripts import run_sprint7e_target_context_subgroup_ablation as runner


def test_target_context_family_mapping_counts_and_aliases() -> None:
    names = _target_context_feature_names()

    validate_target_context_feature_names(names)
    counts = target_context_family_counts(names)

    assert counts["target_sequence_one_hot"] == 115
    assert counts["experimental_epigenetic"] == 6
    assert counts["computed_nucleosome_aggregates"] == 78
    assert counts["computed_nucleosome_missingness"] == 13
    assert counts["unknown"] == 0
    assert len(target_context_mask_indices(names, ["all_nonsequence_context"])) == 97


def test_sprint7e_profiling_writes_declared_outputs(tmp_path) -> None:
    graph_c_dir = _write_graph_c_profile_artifact(tmp_path)
    output_dir = tmp_path / "outputs" / "sprint7e" / "context_feature_profiling"

    profiler.run_sprint7e_target_context_feature_profiling(
        graph_c_dir=graph_c_dir,
        output_dir=output_dir,
        write_figures=True,
    )

    required = [
        output_dir / "sprint7e_context_feature_family_map.csv",
        output_dir / "sprint7e_context_feature_group_summary.csv",
        output_dir / "sprint7e_context_feature_distribution_by_split_label.csv",
        output_dir / "sprint7e_experimental_epigenetic_feature_distribution_by_split_label.csv",
        output_dir / "sprint7e_experimental_epigenetic_feature_smd_by_split.csv",
        output_dir / "sprint7e_context_feature_profile_report.md",
        output_dir / "sprint7e_context_feature_profile_manifest.json",
        output_dir / "figures" / "sprint7e_context_feature_group_missingness.png",
        output_dir / "figures" / "sprint7e_context_feature_group_distribution.png",
        output_dir / "figures" / "sprint7e_experimental_epigenetic_smd_by_split.png",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    family_map = pd.read_csv(output_dir / "sprint7e_context_feature_family_map.csv")
    assert len(family_map) == 212
    assert family_map["target_context_family"].value_counts().to_dict()["target_sequence_one_hot"] == 115
    experimental_distribution = pd.read_csv(
        output_dir / "sprint7e_experimental_epigenetic_feature_distribution_by_split_label.csv"
    )
    experimental_smd = pd.read_csv(output_dir / "sprint7e_experimental_epigenetic_feature_smd_by_split.csv")
    assert len(experimental_distribution) == 36
    assert len(experimental_smd) == 18
    assert experimental_distribution["source_feature_name"].nunique() == 6
    assert experimental_smd["source_feature_name"].nunique() == 6
    manifest = json.loads((output_dir / "sprint7e_context_feature_profile_manifest.json").read_text())
    assert manifest["no_training_performed"] is True
    assert manifest["no_test_performance_selection"] is True
    assert manifest["canonical_slice2_run_ids"] == list(runner.HEADLINE_RUN_IDS)


def test_sprint7e_runner_writes_output_contract(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(runner, "_runtime_info", lambda device: {"device": device})

    output_dir = runner.run_sprint7e_target_context_subgroup_ablation(
        config_path=config_path,
        batch_id="batch",
        max_epochs=1,
    )

    required = [
        output_dir / "target_context_subgroup_ablation.csv",
        output_dir / "target_context_subgroup_ablation_report.md",
        output_dir / "target_context_subgroup_ablation_run_manifest.json",
        output_dir / "graph_artifact_provenance.json",
        output_dir / "diagnostics" / "target_context_subgroup_threshold_metrics.csv",
        output_dir / "diagnostics" / "target_context_subgroup_mask_audit.csv",
        output_dir / "diagnostics" / "target_context_subgroup_attention_summary.csv",
        output_dir / "figures" / "target_context_subgroup_auprc_comparison.png",
        output_dir / "figures" / "target_context_subgroup_attention_by_edge_kind.png",
        output_dir / "runs" / "batch_S7E_R1_mask_target_sequence" / "target_feature_mask_audit.csv",
        output_dir / "runs" / "batch_S7E_R3_mask_computed_nucleosome_aggregates" / "resolved_config.yaml",
        output_dir / "runs" / "batch_S7E_R5_mask_all_nonsequence_context" / "metrics.csv",
    ]
    for path in required:
        assert path.exists(), path
        assert path.stat().st_size > 0, path

    results = pd.read_csv(output_dir / "target_context_subgroup_ablation.csv")
    assert set(results["predeclared_run_id"]) == {
        *runner.REFERENCE_RUN_IDS,
        *runner.HEADLINE_RUN_IDS,
    }
    manifest = json.loads((output_dir / "target_context_subgroup_ablation_run_manifest.json").read_text())
    assert manifest["headline_run_ids"] == list(runner.HEADLINE_RUN_IDS)
    assert len(manifest["runs"]) == len(runner.REFERENCE_RUN_IDS) + len(runner.HEADLINE_RUN_IDS)

    resolved = yaml.safe_load(
        (output_dir / "runs" / "batch_S7E_R3_mask_computed_nucleosome_aggregates" / "resolved_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert resolved["graph"]["schema"] == "graph_c_context_observation"
    assert resolved["model"]["architecture"] == "gatv2"
    assert resolved["model"]["attention"]["drop_context_similarity_edges"] is True
    assert resolved["model"]["attention"]["edge_blind_candidate_attention"] is False
    assert resolved["model"]["target_observation_mask_families"] == ["computed_nucleosome_aggregates"]
    assert resolved["training"]["loss"] == "weighted_bce"
    assert resolved["training"]["loss_params"] == {"pos_weight": "auto"}

    audit = pd.read_csv(output_dir / "diagnostics" / "target_context_subgroup_mask_audit.csv")
    target_sequence = audit.loc[audit["predeclared_run_id"] == "S7E_R1_mask_target_sequence"]
    assert set(target_sequence["masked_column_count"]) == {115}
    assert set(target_sequence["masked_feature_abs_sum_after_mask"]) == {0.0}
    assert set(target_sequence["context_edges_used"]) == {0}
    assert target_sequence["candidate_attention_attr_abs_sum"].min() > 0.0
    assert target_sequence["classifier_candidate_edge_attr_abs_sum"].min() > 0.0
    nonsequence = audit.loc[audit["predeclared_run_id"] == "S7E_R5_mask_all_nonsequence_context"]
    assert set(nonsequence["masked_column_count"]) == {97}


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


def _write_graph_c_profile_artifact(tmp_path: Path) -> Path:
    graph_c_dir = tmp_path / "graph_c_context_observation"
    graph_c_dir.mkdir()
    names = _target_context_feature_names()
    manifest = {
        "graph_name": "graph_c_context_observation",
        "label_scheme": "scheme_a",
        "split_id": "sprint2_main_seed42",
        "feature_tables": {"target_observation_features": 212, "S5F2_energy": 268},
        "metadata": {"visibility_policy": "strict_inductive_primary"},
    }
    (graph_c_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    node_ids = [str(index) for index in range(6)]
    splits = ["train", "train", "val", "val", "test", "test"]
    pd.DataFrame({"node_id": node_ids, "split": splits}).to_parquet(
        graph_c_dir / "nodes_target_observation.parquet",
        index=False,
    )
    feature_rows = []
    for index, node_id in enumerate(node_ids):
        row = {"record_id": node_id}
        for column_index, name in enumerate(names):
            row[name] = float((index + 1) * (column_index % 3))
        feature_rows.append(row)
    pd.DataFrame(feature_rows).to_parquet(
        graph_c_dir / "features_target_observation_features.parquet",
        index=False,
    )
    pd.DataFrame(
        {
            "edge_id": node_ids,
            "label": [1, 0, 1, 0, 1, 0],
            "split": splits,
            "measured": [1] * 6,
            "experiment_id": [1] * 6,
        }
    ).to_parquet(graph_c_dir / "relation_candidate_pair.parquet", index=False)
    return graph_c_dir


def _write_runner_config(tmp_path: Path) -> Path:
    source = runner.load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "sweeps"
        / "sprint7e_target_context_subgroup_ablation.yaml"
    )
    source["data"]["graph_c_artifact_dir"] = "graphs/sprint5b"
    source["outputs"]["output_dir"] = "outputs/sprint7e"
    source["training"]["device"] = "cpu"
    source["training"]["use_compile"] = False
    source["training"]["use_amp"] = False
    path = tmp_path / "sprint7e_target_context_subgroup_ablation.yaml"
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

    def view(self, split: str) -> HeteroData:
        data = HeteroData()
        data.graph_name = "graph_c_context_observation"
        data["sgRNA"].x = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        data["target_observation"].x = torch.arange(636, dtype=torch.float32).reshape(3, 212) + 1.0
        data["target_observation"].feature_names = _target_context_feature_names()
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
    offsets = {
        "target_sequence_one_hot": 0.001,
        "experimental_epigenetic": 0.002,
        "computed_nucleosome_aggregates": 0.003,
        "computed_nucleosome_missingness": 0.004,
        "all_nonsequence_context": 0.005,
    }
    family = ",".join(config.target_observation_mask_families)
    offset = offsets[family]
    results = pd.DataFrame(
        [
            {
                "sprint": "sprint7e",
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
                "target_observation_mask_families": family,
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
                "edge_kind": "self_loop",
                "edge_count": 2,
                "attention_mean": 0.3,
                "attention_std": 0.05,
                "attention_min": 0.2,
                "attention_max": 0.4,
            },
        ]
    )
