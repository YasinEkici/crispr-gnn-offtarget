from pathlib import Path

import pytest

from crispr_gnn.graph.pyg_dataset import validate_gcn_headline_config
from crispr_gnn.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mak2022_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "data" / "mak2022.yaml")
    assert config["dataset"]["name"] == "mak2022"
    assert config["split_rules"]["test_measured_only"] is True


def test_gcn_minimal_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_minimal.yaml")
    assert config["experiment_name"] == "gcn_minimal"
    assert config["task"] == "sprint4_gcn"
    assert config["model"]["name"] == "gcn_graph_a"
    assert config["graph"]["schema"] == "graph_a_minimal_physical_target"
    assert config["data"]["split_id"] == "sprint2_main_seed42"
    assert config["model"]["target_node_representation"] == "zero_type_feature"
    assert config["features"]["edge_feature_sets"] == ["s1_pair", "f1"]
    validate_gcn_headline_config(config)


def test_gcn_graph_c_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_graph_c.yaml")
    assert config["experiment_name"] == "gcn_graph_c"
    assert config["task"] == "sprint4_gcn_graph_c"
    assert config["model"]["name"] == "gcn_graph_c"
    assert config["graph"]["schema"] == "graph_c_context_observation"
    assert config["graph"]["context_placement"] == "target_observation_node"
    assert config["graph"]["target_semantics"] == "observation_level_context"
    assert config["data"]["split_id"] == "sprint2_main_seed42"
    assert config["model"]["target_node_representation"] == "target_observation_context_encoder"
    assert config["features"]["edge_feature_sets"] == ["candidate_pair_features"]
    assert config["evaluation"]["graph_c_interpretation"] == "topology_and_target_semantics_not_topology_only"
    validate_gcn_headline_config(config)


def test_gcn_headline_config_rejects_debug_or_random_edge_rules() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_minimal.yaml")
    config["data"]["split_id"] = "debug"
    with pytest.raises(ValueError, match="locked guide-level split"):
        validate_gcn_headline_config(config)

    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_minimal.yaml")
    config["evaluation"]["protocol"] = "random_edge"
    with pytest.raises(ValueError, match="headline guide-level evaluation"):
        validate_gcn_headline_config(config)

    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_minimal.yaml")
    config["evaluation"]["notes"] = "technical_debug_only"
    with pytest.raises(ValueError, match="Debug/random-edge"):
        validate_gcn_headline_config(config)


def test_sequence_cnn_bilstm_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "sequence_cnn_bilstm.yaml")
    assert config["experiment_name"] == "sequence_cnn_bilstm"
    assert config["task"] == "sprint2_sequence"
    assert config["sequence"]["max_length"] == 23


def test_sequence_cnn_late_fusion_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "sequence_cnn_late_fusion.yaml")
    assert config["experiment_name"] == "sequence_cnn_late_fusion"
    assert config["task"] == "sprint2_sequence_late_fusion"
    assert config["tabular_feature_sets"] == ["F3", "F4"]


def test_graph_schema_ablation_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "sweeps" / "graph_schema_ablation.yaml")
    assert config["visibility_policy"] == "strict_inductive_primary"
    assert config["graph_a"]["context_placement"] == "candidate_pair_edge"
    assert config["graph_b"]["role"] == "bounded_secondary_control"
    assert config["graph_c"]["context_placement"] == "target_observation_node"
