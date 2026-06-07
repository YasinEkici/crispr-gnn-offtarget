import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import write_results_table
from crispr_gnn.utils.config import load_yaml
from crispr_gnn.training.gcn import (
    BASELINE_TEST_AUPRC,
    GRAPH_C_TARGET_REPRESENTATION_POLICY,
    GCNRunConfig,
    gcn_run_config_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gcn_config_records_locked_contract_and_baseline_reference() -> None:
    config = _base_config()

    run_config = gcn_run_config_from_mapping(config)

    assert run_config.graph_schema == "graph_a_minimal_physical_target"
    assert run_config.label_scheme == "scheme_a"
    assert run_config.split_id == "sprint2_main_seed42"
    assert run_config.visibility_policy == "strict_inductive_primary"
    assert run_config.loss == "weighted_bce"
    assert run_config.target_node_representation == "zero_type_feature"
    assert run_config.feature_set == "S1_pair+F1"
    assert BASELINE_TEST_AUPRC == pytest.approx(0.992522)


def test_gcn_config_records_graph_c_observation_context_contract() -> None:
    config = _base_config()
    config["graph"]["schema"] = "graph_c_context_observation"
    config["model"]["name"] = "gcn_graph_c"
    config["model"]["target_node_representation"] = "target_observation_context_encoder"
    config["features"]["edge_feature_sets"] = ["candidate_pair_features"]

    run_config = gcn_run_config_from_mapping(config)

    assert run_config.graph_schema == "graph_c_context_observation"
    assert run_config.model_name == "gcn_graph_c"
    assert run_config.edge_feature_sets == ("candidate_pair_features",)
    assert run_config.feature_set == "CandidatePair"
    assert run_config.target_node_representation == GRAPH_C_TARGET_REPRESENTATION_POLICY


def test_gcn_config_records_sprint5b_graph_c_energy_sensitivity_contract() -> None:
    config = load_yaml(ROOT / "configs" / "sweeps" / "sprint5b_graph_c_energy_sensitivity.yaml")

    run_config = gcn_run_config_from_mapping(config)

    assert run_config.sprint == "sprint5b"
    assert run_config.graph_schema == "graph_c_context_observation"
    assert run_config.model_name == "gcn_graph_c_sprint5b_energy"
    assert run_config.edge_feature_sets == ("s5f2_energy",)
    assert run_config.feature_set == "GraphCContext+S5F2_energy"
    assert run_config.target_node_representation == GRAPH_C_TARGET_REPRESENTATION_POLICY
    assert run_config.split_id == "sprint2_main_seed42"
    assert run_config.loss == "weighted_bce"


def test_gcn_config_rejects_unsupported_schema() -> None:
    config = _base_config()
    config["graph"]["schema"] = "graph_x_unknown"

    with pytest.raises(ValueError, match="Graph A, Graph B, and Graph C"):
        gcn_run_config_from_mapping(config)


def test_gcn_config_records_graph_b_bounded_control_contract() -> None:
    config = _base_config()
    config["graph"]["schema"] = "graph_b_guide_similarity_control"
    config["model"]["name"] = "gcn_graph_b"
    config["model"]["target_node_representation"] = "zero_type_feature"
    config["features"]["edge_feature_sets"] = ["s1_pair", "f1"]

    run_config = gcn_run_config_from_mapping(config)

    assert run_config.graph_schema == "graph_b_guide_similarity_control"
    assert run_config.model_name == "gcn_graph_b"
    assert run_config.edge_feature_sets == ("s1_pair", "f1")
    assert run_config.target_node_representation == "zero_type_feature"
    assert run_config.split_id == "sprint2_main_seed42"
    assert run_config.label_scheme == "scheme_a"
    assert run_config.visibility_policy == "strict_inductive_primary"


def test_gcn_config_rejects_non_featureless_targets_for_graph_b() -> None:
    config = _base_config()
    config["graph"]["schema"] = "graph_b_guide_similarity_control"
    config["model"]["target_node_representation"] = "learned_per_target_id"
    config["features"]["edge_feature_sets"] = ["s1_pair", "f1"]

    with pytest.raises(ValueError, match="zero/type"):
        gcn_run_config_from_mapping(config)


def test_gcn_config_rejects_unapproved_target_representation_or_loss() -> None:
    config = _base_config()
    config["model"]["target_node_representation"] = "learned_per_target_id"
    with pytest.raises(ValueError, match="zero/type"):
        gcn_run_config_from_mapping(config)

    # Sprint 6 permits the predeclared loss set (see docs/DECISIONS.md 2026-06-06);
    # only losses outside that set are rejected. "focal_loss" is not a registry key
    # ("focal" is), so it must still be rejected as unsupported.
    config = _base_config()
    config["training"]["loss"] = "focal_loss"
    with pytest.raises(ValueError, match="Unsupported GCN loss"):
        gcn_run_config_from_mapping(config)

    config = _base_config()
    config["graph"]["schema"] = "graph_c_context_observation"
    config["model"]["target_node_representation"] = "zero_type_feature"
    config["features"]["edge_feature_sets"] = ["candidate_pair_features"]
    with pytest.raises(ValueError, match="observation context encoder"):
        gcn_run_config_from_mapping(config)


def test_gcn_run_config_defaults_to_validation_only_policy_fields() -> None:
    config = GCNRunConfig(sprint="sprint4", split_id="sprint2_main_seed42", seed=42)

    assert config.loss == "weighted_bce"
    assert config.target_node_representation == "zero_type_feature"
    assert config.edge_feature_sets == ("s1_pair", "f1")


def test_gcn_result_upsert_identity_includes_graph_schema(tmp_path) -> None:
    path = tmp_path / "gcn_results.csv"
    graph_a = _result_row("graph_a_minimal_physical_target", test_auprc=0.7)
    graph_b = _result_row("graph_b_guide_similarity_control", test_auprc=0.72)
    graph_c = _result_row("graph_c_context_observation", test_auprc=0.8)

    write_results_table(graph_a, path)
    write_results_table(graph_b, path)
    write_results_table(graph_c, path)

    saved = pd.read_csv(path)
    assert set(saved["graph_schema"]) == {
        "graph_a_minimal_physical_target",
        "graph_b_guide_similarity_control",
        "graph_c_context_observation",
    }
    assert saved.shape[0] == 3


def _base_config() -> dict[str, object]:
    return {
        "sprint": "sprint4",
        "seed": 42,
        "data": {
            "label_scheme": "scheme_a",
            "split_id": "sprint2_main_seed42",
        },
        "graph": {
            "schema": "graph_a_minimal_physical_target",
            "visibility_policy": "strict_inductive_primary",
        },
        "features": {
            "edge_feature_sets": ["s1_pair", "f1"],
        },
        "model": {
            "name": "gcn_graph_a",
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "target_node_representation": "zero_type_feature",
        },
        "training": {
            "loss": "weighted_bce",
            "max_epochs": 2,
            "min_epochs": 1,
            "patience": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "device": "cpu",
        },
    }


def _result_row(graph_schema: str, *, test_auprc: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sprint": "sprint4",
                "label_scheme": "scheme_a",
                "split_id": "sprint2_main_seed42",
                "seed": 42,
                "training_regime": "measured_only",
                "model_name": "gcn_graph_a",
                "feature_set": "S1_pair+F1",
                "graph_schema": graph_schema,
                "test_auprc": test_auprc,
            }
        ]
    )
