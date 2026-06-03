import json
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


def test_sprint5_feature_ablation_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "sweeps" / "sprint5_graph_a_feature_ablation.yaml")
    assert config["experiment_name"] == "sprint5_graph_a_feature_ablation"
    assert config["task"] == "sprint5_graph_a_feature_ablation"
    assert config["sprint"] == "sprint5"
    assert config["graph"]["schema"] == "graph_a_minimal_physical_target"
    assert config["data"]["graph_artifact_dir"] == "data/processed/graphs/sprint5"
    assert config["feature_sets"][0] == "S5F0_seq"
    assert "macro_f1" in config["metrics"]["secondary"]
    validate_gcn_headline_config(config)


def test_sprint5b_graph_c_energy_sensitivity_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "sweeps" / "sprint5b_graph_c_energy_sensitivity.yaml")
    assert config["experiment_name"] == "sprint5b_graph_c_energy_sensitivity"
    assert config["task"] == "sprint5b_graph_c_energy_sensitivity"
    assert config["sprint"] == "sprint5b"
    assert config["graph"]["schema"] == "graph_c_context_observation"
    assert config["graph"]["context_placement"] == "target_observation_node"
    assert config["graph"]["target_semantics"] == "observation_level_context"
    assert config["data"]["graph_artifact_dir"] == "data/processed/graphs/sprint5b"
    assert config["features"]["edge_feature_sets"] == ["s5f2_energy"]
    assert config["features"]["feature_set"] == "GraphCContext+S5F2_energy"
    assert str(config["evaluation"]["graph_c_interpretation"]).startswith("secondary_sensitivity")
    validate_gcn_headline_config(config)


def test_sprint5_colab_inline_imports_include_src_pythonpath() -> None:
    notebook_path = ROOT / "colab" / "sprint5_graph_a_feature_ablation_runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    step5 = next(cell for cell in notebook["cells"] if cell.get("id") == "step5-build-artifacts")
    source = "".join(step5["source"])

    assert "from crispr_gnn.graph.graph_schemas import GRAPH_A" in source
    assert "PYTHONPATH=src uv run python - <<'PY'" in source


def test_sprint5_colab_runner_does_not_delete_drive_outputs() -> None:
    notebook_path = ROOT / "colab" / "sprint5_graph_a_feature_ablation_runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }

    assert "rm -rf" not in "\n".join(sources.values())
    assert "Output already exists in Drive" in sources["step7-copy-outputs"]


def test_sprint5b_colab_runner_contract() -> None:
    notebook_path = ROOT / "colab" / "sprint5b_graph_c_energy_sensitivity_runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }

    assert "rm -rf" not in "\n".join(sources.values())
    assert "Output already exists in Drive" in sources["step7-copy-outputs"]
    assert "Missing required Sprint 3 Graph C artifact" in sources["step4-copy-data"]
    assert "Missing S5F2 source" in sources["step4-copy-data"]
    assert "from crispr_gnn.graph.graph_schemas import GRAPH_C" in sources["step5-build-artifacts"]
    assert "PYTHONPATH=src uv run python - <<'PY'" in sources["step5-build-artifacts"]
    assert "--source-artifact-dir data/processed/graphs/sprint3" in sources["step5-build-artifacts"]
    assert "sprint5/epigenetic-ablation" in sources["step2-clone"]


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
