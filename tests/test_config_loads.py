from pathlib import Path

from crispr_gnn.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_mak2022_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "data" / "mak2022.yaml")
    assert config["dataset"]["name"] == "mak2022"
    assert config["split_rules"]["test_measured_only"] is True


def test_gcn_minimal_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "experiments" / "gcn_minimal.yaml")
    assert config["experiment_name"] == "gcn_minimal"
    assert config["model"]["name"] == "gcn"
    assert config["graph"]["schema"] == "minimal_bipartite"


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
