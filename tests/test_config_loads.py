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
