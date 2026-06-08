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
    assert "data/processed/graphs/sprint5/graph_a_minimal_physical_target" in sources["step5b-copy-artifacts-to-drive"]
    assert "rsync -a data/processed/graphs/sprint5/" in sources["step5b-copy-artifacts-to-drive"]
    assert "$DRIVE_ROOT/data/processed/graphs/sprint5/" in sources["step5b-copy-artifacts-to-drive"]


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
    assert "from crispr_gnn.graph.graph_schemas import GRAPH_C" in sources["step5-build-artifacts"]
    assert "PYTHONPATH=src uv run python - <<'PY'" in sources["step5-build-artifacts"]
    assert "--source-artifact-dir data/processed/graphs/sprint3" in sources["step5-build-artifacts"]
    assert "sprint5/epigenetic-ablation" in sources["step2-clone"]


def test_sprint6_colab_runner_contract() -> None:
    notebook_path = ROOT / "colab" / "sprint6_loss_comparison_runner.ipynb"
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }
    all_code = "\n".join(sources.values())
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "rm -rf" not in all_code
    assert "def " not in all_code
    assert "class " not in all_code
    assert "BCEWithLogitsLoss" not in all_source
    assert "train_graph_a_gcn" not in all_source
    assert "write_sprint6_imbalance" not in all_source
    assert "select_threshold" not in all_source
    assert "build_sprint5_graph_a_features.py" not in all_source
    assert "build_sprint6" not in all_source
    assert "scripts/train.py" not in all_source
    assert "uv sync" in sources["step3-sync"]
    assert "uv run python scripts/run_sprint6_loss_comparison.py" in sources["step6-run-sweep"]
    assert "configs/sweeps/sprint6_loss_comparison.yaml" in sources["step6-run-sweep"]
    assert "--run-id \"$RUN_ID\"" in sources["step6-run-sweep"]
    assert "--include-optional-runs" not in all_source
    assert "S6R8" not in all_source
    assert "S6R9" not in all_source
    assert "S6S1" not in all_source
    assert "data/processed/graphs/sprint5" in all_source
    assert "rsync -a \"$GRAPH_SOURCE/\" data/processed/graphs/sprint5/" in sources["step4-copy-artifacts"]
    assert "from crispr_gnn.graph.graph_schemas import GRAPH_A" in sources["step5-validate-artifacts"]
    assert "from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader" in sources["step5-validate-artifacts"]
    assert "S5F2_energy" in sources["step5-validate-artifacts"]
    assert "Output already exists in Drive" in sources["step7-copy-outputs"]
    assert "optional_runs_executed" in sources["step8-returned-checks"]


def test_sprint7_gat_gatv2_config_loads() -> None:
    config = load_yaml(ROOT / "configs" / "sweeps" / "sprint7_gat_gatv2.yaml")
    assert config["experiment_name"] == "sprint7_gat_gatv2_attention"
    assert config["task"] == "sprint7_gat_gatv2_attention"
    assert config["sprint"] == "sprint7"
    assert config["graph"]["schema"] == "graph_a_minimal_physical_target"
    assert config["data"]["graph_artifact_dir"] == "data/processed/graphs/sprint5"
    assert config["features"]["edge_feature_sets"] == ["s5f2_energy"]
    assert config["features"]["feature_set"] == "S5F2_energy"
    assert config["training"]["loss"] == "weighted_bce"
    assert config["training"]["loss_params"] == {"pos_weight": "auto"}
    assert [run["id"] for run in config["runs"]] == ["S7R1_gat_edge_aware", "S7R2_gatv2_edge_aware"]
    assert all(run["edge_aware_attention"] is True for run in config["runs"])
    assert all(run["attention"]["self_loop_edge_fill"] == 0.0 for run in config["runs"])
    validate_gcn_headline_config(config)


def test_sprint7_colab_runner_contract() -> None:
    notebook_path = ROOT / "colab" / "sprint7_gat_gatv2_runner.ipynb"
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {
        cell.get("id"): "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    }
    all_code = "\n".join(sources.values())
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "rm -rf" not in all_code
    assert "def " not in all_code
    assert "class " not in all_code
    assert "BCEWithLogitsLoss" not in all_source
    assert "train_graph_a_gcn" not in all_source
    assert "select_threshold" not in all_source
    assert "scripts/train.py" not in all_source
    assert "uv sync" in sources["step3-sync"]
    assert "GIT_REF=\"sprint7/gat-gatv2\"" in sources["step2-clone"]
    assert "uv run python scripts/run_sprint7_gat_comparison.py" in sources["step6-run-sweep"]
    assert "configs/sweeps/sprint7_gat_gatv2.yaml" in sources["step6-run-sweep"]
    assert "--run-id \"$RUN_ID\"" in sources["step6-run-sweep"]
    assert "--include-optional-runs" not in all_source
    assert "S7R3" not in all_source
    assert "data/processed/graphs/sprint5" in all_source
    assert "rsync -a \"$GRAPH_SOURCE/\" data/processed/graphs/sprint5/" in sources["step4-copy-artifacts"]
    assert "from crispr_gnn.graph.graph_schemas import GRAPH_A" in sources["step5-validate-artifacts"]
    assert "from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader" in sources["step5-validate-artifacts"]
    assert "S5F2_energy" in sources["step5-validate-artifacts"]
    assert "Output already exists in Drive" in sources["step7-copy-outputs"]
    assert "S7R1_gat_edge_aware" in sources["step8-returned-checks"]
    assert "S7R2_gatv2_edge_aware" in sources["step8-returned-checks"]
    assert "attention_weight_summary.csv" in sources["step8-returned-checks"]


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
