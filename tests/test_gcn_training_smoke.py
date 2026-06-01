import pandas as pd

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.graph import GRAPH_A, GRAPH_C, build_graph_artifacts, write_graph_artifacts
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader
from crispr_gnn.training.gcn import (
    CHECKPOINT_POLICY,
    GCNRunConfig,
    train_graph_a_gcn,
    train_graph_c_gcn,
)


def test_graph_a_gcn_cpu_smoke_training_preserves_validation_only_selection(tmp_path) -> None:
    materialized = _tiny_materialized_graph_a(tmp_path)
    config = GCNRunConfig(
        sprint="sprint4",
        split_id="sprint2_main_seed42",
        seed=7,
        hidden_dim=12,
        num_layers=1,
        dropout=0.0,
        max_epochs=3,
        min_epochs=1,
        patience=2,
        learning_rate=0.01,
        edge_feature_sets=("s1_pair", "f1"),
        device="cpu",
        num_threads=1,
        use_compile=False,
        use_amp=False,
    )

    results, predictions, history = train_graph_a_gcn(materialized, config)

    row = results.iloc[0]
    assert row["graph_schema"] == GRAPH_A
    assert row["label_scheme"] == "scheme_a"
    assert row["split_id"] == "sprint2_main_seed42"
    assert row["visibility_policy"] == "strict_inductive_primary"
    assert row["target_node_representation"] == "zero_type_feature"
    assert row["checkpoint_policy"] == CHECKPOINT_POLICY
    assert row["checkpoint_selection_split"] == "validation"
    assert row["threshold_selection_split"] == "validation"
    assert row["baseline_reference"] == "xgboost_unweighted / F4"
    assert row["val_rows"] == 2
    assert row["test_rows"] == 2
    assert set(predictions["split"]) == {"val", "test"}
    assert "test_auprc" in results.columns
    assert "test" not in set(history["selection_split"])
    assert history["epoch"].max() <= 3
    assert "val_loss" in history.columns, "val_loss must be tracked in training history"
    assert "lr" in history.columns, "learning rate must be tracked in training history"
    assert (history["val_loss"] >= 0).all(), "val_loss must be non-negative"
    assert "use_compile" in results.columns, "use_compile provenance must be in results"
    assert "use_amp" in results.columns, "use_amp provenance must be in results"


def test_graph_a_gcn_saves_checkpoint_when_path_provided(tmp_path) -> None:
    materialized = _tiny_materialized_graph_a(tmp_path)
    config = GCNRunConfig(
        sprint="sprint4",
        split_id="sprint2_main_seed42",
        seed=7,
        hidden_dim=12,
        num_layers=1,
        dropout=0.0,
        clip_grad_norm=1.0,
        scheduler="reduce_on_plateau",
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_min_lr=1e-5,
        max_epochs=2,
        min_epochs=1,
        patience=2,
        learning_rate=0.01,
        edge_feature_sets=("s1_pair", "f1"),
        device="cpu",
        num_threads=1,
    )
    checkpoint_path = tmp_path / "model.pt"

    train_graph_a_gcn(materialized, config, checkpoint_path=checkpoint_path)

    assert checkpoint_path.exists(), "model.pt must be written when checkpoint_path is provided"
    import torch
    state = torch.load(checkpoint_path, weights_only=True)
    assert isinstance(state, dict) and len(state) > 0, "checkpoint must be a non-empty state dict"


def test_graph_c_gcn_cpu_smoke_training_uses_observation_context_encoder(tmp_path) -> None:
    materialized = _tiny_materialized_graph(tmp_path, GRAPH_C)
    config = GCNRunConfig(
        sprint="sprint4",
        split_id="sprint2_main_seed42",
        seed=7,
        graph_schema=GRAPH_C,
        model_name="gcn_graph_c",
        feature_set="CandidatePair",
        edge_feature_sets=("candidate_pair_features",),
        target_node_representation="target_observation_context_encoder",
        hidden_dim=12,
        num_layers=1,
        dropout=0.0,
        max_epochs=3,
        min_epochs=1,
        patience=2,
        learning_rate=0.01,
        device="cpu",
        num_threads=1,
        use_compile=False,
        use_amp=False,
    )

    results, predictions, history = train_graph_c_gcn(materialized, config)

    row = results.iloc[0]
    assert row["graph_schema"] == GRAPH_C
    assert row["target_node_representation"] == "target_observation_context_encoder"
    assert row["target_semantics"] == "observation_level_context_target"
    assert "topology and target semantics" in row["notes"]
    assert row["checkpoint_selection_split"] == "validation"
    assert row["threshold_selection_split"] == "validation"
    assert set(predictions["split"]) == {"val", "test"}
    assert "test" not in set(history["selection_split"])
    assert history["epoch"].max() <= 3


def _tiny_materialized_graph_a(tmp_path):
    return _tiny_materialized_graph(tmp_path, GRAPH_A)


def _tiny_materialized_graph(tmp_path, graph_name: str):
    artifacts = build_graph_artifacts(_graph_rows())
    graph_dir = tmp_path / "graphs"
    write_graph_artifacts(
        artifacts,
        artifact_dir=graph_dir,
        report_path=tmp_path / "graph_schema_report.md",
        split_id="sprint2_main_seed42",
    )
    expected_counts = {
        name: {
            "nodes": {node_type: len(table) for node_type, table in artifact.nodes.items()},
            "relations": {relation_type: len(table) for relation_type, table in artifact.relations.items()},
        }
        for name, artifact in artifacts.items()
    }
    loader = Sprint3HeteroDataLoader(
        graph_dir,
        expected_counts=expected_counts,
        expected_candidate_split_counts={"train": 6, "val": 2, "test": 2},
    )
    return loader.load(graph_name)


def _graph_rows() -> pd.DataFrame:
    valid_array = "[" + " ".join(str(value) for value in range(23)) + "]"
    rows = []
    source_id = 1
    for split, guides in {"train": ["1", "2", "3"], "val": ["4"], "test": ["5"]}.items():
        for guide_position, guide in enumerate(guides):
            for row_position in range(2):
                row = {
                    "id": source_id,
                    "split": split,
                    "label": int(row_position == 0),
                    "measured": 1,
                    "experiment_id": 1,
                    "cleavage_freq": 2e-5 if row_position == 0 else 0.0,
                    "grna_target_id": guide,
                    "grna_target_sequence": f"ACGTACGTACGTACGTACG{guide}AGG"[:23],
                    "target_sequence": (
                        "ACGTACGTACGTACGTACGTAGG"
                        if row_position == 0
                        else "TCGTACGTACGTACGTACGTAGG"
                    ),
                    "genome": "hg19",
                    "target_chr": "chr1",
                    "target_start": 100 + source_id,
                    "target_end": 123 + source_id,
                    "target_strand": "+",
                    "energy_1": float(row_position),
                    "energy_2": 0.2,
                    "energy_3": 0.3,
                    "energy_4": 0.4,
                    "energy_5": 0.5,
                    "epigen_ctcf": float(guide_position),
                    "epigen_dnase": 0.2,
                    "epigen_rrbs": 0.3,
                    "epigen_h3k4me3": 0.4,
                    "epigen_drip": 0.5,
                    "MNase": float(source_id),
                }
                for feature in COMPUTED_NUCLEOSOME_FEATURES:
                    row[feature] = valid_array
                rows.append(row)
                source_id += 1
    return pd.DataFrame(rows)
