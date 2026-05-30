import json
from copy import deepcopy

import pandas as pd
import pytest
from torch_geometric.data import HeteroData

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.graph import (
    GRAPH_A,
    GRAPH_B,
    GRAPH_C,
    build_graph_artifacts,
    write_graph_artifacts,
)
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader


def make_graph_rows() -> pd.DataFrame:
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
                    "target_start": 100 if row_position == 0 else 100 + source_id,
                    "target_end": 123 if row_position == 0 else 123 + source_id,
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


@pytest.fixture
def serialized_loader(tmp_path) -> Sprint3HeteroDataLoader:
    artifacts = build_graph_artifacts(make_graph_rows())
    write_graph_artifacts(
        artifacts,
        artifact_dir=tmp_path / "graphs",
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
    return Sprint3HeteroDataLoader(
        tmp_path / "graphs",
        expected_counts=expected_counts,
        expected_candidate_split_counts={"train": 6, "val": 2, "test": 2},
    )


def test_loader_materializes_strict_inductive_heterodata_views(serialized_loader) -> None:
    materialized = serialized_loader.load(GRAPH_A)
    assert set(materialized.views) == {"train", "val", "test"}

    expected_visible_edges = {"train": 6, "val": 8, "test": 8}
    for view, expected_count in expected_visible_edges.items():
        data = materialized.view(view)
        relation = data[("sgRNA", "candidate_pair", "physical_target_site")]
        assert isinstance(data, HeteroData)
        assert relation.edge_index.shape[1] == expected_count
        assert int(relation.supervision_mask.sum()) == {"train": 6, "val": 2, "test": 2}[view]
        assert data.split_id == "sprint2_main_seed42"
        assert data.visibility_policy == "strict_inductive_primary"


def test_loader_preserves_graph_a_feature_placement_without_metadata_features(serialized_loader) -> None:
    data = serialized_loader.load(GRAPH_A).view("train")
    candidate = data[("sgRNA", "candidate_pair", "physical_target_site")]

    assert "x" not in data["physical_target_site"]
    assert "edge_attr_f4" in candidate
    assert candidate.edge_attr_f4.shape[1] == 135
    forbidden = {
        "label",
        "cleavage_freq",
        "measured",
        "experiment_id",
        "split",
        "genome",
        "target_chr",
        "target_start",
        "target_end",
        "target_strand",
    }
    predictive_names = {name.removeprefix("feature__") for name in candidate.feature_names_f4}
    assert not predictive_names & forbidden


def test_loader_preserves_graph_b_and_graph_c_test_visibility(serialized_loader) -> None:
    graph_b = serialized_loader.load(GRAPH_B).view("test")
    _assert_no_validation_or_heldout_to_heldout_links(
        graph_b,
        node_type="sgRNA",
        edge_type=("sgRNA", "sequence_similar_to", "sgRNA"),
    )

    graph_c = serialized_loader.load(GRAPH_C).view("test")
    _assert_no_validation_or_heldout_to_heldout_links(
        graph_c,
        node_type="target_observation",
        edge_type=("target_observation", "context_similar_to", "target_observation"),
    )
    assert "x" in graph_c["target_observation"]


def test_loader_fails_fast_on_manifest_contract_drift(serialized_loader) -> None:
    graph_dir = serialized_loader.artifact_dir / GRAPH_A
    manifest_path = graph_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    drifted = deepcopy(manifest)
    drifted["split_id"] = "wrong_split"
    manifest_path.write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(ValueError, match="split contract drift"):
        serialized_loader.load(GRAPH_A)


@pytest.mark.parametrize("graph_name", [GRAPH_B, GRAPH_C])
def test_loader_rejects_candidate_label_drift_from_graph_a(serialized_loader, graph_name) -> None:
    relation_path = serialized_loader.artifact_dir / graph_name / "relation_candidate_pair.parquet"
    relation = pd.read_parquet(relation_path)
    relation.loc[relation.index[0], "label"] = 1 - int(relation.loc[relation.index[0], "label"])
    relation.to_parquet(relation_path, index=False)

    with pytest.raises(ValueError, match="Candidate supervised contract drift"):
        serialized_loader.load(graph_name)


def test_loader_rejects_graph_b_physical_target_relation_drift(serialized_loader) -> None:
    relation_path = serialized_loader.artifact_dir / GRAPH_B / "relation_candidate_pair.parquet"
    relation = pd.read_parquet(relation_path)
    relation.loc[relation.index[0], "target_node_id"] = relation.loc[relation.index[1], "target_node_id"]
    relation.to_parquet(relation_path, index=False)

    with pytest.raises(ValueError, match="Candidate supervised contract drift"):
        serialized_loader.load(GRAPH_B)


def test_loader_rejects_graph_c_observation_identity_drift(serialized_loader) -> None:
    relation_path = serialized_loader.artifact_dir / GRAPH_C / "relation_candidate_pair.parquet"
    relation = pd.read_parquet(relation_path)
    relation.loc[relation.index[0], "target_observation_id"] = relation.loc[relation.index[1], "target_observation_id"]
    relation.to_parquet(relation_path, index=False)

    with pytest.raises(ValueError, match="source-row observation identity"):
        serialized_loader.load(GRAPH_C)


def _assert_no_validation_or_heldout_to_heldout_links(
    data: HeteroData,
    *,
    node_type: str,
    edge_type: tuple[str, str, str],
) -> None:
    node_splits = data[node_type].audit_splits
    relation = data[edge_type].edge_index
    visible_splits = {
        node_splits[index]
        for index in relation.reshape(-1).tolist()
    }
    assert "val" not in visible_splits
    for source, target in relation.T.tolist():
        assert not (node_splits[source] == "test" and node_splits[target] == "test")
