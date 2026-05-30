import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.graph import GRAPH_A, GRAPH_B, GRAPH_C, GraphBuildConfig, build_graph_artifacts, write_graph_artifacts
from crispr_gnn.graph.graph_schemas import SimilarityConfig


ROOT = Path(__file__).resolve().parents[1]


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
                    "target_sequence": "ACGTACGTACGTACGTACGTAGG" if row_position == 0 else "TCGTACGTACGTACGTACGTAGG",
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


def test_graph_construction_package_import_does_not_load_pyg() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import crispr_gnn.graph; assert 'torch_geometric' not in sys.modules",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_build_graph_artifacts_preserves_candidate_edges_and_context_placement() -> None:
    assigned = make_graph_rows()
    artifacts = build_graph_artifacts(
        assigned,
        GraphBuildConfig(max_length=23),
    )

    graph_a = artifacts[GRAPH_A]
    assert set(artifacts) == {GRAPH_A, GRAPH_B, GRAPH_C}
    assert graph_a.relations["candidate_pair"].shape[0] == assigned.shape[0]
    assert set(graph_a.relations["candidate_pair"]["edge_id"]) == set(assigned["id"].astype(str))
    assert not any(column.startswith("feature__") for column in graph_a.nodes["physical_target_site"].columns)
    assert "F4" in graph_a.feature_tables
    assert not graph_a.feature_tables["F4"].isna().any().any()
    assert graph_a.preprocessing["F4"]["fit_scope"] == "train_only"
    assert graph_a.metadata["candidate_relation_visibility"] == "filter_by_split_for_model_views"


def test_graph_c_uses_observation_nodes_and_context_similarity() -> None:
    assigned = make_graph_rows()
    graph_c = build_graph_artifacts(assigned)[GRAPH_C]

    assert graph_c.nodes["target_observation"].shape[0] == assigned.shape[0]
    assert graph_c.nodes["target_observation"]["node_id"].is_unique
    assert graph_c.preprocessing["target_observation_context"]["fit_scope"] == "train_only"
    relation = graph_c.relations["context_similar_to"]
    split_by_node = graph_c.nodes["target_observation"].set_index("node_id")["split"].to_dict()
    for view in ["val", "test"]:
        part = relation.loc[relation["view"] == view]
        for source, target in part[["source_observation_id", "target_observation_id"]].itertuples(index=False):
            assert {split_by_node[source], split_by_node[target]} <= {"train", view}
            assert not (split_by_node[source] == view and split_by_node[target] == view)


def test_graph_b_connects_held_out_guides_only_to_training_guides() -> None:
    graph_b = build_graph_artifacts(make_graph_rows())[GRAPH_B]
    split_by_node = graph_b.nodes["sgRNA"].set_index("node_id")["split"].to_dict()
    relation = graph_b.relations["sequence_similar_to"]

    for view in ["val", "test"]:
        part = relation.loc[relation["view"] == view]
        for source, target in part[["source_sgrna_id", "target_sgrna_id"]].itertuples(index=False):
            assert {split_by_node[source], split_by_node[target]} <= {"train", view}
            assert not (split_by_node[source] == view and split_by_node[target] == view)

    assert graph_b.metadata["auxiliary_relation_view_composition"]["test"] == ["train", "test"]


def test_graph_c_topology_is_invariant_to_input_row_order() -> None:
    assigned = make_graph_rows()
    original = build_graph_artifacts(assigned)[GRAPH_C].relations["context_similar_to"]
    shuffled = build_graph_artifacts(assigned.sample(frac=1, random_state=11))[GRAPH_C].relations["context_similar_to"]
    sort_columns = ["view", "source_observation_id", "target_observation_id", "distance"]

    pd.testing.assert_frame_equal(
        original.sort_values(sort_columns).reset_index(drop=True),
        shuffled.sort_values(sort_columns).reset_index(drop=True),
    )


def test_graph_c_equal_distance_ties_are_resolved_by_row_id() -> None:
    assigned = make_graph_rows()
    context_columns = ["epigen_ctcf", "epigen_dnase", "epigen_rrbs", "epigen_h3k4me3", "epigen_drip", "MNase"]
    assigned[context_columns] = 0.0
    assigned.loc[assigned["split"] == "train", "id"] = [130, 131, 110, 111, 120, 121]
    config = GraphBuildConfig(graph_c=SimilarityConfig(top_k=1, metric="euclidean"))

    original = build_graph_artifacts(assigned, config)[GRAPH_C].relations["context_similar_to"]
    shuffled = build_graph_artifacts(assigned.sample(frac=1, random_state=3), config)[GRAPH_C].relations["context_similar_to"]
    val_sources = set(assigned.loc[assigned["split"] == "val", "id"].astype(str))
    original_targets = original.loc[
        original["source_observation_id"].isin(val_sources) & (original["source_split"] == "val"),
        "target_observation_id",
    ].tolist()
    shuffled_targets = shuffled.loc[
        shuffled["source_observation_id"].isin(val_sources) & (shuffled["source_split"] == "val"),
        "target_observation_id",
    ].tolist()

    assert original_targets == ["110", "110"]
    assert shuffled_targets == original_targets


def test_graph_b_topology_is_invariant_to_input_row_order() -> None:
    assigned = make_graph_rows()
    original = build_graph_artifacts(assigned)[GRAPH_B].relations["sequence_similar_to"]
    shuffled = build_graph_artifacts(assigned.sample(frac=1, random_state=13))[GRAPH_B].relations["sequence_similar_to"]
    sort_columns = ["view", "source_sgrna_id", "target_sgrna_id", "distance"]

    pd.testing.assert_frame_equal(
        original.sort_values(sort_columns).reset_index(drop=True),
        shuffled.sort_values(sort_columns).reset_index(drop=True),
    )


def test_serialized_artifacts_preserve_manifests_report_and_feature_audit(tmp_path) -> None:
    assigned = make_graph_rows()
    artifacts = build_graph_artifacts(assigned)
    report_path, _ = write_graph_artifacts(
        artifacts,
        artifact_dir=tmp_path / "graphs",
        report_path=tmp_path / "graph_schema_report.md",
        split_id="sprint2_main_seed42",
    )
    report = report_path.read_text(encoding="utf-8")
    forbidden = {
        "id",
        "label",
        "cleavage_freq",
        "measured",
        "experiment_id",
        "cell_line",
        "genome",
        "target_chr",
        "target_start",
        "target_end",
        "target_strand",
        "grna_target_id",
        "split",
    }

    for graph_name, artifact in artifacts.items():
        graph_dir = tmp_path / "graphs" / graph_name
        manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["graph_name"] == graph_name
        assert manifest["split_id"] == "sprint2_main_seed42"
        assert manifest["label_scheme"] == "scheme_a"
        assert manifest["metadata"]["visibility_policy"] == "strict_inductive_primary"
        assert manifest["nodes"] == {name: len(table) for name, table in artifact.nodes.items()}
        assert manifest["relations"] == {name: len(table) for name, table in artifact.relations.items()}
        assert f"`{graph_name}`" in report
        if graph_name in {GRAPH_B, GRAPH_C}:
            assert manifest["metadata"]["top_k"] == 5

        for feature_path in graph_dir.glob("features_*.parquet"):
            columns = pd.read_parquet(feature_path).columns.tolist()
            assert columns[0] == "record_id"
            predictive = {column.removeprefix("feature__") for column in columns[1:]}
            assert not predictive & forbidden


def test_all_graph_candidates_preserve_the_same_supervised_contract() -> None:
    assigned = make_graph_rows()
    artifacts = build_graph_artifacts(assigned)
    expected = assigned.assign(
        edge_id=assigned["id"].astype(str),
        expected_split=assigned["split"],
        expected_label=assigned["label"],
    ).set_index("edge_id")

    for artifact in artifacts.values():
        candidate = artifact.relations["candidate_pair"].set_index("edge_id")
        assert set(candidate.index) == set(expected.index)
        assert candidate["split"].to_dict() == expected["expected_split"].to_dict()
        assert candidate["label"].to_dict() == expected["expected_label"].to_dict()
        assert (candidate["measured"] == 1).all()
        assert (candidate["experiment_id"] != 18).all()
