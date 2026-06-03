import pandas as pd
import pytest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sprint5b_graph_c_energy_features import _load_or_build_s5f2_table
from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.features.sprint5 import (
    SPRINT5_FEATURE_SET_ORDER,
    build_computed_nucleosome_position_features,
    build_sequence_only_pair_features,
    build_sprint5_feature_set,
    build_sprint5_feature_tables,
)
from crispr_gnn.models.gcn import graph_a_edge_feature_attrs, graph_b_edge_feature_attrs, graph_c_edge_feature_attrs


def test_s5f0_sequence_only_excludes_explicit_mismatch_channel() -> None:
    features = build_sequence_only_pair_features(_rows().iloc[:1])

    assert features.shape[1] == 23 * 10
    assert not any("mismatch" in column for column in features.columns)
    assert build_sprint5_feature_set(_rows().iloc[:1], "S5F0_seq").shape[1] == 23 * 10


def test_s5f5_position_resolved_computed_features_have_expected_shape() -> None:
    features = build_computed_nucleosome_position_features(_rows())

    assert features.shape[1] == len(COMPUTED_NUCLEOSOME_FEATURES) * 23
    assert f"{COMPUTED_NUCLEOSOME_FEATURES[0]}_pos_00" in features.columns
    assert f"{COMPUTED_NUCLEOSOME_FEATURES[0]}_pos_22" in features.columns


def test_sprint5_feature_tables_are_keyed_and_imputed_train_only() -> None:
    tables, sources, preprocessing = build_sprint5_feature_tables(_rows())

    assert set(tables) == set(SPRINT5_FEATURE_SET_ORDER)
    for feature_set, table in tables.items():
        assert table.columns[0] == "record_id"
        assert table["record_id"].astype(str).tolist() == ["1", "2", "3"]
        assert not table.drop(columns=["record_id"]).isna().any().any()
        assert sources[feature_set]
        assert preprocessing[feature_set]["fit_scope"] == "train_only"


def test_graph_a_accepts_sprint5_feature_attrs() -> None:
    attrs = graph_a_edge_feature_attrs(["s5f0_seq", "S5F5_computed_pos"])

    assert attrs == ["edge_attr_s5f0_seq", "edge_attr_s5f5_computed_pos"]


def test_sprint5b_does_not_open_graph_b_or_full_graph_c_feature_ladders() -> None:
    assert graph_c_edge_feature_attrs(["s5f2_energy"]) == ["edge_attr_s5f2_energy"]
    with pytest.raises(ValueError, match="Unsupported Graph B"):
        graph_b_edge_feature_attrs(["s5f2_energy"])
    with pytest.raises(ValueError, match="Unsupported Graph C"):
        graph_c_edge_feature_attrs(["s5f0_seq"])


def test_sprint5b_prefers_existing_s5f2_feature_table_without_raw_dataset(tmp_path) -> None:
    feature_dir = tmp_path / "data" / "processed" / "graphs" / "sprint5" / "graph_a_minimal_physical_target"
    feature_dir.mkdir(parents=True)
    table = pd.DataFrame({"record_id": ["1", "2"], "feature__energy_1": [0.1, 0.2]})
    table.to_parquet(feature_dir / "features_S5F2_energy.parquet", index=False)
    (feature_dir / "manifest.json").write_text(
        (
            '{"feature_sources":{"S5F2_energy":["energy_1"]},'
            '"preprocessing":{"S5F2_energy":{"fit_scope":"train_only"}}}'
        ),
        encoding="utf-8",
    )

    loaded, sources, preprocessing = _load_or_build_s5f2_table(
        data_config={"dataset": {"raw_path": "missing.parquet", "processed_path": "also_missing.parquet"}},
        schema_mapping={"split_manifest": "missing_split.json"},
        graph_config=object(),
        source_sprint5_feature_table=feature_dir / "features_S5F2_energy.parquet",
    )

    assert loaded.equals(table)
    assert sources == ["energy_1"]
    assert preprocessing == {"fit_scope": "train_only"}


def _rows() -> pd.DataFrame:
    valid_array = "[" + " ".join(str(value) for value in range(23)) + "]"
    rows = []
    for index, split in enumerate(["train", "train", "test"], start=1):
        row = {
            "id": index,
            "split": split,
            "grna_target_sequence": "ACGTACGTACGTACGTACGTAGG"[:23],
            "target_sequence": "TCGTACGTACGTACGTACGTAGG"[:23],
            "energy_1": 0.1,
            "energy_2": 0.2,
            "energy_3": 0.3,
            "energy_4": 0.4,
            "energy_5": 0.5,
            "epigen_ctcf": 1.0,
            "epigen_dnase": 2.0,
            "epigen_rrbs": 3.0,
            "epigen_h3k4me3": 4.0,
            "epigen_drip": 5.0,
            "MNase": 6.0,
        }
        for feature in COMPUTED_NUCLEOSOME_FEATURES:
            row[feature] = valid_array
        rows.append(row)
    return pd.DataFrame(rows)
