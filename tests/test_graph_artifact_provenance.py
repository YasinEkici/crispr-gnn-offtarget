import json
import importlib.util
from pathlib import Path

import pandas as pd

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.graph import build_graph_artifacts, write_graph_artifacts
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_graph_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("validate_graph_artifacts", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_provenance_record = _MODULE.build_provenance_record


def test_graph_artifact_provenance_validates_and_hashes_serialized_artifacts(tmp_path) -> None:
    artifacts = build_graph_artifacts(_make_graph_rows())
    artifact_dir = tmp_path / "graphs"
    write_graph_artifacts(
        artifacts,
        artifact_dir=artifact_dir,
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

    record = build_provenance_record(
        artifact_dir,
        approved_source="test_fixture",
        loader_factory=lambda path: Sprint3HeteroDataLoader(
            path,
            expected_counts=expected_counts,
            expected_candidate_split_counts={"train": 6, "val": 2, "test": 2},
        ),
    )

    assert record["provenance_type"] == "sprint3_graph_artifacts_sha256"
    assert record["approved_source"] == "test_fixture"
    assert set(record["schemas"]) == set(expected_counts)
    assert all(len(row["sha256"]) == 64 for row in record["files"])
    assert any(row["path"].endswith("manifest.json") for row in record["files"])
    json.dumps(record)


def _make_graph_rows() -> pd.DataFrame:
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
