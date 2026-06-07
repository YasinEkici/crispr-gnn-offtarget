from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.features.sprint5 import build_sprint5_feature_tables  # noqa: E402
from crispr_gnn.graph.graph_builder import FEATURE_PREFIX  # noqa: E402
from crispr_gnn.graph.graph_schemas import GRAPH_A, GRAPH_B, GraphBuildConfig  # noqa: E402
from crispr_gnn.graph.pyg_dataset import Sprint3HeteroDataLoader  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sprint 7B Graph B S5F2-energy artifacts.")
    parser.add_argument("--data-config", default="configs/data/mak2022.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--schema-config",
        default="configs/sweeps/graph_schema_ablation.yaml",
        help="Path to Sprint 3 graph schema YAML, used for split and representation settings.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="data/processed/graphs/sprint7b",
        help="Output graph artifact directory. Contains Graph A reference plus Graph B with S5F2 energy edge features.",
    )
    parser.add_argument(
        "--source-artifact-dir",
        default="data/processed/graphs/sprint3",
        help="Canonical Sprint 3 graph artifact directory to copy. Graph B topology is not rebuilt.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/sprint7b/graph_b_s5f2_artifact_report.md",
        help="Output artifact report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_yaml(args.data_config)
    schema_mapping = load_yaml(args.schema_config)
    graph_config = GraphBuildConfig.from_mapping(schema_mapping)
    dataset_path = _resolve_dataset_path(data_config.get("dataset", {}))
    split_path = ROOT / Path(str(schema_mapping.get("split_manifest", "outputs/splits/sprint2_guides.json")))
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    print(f"Dataset path: {dataset_path.relative_to(ROOT)}")
    raw = pd.read_parquet(dataset_path)
    split = load_split_manifest(split_path)
    assigned = assign_measured_splits(raw, split)
    source_artifact_dir = ROOT / args.source_artifact_dir
    _validate_source_artifacts(source_artifact_dir)

    feature_tables, feature_sources, preprocessing = build_sprint5_feature_tables(
        assigned,
        max_length=graph_config.max_length,
        scale=False,
    )
    artifact_dir = ROOT / args.artifact_dir
    written = _copy_canonical_graphs(source_artifact_dir, artifact_dir)
    graph_b_dir = artifact_dir / GRAPH_B
    s5f2_table = feature_tables["S5F2_energy"]
    _validate_s5f2_matches_graph_b_candidates(graph_b_dir, s5f2_table)
    s5f2_path = graph_b_dir / "features_S5F2_energy.parquet"
    s5f2_table.to_parquet(s5f2_path, index=False)
    written.append(s5f2_path)

    manifest_path = graph_b_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("feature_tables", {})["S5F2_energy"] = int(s5f2_table.shape[1] - 1)
    manifest.setdefault("feature_sources", {})["S5F2_energy"] = feature_sources["S5F2_energy"]
    manifest.setdefault("preprocessing", {})["sprint7b_graph_b_s5f2"] = {
        "fit_scope": "train_only",
        "feature_set": "S5F2_energy",
        "preprocessing": preprocessing["S5F2_energy"],
    }
    manifest.setdefault("metadata", {})["sprint7b_graph_b_s5f2"] = {
        "role": "matched_graph_b_s5f2_candidate_edge_feature_attachment",
        "topology": "Graph B guide-similarity topology unchanged",
        "base_graph": GRAPH_A,
        "candidate_edge_feature_set": "S5F2_energy",
        "feature_table_storage": GRAPH_B,
        "attention_edge_policy": (
            "candidate edges use S5F2_energy edge_attr; sequence_similar_to edges are topology-only "
            "and receive zero edge_attr in Sprint 7B GATv2 message passing"
        ),
        "interpretation_boundary": (
            "This is a matched Graph B S5F2 comparison for Sprint 7B, not a new feature-family ablation."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(manifest_path)

    Sprint3HeteroDataLoader(artifact_dir).load(GRAPH_B)
    report_path = ROOT / args.report_path
    _write_artifact_report(report_path, manifest=manifest)

    print(f"Sprint 7B Graph B artifact report written: {report_path.relative_to(ROOT)}")
    print(f"Sprint 7B artifact directory: {artifact_dir.relative_to(ROOT)}")
    print(f"Graph B feature tables: {', '.join(manifest.get('feature_tables', {}))}")
    print(f"Total artifact files written: {len(written)}")
    return 0


def _validate_source_artifacts(source_artifact_dir: Path) -> None:
    if not source_artifact_dir.exists():
        raise FileNotFoundError(f"Source graph artifacts not found: {source_artifact_dir}")
    Sprint3HeteroDataLoader(source_artifact_dir).load(GRAPH_A)
    Sprint3HeteroDataLoader(source_artifact_dir).load(GRAPH_B)


def _copy_canonical_graphs(source_artifact_dir: Path, artifact_dir: Path) -> list[Path]:
    written: list[Path] = []
    for graph_name in (GRAPH_A, GRAPH_B):
        source = source_artifact_dir / graph_name
        destination = artifact_dir / graph_name
        if not source.exists():
            raise FileNotFoundError(f"Source graph artifact missing: {source}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        written.extend(path for path in destination.rglob("*") if path.is_file())
    return written


def _validate_s5f2_matches_graph_b_candidates(graph_b_dir: Path, table: pd.DataFrame) -> None:
    relation = pd.read_parquet(graph_b_dir / "relation_candidate_pair.parquet")
    expected = set(relation["edge_id"].astype(str))
    actual = set(table["record_id"].astype(str))
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"S5F2 feature keys do not match Graph B candidate edges; missing={missing}, extra={extra}")
    feature_columns = [column for column in table.columns if column.startswith(FEATURE_PREFIX)]
    if len(feature_columns) != table.shape[1] - 1:
        raise ValueError("S5F2 feature table contains unexpected non-feature columns")


def _resolve_dataset_path(dataset: dict[str, Any]) -> Path:
    candidates = [
        ROOT / Path(str(dataset.get("raw_path", ""))),
        ROOT / Path(str(dataset.get("processed_path", ""))),
    ]
    for path in candidates:
        if path.exists():
            return path
    readable = ", ".join(str(path) for path in candidates if str(path) != str(ROOT))
    raise FileNotFoundError(f"Dataset not found. Checked: {readable}")


def _write_artifact_report(path: Path, *, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_counts = manifest.get("feature_tables", {})
    lines = [
        "# Sprint 7B Graph B S5F2 Artifact Report",
        "",
        "Sprint 7B keeps Graph B topology fixed and attaches the Sprint 5 `S5F2_energy` candidate-edge feature table for a matched GCN-vs-GATv2 comparison.",
        "",
        "## Frozen Contract",
        "",
        f"- Graph schema: `{manifest.get('graph_name')}`.",
        f"- Label scheme: `{manifest.get('label_scheme')}`.",
        f"- Split: `{manifest.get('split_id')}`.",
        "- Universe: measured-only rows, with `experiment_id=18` excluded by the locked split assignment.",
        "- Candidate edge feature set: `S5F2_energy`.",
        "- Auxiliary relation: `sequence_similar_to` remains label-free topology.",
        "",
        "## Feature Tables",
        "",
        "| feature_set | columns |",
        "| --- | ---: |",
    ]
    for feature_set, count in feature_counts.items():
        lines.append(f"| `{feature_set}` | {count} |")
    lines.extend(
        [
            "",
            "## Sprint 7B Attention Policy",
            "",
            "- Candidate edges use `S5F2_energy` in both the edge classifier and GATv2 `edge_attr`/`edge_dim` message-passing path.",
            "- Reverse candidate edges duplicate the candidate edge features.",
            "- `sequence_similar_to` edges are topology-only and use zero edge_attr vectors in GATv2.",
            "- Attention summaries are interpretation-only model artifacts, not biological causal evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
