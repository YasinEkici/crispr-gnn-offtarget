from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.graph import GraphBuildConfig, build_graph_artifacts, write_graph_artifacts  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-controlled Sprint 3 graph artifacts.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument(
        "--schema-config",
        default="configs/sweeps/graph_schema_ablation.yaml",
        help="Path to Sprint 3 graph schema YAML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_yaml(args.config)
    schema_mapping = load_yaml(args.schema_config)
    graph_config = GraphBuildConfig.from_mapping(schema_mapping)
    dataset = data_config.get("dataset", {})
    raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
    split_path = ROOT / Path(str(schema_mapping.get("split_manifest", "outputs/splits/sprint2_guides.json")))
    if not raw_path.exists():
        print(f"Dataset not found: {raw_path}")
        return 1
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    raw = pd.read_parquet(raw_path)
    split = load_split_manifest(split_path)
    assigned = assign_measured_splits(raw, split)
    artifacts = build_graph_artifacts(assigned, graph_config)
    report_path, written = write_graph_artifacts(
        artifacts,
        artifact_dir=ROOT / graph_config.artifact_dir,
        report_path=ROOT / graph_config.report_path,
        split_id=split.config.split_id,
    )

    print(f"Graph report written: {report_path.relative_to(ROOT)}")
    print(f"Artifact files written under: {graph_config.artifact_dir}")
    for artifact in artifacts.values():
        node_counts = ", ".join(f"{name}={len(table)}" for name, table in artifact.nodes.items())
        relation_counts = ", ".join(f"{name}={len(table)}" for name, table in artifact.relations.items())
        print(f"{artifact.name}: nodes [{node_counts}], relations [{relation_counts}]")
    print(f"Total artifact files written: {len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
