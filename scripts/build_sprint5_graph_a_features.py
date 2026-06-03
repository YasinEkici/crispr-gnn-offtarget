from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.features.sprint5 import build_sprint5_feature_tables  # noqa: E402
from crispr_gnn.graph.graph_builder import build_graph_artifacts, write_graph_artifacts  # noqa: E402
from crispr_gnn.graph.graph_schemas import GRAPH_A, GraphBuildConfig  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sprint 5 Graph A feature-ablation artifacts.")
    parser.add_argument("--data-config", default="configs/data/mak2022.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--schema-config",
        default="configs/sweeps/graph_schema_ablation.yaml",
        help="Path to Sprint 3 graph schema YAML, used for split and representation settings.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="data/processed/graphs/sprint5",
        help="Output graph artifact directory. Contains Graph A with S5F0-S5F5 feature tables.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/sprint5/graph_a_feature_ablation_artifact_report.md",
        help="Output artifact report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_yaml(args.data_config)
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
    graph_a = artifacts[GRAPH_A]
    feature_tables, feature_sources, preprocessing = build_sprint5_feature_tables(
        assigned,
        max_length=graph_config.max_length,
        scale=False,
    )
    graph_a.feature_tables.update(feature_tables)
    graph_a.feature_sources.update(feature_sources)
    graph_a.preprocessing["sprint5_feature_ablation"] = preprocessing
    graph_a.metadata["sprint5_feature_ablation"] = {
        "role": "primary_fixed_topology_feature_ablation",
        "topology": "Graph A unchanged",
        "feature_sets": list(feature_tables),
        "s5f0_policy": "guide_target_sequence_one_hot_without_mismatch_channel",
    }

    artifact_dir = ROOT / args.artifact_dir
    report_path, written = write_graph_artifacts(
        {GRAPH_A: graph_a},
        artifact_dir=artifact_dir,
        report_path=ROOT / args.report_path,
        split_id=split.config.split_id,
    )
    manifest_path = artifact_dir / GRAPH_A / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_sprint5_artifact_report(
        report_path,
        manifest=manifest,
        source_report=report_path,
    )

    print(f"Sprint 5 Graph A artifact report written: {report_path.relative_to(ROOT)}")
    print(f"Sprint 5 Graph A artifact directory: {(artifact_dir / GRAPH_A).relative_to(ROOT)}")
    print(f"Feature tables: {', '.join(feature_tables)}")
    print(f"Total artifact files written: {len(written)}")
    return 0


def _write_sprint5_artifact_report(path: Path, *, manifest: dict[str, Any], source_report: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_counts = manifest.get("feature_tables", {})
    lines = [
        "# Sprint 5 Graph A Feature-Ablation Artifact Report",
        "",
        "Sprint 5 keeps Graph A topology fixed and adds candidate-edge feature tables for S5F0-S5F5.",
        "",
        "## Frozen Contract",
        "",
        f"- Graph schema: `{manifest.get('graph_name')}`.",
        f"- Label scheme: `{manifest.get('label_scheme')}`.",
        f"- Split: `{manifest.get('split_id')}`.",
        "- Universe: measured-only rows, with `experiment_id=18` excluded by the locked split assignment.",
        "- Topology: unchanged Graph A physical-target topology; no Graph C observation/context topology is used.",
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
            "## Notes",
            "",
            "- `S5F0_seq` is stricter than Sprint 4 `S1_pair`; it excludes the explicit mismatch channel.",
            "- Each Sprint 5 feature table uses train-only median imputation, no scaling, matching Sprint 4 Graph A edge-feature scale policy.",
            f"- Base graph writer report path: `{source_report.relative_to(ROOT)}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
