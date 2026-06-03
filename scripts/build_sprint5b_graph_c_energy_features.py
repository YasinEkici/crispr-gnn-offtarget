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
from crispr_gnn.graph.graph_schemas import GRAPH_A, GRAPH_C, GraphBuildConfig  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sprint 5B Graph C energy-sensitivity artifacts.")
    parser.add_argument("--data-config", default="configs/data/mak2022.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--schema-config",
        default="configs/sweeps/graph_schema_ablation.yaml",
        help="Path to Sprint 3 graph schema YAML, used for split and representation settings.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="data/processed/graphs/sprint5b",
        help="Output graph artifact directory. Contains Graph A reference plus Graph C with S5F2 energy edge features.",
    )
    parser.add_argument(
        "--report-path",
        default="outputs/sprint5b/graph_c_energy_sensitivity_artifact_report.md",
        help="Output artifact report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_yaml(args.data_config)
    schema_mapping = load_yaml(args.schema_config)
    graph_config = GraphBuildConfig.from_mapping(schema_mapping)
    dataset = data_config.get("dataset", {})
    dataset_path = _resolve_dataset_path(dataset)
    split_path = ROOT / Path(str(schema_mapping.get("split_manifest", "outputs/splits/sprint2_guides.json")))
    if not split_path.exists():
        print(f"Split manifest not found: {split_path}")
        return 1

    print(f"Dataset path: {dataset_path.relative_to(ROOT)}")
    raw = pd.read_parquet(dataset_path)
    split = load_split_manifest(split_path)
    assigned = assign_measured_splits(raw, split)
    artifacts = build_graph_artifacts(assigned, graph_config)
    graph_c = artifacts[GRAPH_C]

    feature_tables, feature_sources, preprocessing = build_sprint5_feature_tables(
        assigned,
        max_length=graph_config.max_length,
        scale=False,
    )
    graph_c.feature_tables["S5F2_energy"] = feature_tables["S5F2_energy"]
    graph_c.feature_sources["S5F2_energy"] = feature_sources["S5F2_energy"]
    graph_c.preprocessing["sprint5b_edge_feature_sensitivity"] = {
        "fit_scope": "train_only",
        "feature_set": "S5F2_energy",
        "preprocessing": preprocessing["S5F2_energy"],
    }
    graph_c.metadata["sprint5b_energy_sensitivity"] = {
        "role": "secondary_sensitivity_not_primary_feature_ablation",
        "topology": "Graph C context-similarity topology unchanged",
        "target_semantics": "target_observation_context_encoder unchanged",
        "candidate_edge_feature_set": "S5F2_energy",
        "interpretation_boundary": (
            "This is not a clean primary feature ablation because Graph C already "
            "encodes context through target_observation nodes and context_similar_to topology."
        ),
    }

    artifact_dir = ROOT / args.artifact_dir
    report_path, written = write_graph_artifacts(
        {GRAPH_A: artifacts[GRAPH_A], GRAPH_C: graph_c},
        artifact_dir=artifact_dir,
        report_path=ROOT / args.report_path,
        split_id=split.config.split_id,
    )
    manifest_path = artifact_dir / GRAPH_C / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_sprint5b_artifact_report(report_path, manifest=manifest)

    print(f"Sprint 5B Graph C artifact report written: {report_path.relative_to(ROOT)}")
    print(f"Sprint 5B artifact directory: {artifact_dir.relative_to(ROOT)}")
    print(f"Graph C feature tables: {', '.join(manifest.get('feature_tables', {}))}")
    print(f"Total artifact files written: {len(written)}")
    return 0


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


def _write_sprint5b_artifact_report(path: Path, *, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_counts = manifest.get("feature_tables", {})
    lines = [
        "# Sprint 5B Graph C Energy-Sensitivity Artifact Report",
        "",
        "Sprint 5B keeps Graph C topology and target-observation semantics fixed, then adds the Sprint 5 `S5F2_energy` candidate-edge feature table as a secondary sensitivity.",
        "",
        "## Interpretation Boundary",
        "",
        "- This is not the primary Sprint 5 feature ablation.",
        "- Graph C already encodes context through target-observation nodes and context-similarity edges.",
        "- The result may compare Graph C context representation with the best Graph A energy-focused candidate-edge setting, but must not be described as a clean feature-family isolation.",
        "",
        "## Frozen Contract",
        "",
        f"- Graph schema: `{manifest.get('graph_name')}`.",
        f"- Label scheme: `{manifest.get('label_scheme')}`.",
        f"- Split: `{manifest.get('split_id')}`.",
        "- Universe: measured-only rows, with `experiment_id=18` excluded by the locked split assignment.",
        "- Candidate edge feature set: `S5F2_energy`.",
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
            "- `target_observation_features` and `context_similar_to` remain full Graph C context representations from the Sprint 3 schema.",
            "- `S5F2_energy` contains guide/target sequence one-hot, sequence/mismatch features, and `energy_1`-`energy_5`; binding-energy features are not epigenetic features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
