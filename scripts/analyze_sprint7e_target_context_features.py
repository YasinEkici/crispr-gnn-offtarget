from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.features.target_context import (  # noqa: E402
    TARGET_CONTEXT_EXPECTED_COUNTS,
    TARGET_CONTEXT_FAMILY_ORDER,
    target_context_family_counts,
    target_context_feature_family,
    validate_target_context_feature_names,
)
from crispr_gnn.graph.graph_schemas import GRAPH_C  # noqa: E402
from crispr_gnn.graph.pyg_dataset import LABEL_SCHEME, SPLIT_ID, VISIBILITY_POLICY  # noqa: E402


DEFAULT_GRAPH_C_DIR = Path("data/processed/graphs/sprint5b/graph_c_context_observation")
DEFAULT_OUTPUT_DIR = Path("outputs/sprint7e/context_feature_profiling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Sprint 7E Graph C target-observation feature families.")
    parser.add_argument("--graph-c-dir", default=str(DEFAULT_GRAPH_C_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run_sprint7e_target_context_feature_profiling(
        graph_c_dir=ROOT / args.graph_c_dir,
        output_dir=ROOT / args.output_dir,
        write_figures=not args.skip_figures,
    )
    print(f"Sprint 7E context-feature profiling output: {_relative(output_dir)}")
    print(f"Report: {_relative(output_dir / 'sprint7e_context_feature_profile_report.md')}")
    return 0


def run_sprint7e_target_context_feature_profiling(
    *,
    graph_c_dir: Path,
    output_dir: Path,
    write_figures: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((graph_c_dir / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    nodes = pd.read_parquet(graph_c_dir / "nodes_target_observation.parquet")
    features = pd.read_parquet(graph_c_dir / "features_target_observation_features.parquet")
    candidate = pd.read_parquet(graph_c_dir / "relation_candidate_pair.parquet")

    feature_columns = [column for column in features.columns if column.startswith("feature__")]
    validate_target_context_feature_names(feature_columns)

    merged = _joined_feature_frame(nodes, features, candidate)
    family_map = _family_map(feature_columns)
    group_summary = _group_summary(merged, family_map)
    distribution = _distribution_by_split_label(merged, family_map)

    family_map_path = output_dir / "sprint7e_context_feature_family_map.csv"
    summary_path = output_dir / "sprint7e_context_feature_group_summary.csv"
    distribution_path = output_dir / "sprint7e_context_feature_distribution_by_split_label.csv"
    report_path = output_dir / "sprint7e_context_feature_profile_report.md"
    manifest_path = output_dir / "sprint7e_context_feature_profile_manifest.json"

    family_map.to_csv(family_map_path, index=False)
    group_summary.to_csv(summary_path, index=False)
    distribution.to_csv(distribution_path, index=False)

    figure_paths: list[Path] = []
    if write_figures:
        figure_paths = _write_figures(group_summary, distribution, figures_dir)

    report_path.write_text(
        _write_report(
            manifest=manifest,
            family_map=family_map,
            group_summary=group_summary,
            distribution=distribution,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    _write_manifest(
        manifest_path,
        graph_c_dir=graph_c_dir,
        outputs=[family_map_path, summary_path, distribution_path, report_path, *figure_paths],
        graph_manifest=manifest,
    )
    return output_dir


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("graph_name") != GRAPH_C:
        raise ValueError("Sprint 7E profiling requires Graph C artifacts")
    if manifest.get("split_id") != SPLIT_ID or manifest.get("label_scheme") != LABEL_SCHEME:
        raise ValueError("Sprint 7E profiling requires frozen Scheme A / sprint2_main_seed42 artifacts")
    if manifest.get("metadata", {}).get("visibility_policy") != VISIBILITY_POLICY:
        raise ValueError("Sprint 7E profiling requires strict-inductive artifacts")
    feature_tables = manifest.get("feature_tables", {})
    if int(feature_tables.get("target_observation_features", 0)) != sum(TARGET_CONTEXT_EXPECTED_COUNTS.values()):
        raise ValueError("Graph C target_observation_features count drift")
    if int(feature_tables.get("S5F2_energy", 0)) != 268:
        raise ValueError("Sprint 7E expects the Sprint 5B Graph C S5F2 artifact")


def _joined_feature_frame(nodes: pd.DataFrame, features: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required_node = {"node_id", "split"}
    required_feature = {"record_id"}
    required_candidate = {"edge_id", "label", "split", "measured", "experiment_id"}
    _require_columns(nodes, required_node, "nodes_target_observation")
    _require_columns(features, required_feature, "features_target_observation_features")
    _require_columns(candidate, required_candidate, "relation_candidate_pair")
    frame = nodes[["node_id", "split"]].copy()
    frame = frame.merge(
        features,
        left_on="node_id",
        right_on="record_id",
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(
        candidate[["edge_id", "label", "split", "measured", "experiment_id"]].rename(columns={"split": "candidate_split"}),
        left_on="node_id",
        right_on="edge_id",
        how="left",
        validate="one_to_one",
    )
    if frame[[column for column in features.columns if column.startswith("feature__")]].isna().any().any():
        raise ValueError("Target-observation feature table contains missing values after preprocessing")
    if not frame["split"].astype(str).equals(frame["candidate_split"].astype(str)):
        raise ValueError("Target-observation split drift relative to candidate relation")
    if (frame["measured"] != 1).any() or (frame["experiment_id"] == 18).any():
        raise ValueError("Sprint 7E profiling artifact violates measured-only / experiment exclusion contract")
    return frame


def _family_map(feature_columns: list[str]) -> pd.DataFrame:
    counts = target_context_family_counts(feature_columns)
    rows = []
    for index, column in enumerate(feature_columns):
        family = target_context_feature_family(column)
        rows.append(
            {
                "feature_index": index,
                "feature_column": column,
                "source_feature_name": column.removeprefix("feature__"),
                "target_context_family": family,
                "family_expected_count": TARGET_CONTEXT_EXPECTED_COUNTS.get(family),
                "family_observed_count": counts.get(family),
            }
        )
    return pd.DataFrame(rows)


def _group_summary(frame: pd.DataFrame, family_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in TARGET_CONTEXT_FAMILY_ORDER:
        columns = family_map.loc[family_map["target_context_family"] == family, "feature_column"].tolist()
        values = frame[columns]
        rows.append(
            {
                "target_context_family": family,
                "feature_columns": len(columns),
                "rows": len(frame),
                "missing_values_after_preprocessing": int(values.isna().sum().sum()),
                "feature_abs_mean": float(values.abs().to_numpy().mean()),
                "feature_abs_std": float(values.abs().to_numpy().std()),
                "feature_mean": float(values.to_numpy().mean()),
                "feature_std": float(values.to_numpy().std()),
                "nonzero_fraction": float((values.to_numpy() != 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _distribution_by_split_label(frame: pd.DataFrame, family_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, label), group in frame.groupby(["split", "label"], sort=True):
        for family in TARGET_CONTEXT_FAMILY_ORDER:
            columns = family_map.loc[family_map["target_context_family"] == family, "feature_column"].tolist()
            values = group[columns].to_numpy(dtype=float)
            rows.append(
                {
                    "split": split,
                    "label": int(label),
                    "target_context_family": family,
                    "rows": int(len(group)),
                    "feature_columns": int(len(columns)),
                    "feature_abs_mean": float(abs(values).mean()),
                    "feature_mean": float(values.mean()),
                    "feature_std": float(values.std()),
                    "feature_min": float(values.min()),
                    "feature_max": float(values.max()),
                    "nonzero_fraction": float((values != 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_figures(group_summary: pd.DataFrame, distribution: pd.DataFrame, figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    path = figures_dir / "sprint7e_context_feature_group_missingness.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    group_summary.set_index("target_context_family")["missing_values_after_preprocessing"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Missing values after preprocessing")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figures_dir / "sprint7e_context_feature_group_distribution.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = distribution.pivot_table(
        index="target_context_family",
        columns=["split", "label"],
        values="feature_abs_mean",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean absolute feature value")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _write_report(
    *,
    manifest: dict[str, Any],
    family_map: pd.DataFrame,
    group_summary: pd.DataFrame,
    distribution: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    counts = family_map["target_context_family"].value_counts().reindex(TARGET_CONTEXT_FAMILY_ORDER).reset_index()
    counts.columns = ["target_context_family", "feature_columns"]
    return f"""# Sprint 7E Target-Observation Feature Profiling Report

## Contract

- Graph schema: `{manifest.get("graph_name")}`.
- Label/split: frozen `{LABEL_SCHEME}` / `{SPLIT_ID}`.
- Visibility policy: `{manifest.get("metadata", {}).get("visibility_policy")}`.
- This slice performs feature-family profiling only. It does not train models and does not select runs from test performance.

## Family Counts

{_markdown_table(counts)}

## Group Summary

{_markdown_table(group_summary)}

## Split/Label Distribution Summary

{_markdown_table(distribution.head(24))}

## Run Matrix Freeze

Slice 2 remains source-family-defined: mask target sequence, experimental epigenetic, computed nucleosome aggregates, computed nucleosome missingness, and all nonsequence context. These runs are not selected or reordered from test diagnostics.

## Figure Index

{chr(10).join(f"- `{_relative(path)}`" for path in figure_paths)}
"""


def _write_manifest(path: Path, *, graph_c_dir: Path, outputs: list[Path], graph_manifest: dict[str, Any]) -> None:
    payload = {
        "manifest_type": "sprint7e_target_context_feature_profile_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "graph_c_dir": _relative(graph_c_dir),
        "graph_name": graph_manifest.get("graph_name"),
        "split_id": graph_manifest.get("split_id"),
        "label_scheme": graph_manifest.get("label_scheme"),
        "feature_tables": graph_manifest.get("feature_tables", {}),
        "outputs": [_relative(path) for path in outputs],
        "canonical_slice2_run_ids": [
            "S7E_R1_mask_target_sequence",
            "S7E_R2_mask_experimental_epigenetic",
            "S7E_R3_mask_computed_nucleosome_aggregates",
            "S7E_R4_mask_computed_nucleosome_missingness",
            "S7E_R5_mask_all_nonsequence_context",
        ],
        "no_training_performed": True,
        "no_test_performance_selection": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_columns(table: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
