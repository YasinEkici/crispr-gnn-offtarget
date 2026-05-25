"""Leakage-controlled typed graph artifact construction for Sprint 3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from crispr_gnn.data.labels import SCHEME_A_THRESHOLD
from crispr_gnn.data.schemas import (
    BINDING_ENERGY_FEATURES,
    GUIDE_KEY,
)
from crispr_gnn.data.splits import LABEL_COLUMN, SPLIT_COLUMN
from crispr_gnn.features.sequence import build_sequence_pair_encoding
from crispr_gnn.features.tabular import (
    FEATURE_SET_ORDER,
    FORBIDDEN_PREDICTIVE_COLUMNS,
    TrainOnlyPreprocessor,
    build_computed_nucleosome_features,
    build_experimental_epigenetic_features,
    build_feature_set,
    build_sequence_mismatch_features,
)
from crispr_gnn.graph.graph_schemas import (
    GRAPH_A,
    GRAPH_B,
    GRAPH_C,
    PHYSICAL_TARGET_KEY_FIELDS,
    GraphBuildConfig,
)


FEATURE_PREFIX = "feature__"
VIEWS = ("train", "val", "test")
AUXILIARY_VIEW_COMPOSITION = {
    "train": ["train"],
    "val": ["train", "val"],
    "test": ["train", "test"],
}


@dataclass
class GraphArtifact:
    name: str
    description: str
    nodes: dict[str, pd.DataFrame]
    relations: dict[str, pd.DataFrame]
    feature_tables: dict[str, pd.DataFrame]
    feature_sources: dict[str, list[str]]
    preprocessing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_graph_artifacts(assigned: pd.DataFrame, config: GraphBuildConfig | None = None) -> dict[str, GraphArtifact]:
    """Build Graph A/B/C from locked Sprint 2 assigned rows without model dependencies."""
    config = config or GraphBuildConfig()
    validate_main_graph_universe(assigned)
    assigned = _canonicalize_rows_by_id(assigned)
    graph_a = _build_graph_a(assigned, config)
    graph_b = _build_graph_b(assigned, graph_a, config)
    graph_c = _build_graph_c(assigned, graph_a.nodes["sgRNA"], config)
    artifacts = {artifact.name: artifact for artifact in (graph_a, graph_b, graph_c)}
    validate_graph_artifacts(assigned, artifacts)
    return artifacts


def _canonicalize_rows_by_id(assigned: pd.DataFrame) -> pd.DataFrame:
    order = sorted(
        range(assigned.shape[0]),
        key=lambda position: _sortable_id(str(assigned.iloc[position]["id"])),
    )
    return assigned.iloc[order].reset_index(drop=True)


def validate_main_graph_universe(assigned: pd.DataFrame) -> None:
    required = {
        "id",
        GUIDE_KEY,
        "grna_target_sequence",
        "target_sequence",
        "genome",
        "target_chr",
        "target_start",
        "target_end",
        "target_strand",
        "cleavage_freq",
        "measured",
        "experiment_id",
        LABEL_COLUMN,
        SPLIT_COLUMN,
    }
    missing = sorted(required.difference(assigned.columns))
    if missing:
        raise ValueError(f"Assigned graph rows are missing required columns: {missing}")
    if assigned["id"].isna().any() or not assigned["id"].is_unique:
        raise ValueError("Candidate-pair source id must be non-null and unique")
    if assigned[LABEL_COLUMN].isna().any() or assigned["cleavage_freq"].isna().any():
        raise ValueError("Graph rows must be label-eligible")
    expected_labels = (pd.to_numeric(assigned["cleavage_freq"]) > SCHEME_A_THRESHOLD).astype(int)
    if not expected_labels.equals(assigned[LABEL_COLUMN].astype(int)):
        raise ValueError("Graph labels differ from Scheme A")
    if (assigned["measured"] != 1).any():
        raise ValueError("Main graph universe contains measured=0 rows")
    if (assigned["experiment_id"] == 18).any():
        raise ValueError("Main graph universe contains experiment_id=18 rows")
    if set(assigned[SPLIT_COLUMN]) != set(VIEWS):
        raise ValueError("Main graph universe must contain train, val, and test rows")
    split_guides = {
        split: set(assigned.loc[assigned[SPLIT_COLUMN] == split, GUIDE_KEY].astype(str))
        for split in VIEWS
    }
    if (
        split_guides["train"] & split_guides["val"]
        or split_guides["train"] & split_guides["test"]
        or split_guides["val"] & split_guides["test"]
    ):
        raise ValueError("Main graph universe contains guide leakage across splits")


def _build_graph_a(assigned: pd.DataFrame, config: GraphBuildConfig) -> GraphArtifact:
    guide_nodes = _build_guide_nodes(assigned, max_length=config.max_length)
    target_nodes, target_map = _build_physical_target_nodes(assigned)
    candidate_edges = _candidate_edges(
        assigned,
        destination=target_map.to_numpy(),
        destination_column="target_node_id",
    )
    feature_tables, feature_sources, preprocessing = _graph_a_edge_features(assigned, max_length=config.max_length)
    return GraphArtifact(
        name=GRAPH_A,
        description="Minimal physical-target bipartite graph with row-varying context on candidate edges.",
        nodes={"sgRNA": guide_nodes, "physical_target_site": target_nodes},
        relations={"candidate_pair": candidate_edges},
        feature_tables=feature_tables,
        feature_sources=feature_sources,
        preprocessing=preprocessing,
        metadata={
            "target_key_fields": list(PHYSICAL_TARGET_KEY_FIELDS),
            "candidate_edge_key": "source row id",
            "candidate_relation_visibility": "filter_by_split_for_model_views",
            "context_placement": "candidate_pair_edge",
            "visibility_policy": config.visibility_policy,
        },
    )


def _build_graph_b(assigned: pd.DataFrame, graph_a: GraphArtifact, config: GraphBuildConfig) -> GraphArtifact:
    relation = _guide_similarity_edges(assigned, top_k=config.graph_b.top_k, max_length=config.max_length)
    return GraphArtifact(
        name=GRAPH_B,
        description="Graph A with bounded label-free guide-sequence similarity control edges.",
        nodes=graph_a.nodes,
        relations={
            "candidate_pair": graph_a.relations["candidate_pair"].copy(),
            "sequence_similar_to": relation,
        },
        feature_tables={},
        feature_sources={},
        preprocessing={},
        metadata={
            "base_graph": GRAPH_A,
            "inherits_feature_tables_from": GRAPH_A,
            "candidate_relation_visibility": "filter_by_split_for_model_views",
            "similarity_metric": "aligned_hamming_distance",
            "top_k": config.graph_b.top_k,
            "visibility_policy": config.visibility_policy,
            "auxiliary_relation_view_composition": AUXILIARY_VIEW_COMPOSITION,
            "role": "bounded_secondary_control",
        },
    )


def _build_graph_c(assigned: pd.DataFrame, guide_nodes: pd.DataFrame, config: GraphBuildConfig) -> GraphArtifact:
    observation_nodes, node_features, node_sources, preprocessing, transformed_context = _build_observation_nodes(assigned, config)
    candidate_edges = _candidate_edges(
        assigned,
        destination=assigned["id"].astype(str).to_numpy(),
        destination_column="target_observation_id",
    )
    pair_features, pair_sources = _observation_candidate_edge_features(assigned)
    context_relation = _context_similarity_edges(
        assigned,
        transformed_context,
        top_k=config.graph_c.top_k,
    )
    return GraphArtifact(
        name=GRAPH_C,
        description="Context-observation graph with train-fitted label-free context similarity.",
        nodes={"sgRNA": guide_nodes.copy(), "target_observation": observation_nodes},
        relations={"candidate_pair": candidate_edges, "context_similar_to": context_relation},
        feature_tables={"target_observation_features": node_features, "candidate_pair_features": pair_features},
        feature_sources={"target_observation_features": node_sources, "candidate_pair_features": pair_sources},
        preprocessing=preprocessing,
        metadata={
            "target_observation_key": "source row id",
            "candidate_relation_visibility": "filter_by_split_for_model_views",
            "similarity_metric": config.graph_c.metric,
            "top_k": config.graph_c.top_k,
            "context_placement": "target_observation_node",
            "visibility_policy": config.visibility_policy,
            "auxiliary_relation_view_composition": AUXILIARY_VIEW_COMPOSITION,
            "semantic_difference_from_graph_a": "target nodes represent observations rather than physical loci",
        },
    )


def _build_guide_nodes(assigned: pd.DataFrame, *, max_length: int) -> pd.DataFrame:
    consistency = assigned.groupby(GUIDE_KEY)["grna_target_sequence"].nunique(dropna=False)
    if (consistency > 1).any():
        raise ValueError("A guide ID maps to multiple guide sequences")
    rows = (
        assigned.sort_values([GUIDE_KEY, SPLIT_COLUMN])
        .drop_duplicates(GUIDE_KEY)[[GUIDE_KEY, "grna_target_sequence", SPLIT_COLUMN]]
        .copy()
    )
    rows["node_id"] = rows[GUIDE_KEY].astype(str)
    feature_values = np.vstack([_one_hot_sequence(sequence, max_length) for sequence in rows["grna_target_sequence"]])
    feature_names = [
        f"{FEATURE_PREFIX}guide_pos_{position:02d}_{base}"
        for position in range(max_length)
        for base in ("A", "C", "G", "T", "N")
    ]
    features = pd.DataFrame(feature_values, columns=feature_names, index=rows.index)
    return pd.concat([rows[["node_id", GUIDE_KEY, SPLIT_COLUMN]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def _build_physical_target_nodes(assigned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    targets = assigned[list(PHYSICAL_TARGET_KEY_FIELDS)].drop_duplicates().sort_values(list(PHYSICAL_TARGET_KEY_FIELDS)).copy()
    targets["node_id"] = [_physical_target_id(row) for _, row in targets.iterrows()]
    target_map = assigned.apply(_physical_target_id, axis=1)
    return targets[["node_id", *PHYSICAL_TARGET_KEY_FIELDS]].reset_index(drop=True), target_map


def _physical_target_id(row: pd.Series) -> str:
    return "|".join(str(row[field]) for field in PHYSICAL_TARGET_KEY_FIELDS)


def _candidate_edges(assigned: pd.DataFrame, *, destination: np.ndarray, destination_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "edge_id": assigned["id"].astype(str).to_numpy(),
            "source_sgrna_id": assigned[GUIDE_KEY].astype(str).to_numpy(),
            destination_column: destination,
            "label": assigned[LABEL_COLUMN].astype(int).to_numpy(),
            "split": assigned[SPLIT_COLUMN].astype(str).to_numpy(),
            "measured": assigned["measured"].astype(int).to_numpy(),
            "experiment_id": assigned["experiment_id"].astype(int).to_numpy(),
        }
    )


def _graph_a_edge_features(
    assigned: pd.DataFrame,
    *,
    max_length: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, Any]]:
    ids = assigned["id"].astype(str).reset_index(drop=True)
    feature_tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, list[str]] = {}

    encoded = build_sequence_pair_encoding(assigned, max_length=max_length).encoded.reshape(assigned.shape[0], -1)
    s1_names = [f"s1_pos_{position:02d}_channel_{channel:02d}" for position in range(max_length) for channel in range(11)]
    feature_tables["S1_pair"] = _keyed_features(ids, pd.DataFrame(encoded, columns=s1_names))
    sources["S1_pair"] = ["grna_target_sequence", "target_sequence"]

    preprocessing: dict[str, Any] = {}
    for feature_set in FEATURE_SET_ORDER:
        features = build_feature_set(assigned, feature_set)
        _assert_allowed_sources(features.columns.tolist())
        if feature_set == "F4":
            train_mask = assigned[SPLIT_COLUMN].eq("train")
            preprocessor = TrainOnlyPreprocessor(scale=False).fit(features.loc[train_mask])
            features = preprocessor.transform(features)
            preprocessing["F4"] = {
                "fit_scope": "train_only",
                "imputation": "median",
                "scaling": False,
                "feature_columns": features.columns.tolist(),
                "median_values": _imputer_statistics(preprocessor),
            }
        feature_tables[feature_set] = _keyed_features(ids, features.reset_index(drop=True))
        sources[feature_set] = features.columns.tolist()
    return feature_tables, sources, preprocessing


def _build_observation_nodes(
    assigned: pd.DataFrame,
    config: GraphBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], pd.DataFrame]:
    observations = pd.DataFrame(
        {
            "node_id": assigned["id"].astype(str).to_numpy(),
            "split": assigned[SPLIT_COLUMN].astype(str).to_numpy(),
        }
    )
    target_sequence = np.vstack([_one_hot_sequence(sequence, config.max_length) for sequence in assigned["target_sequence"]])
    sequence_names = [f"target_pos_{position:02d}_{base}" for position in range(config.max_length) for base in ("A", "C", "G", "T", "N")]
    sequence_features = pd.DataFrame(target_sequence, columns=sequence_names, index=assigned.index)

    context_raw = pd.concat(
        [
            build_experimental_epigenetic_features(assigned),
            build_computed_nucleosome_features(assigned),
        ],
        axis=1,
    )
    _assert_allowed_sources(context_raw.columns.tolist())
    train_mask = assigned[SPLIT_COLUMN].eq("train")
    imputer = SimpleImputer(strategy="median").fit(context_raw.loc[train_mask])
    imputed = pd.DataFrame(imputer.transform(context_raw), columns=context_raw.columns, index=assigned.index)
    scaler = StandardScaler().fit(imputed.loc[train_mask])
    transformed_context = pd.DataFrame(scaler.transform(imputed), columns=context_raw.columns, index=assigned.index)
    feature_values = pd.concat([sequence_features, transformed_context], axis=1)
    feature_table = _keyed_features(observations["node_id"], feature_values.reset_index(drop=True))
    preprocessing = {
        "target_observation_context": {
            "fit_scope": "train_only",
            "imputation": "median",
            "scaling": "standard",
            "similarity_feature_columns": context_raw.columns.tolist(),
            "median_values": {name: float(value) for name, value in zip(context_raw.columns, imputer.statistics_, strict=True)},
            "scale_mean": {name: float(value) for name, value in zip(context_raw.columns, scaler.mean_, strict=True)},
            "scale_std": {name: float(value) for name, value in zip(context_raw.columns, scaler.scale_, strict=True)},
        }
    }
    return (
        observations,
        feature_table,
        ["target_sequence", *context_raw.columns.tolist()],
        preprocessing,
        transformed_context.reset_index(drop=True),
    )


def _observation_candidate_edge_features(assigned: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mismatch = build_sequence_mismatch_features(assigned)
    keep = [
        column
        for column in mismatch.columns
        if column.startswith("mismatch_pos_") or column in {"aligned_length", "length_delta", "mismatch_count", "mismatch_rate"}
    ]
    pair_features = pd.concat([mismatch[keep], assigned[BINDING_ENERGY_FEATURES].apply(pd.to_numeric)], axis=1)
    _assert_allowed_sources(pair_features.columns.tolist())
    return _keyed_features(assigned["id"].astype(str).reset_index(drop=True), pair_features.reset_index(drop=True)), pair_features.columns.tolist()


def _guide_similarity_edges(assigned: pd.DataFrame, *, top_k: int, max_length: int) -> pd.DataFrame:
    guide_rows = (
        assigned.sort_values([GUIDE_KEY, SPLIT_COLUMN])
        .drop_duplicates(GUIDE_KEY)[[GUIDE_KEY, "grna_target_sequence", SPLIT_COLUMN]]
        .copy()
    )
    guide_rows[GUIDE_KEY] = guide_rows[GUIDE_KEY].astype(str)
    train = guide_rows.loc[guide_rows[SPLIT_COLUMN] == "train"].sort_values(GUIDE_KEY)
    relations = []
    for view in VIEWS:
        queries = train if view == "train" else guide_rows.loc[guide_rows[SPLIT_COLUMN] == view].sort_values(GUIDE_KEY)
        relations.extend(
            _rank_guide_neighbors(queries, train, view=view, top_k=top_k, max_length=max_length)
        )
    return _symmetrized_relation(
        relations,
        source_column="source_sgrna_id",
        target_column="target_sgrna_id",
    )


def _rank_guide_neighbors(
    queries: pd.DataFrame,
    train: pd.DataFrame,
    *,
    view: str,
    top_k: int,
    max_length: int,
) -> list[dict[str, object]]:
    rows = []
    for _, query in queries.iterrows():
        candidates = []
        query_id = str(query[GUIDE_KEY])
        for _, reference in train.iterrows():
            reference_id = str(reference[GUIDE_KEY])
            if view == "train" and query_id == reference_id:
                continue
            distance = _hamming_distance(query["grna_target_sequence"], reference["grna_target_sequence"], max_length)
            candidates.append((distance, _sortable_id(reference_id), reference_id))
        for distance, _, reference_id in sorted(candidates)[: min(top_k, len(candidates))]:
            rows.append(
                {
                    "view": view,
                    "source_sgrna_id": query_id,
                    "target_sgrna_id": reference_id,
                    "distance": float(distance),
                    "source_split": view,
                    "target_split": "train",
                }
            )
    return rows


def _context_similarity_edges(assigned: pd.DataFrame, transformed_context: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    ids = assigned["id"].astype(str).reset_index(drop=True)
    splits = assigned[SPLIT_COLUMN].astype(str).reset_index(drop=True)
    train_positions = np.flatnonzero(splits.eq("train").to_numpy())
    train_ids = ids.iloc[train_positions].to_numpy()
    train_features = transformed_context.iloc[train_positions].to_numpy(dtype=float)
    model = NearestNeighbors(metric="euclidean")
    model.fit(train_features)

    relations: list[dict[str, object]] = []
    for view in VIEWS:
        positions = np.flatnonzero(splits.eq(view).to_numpy())
        query_features = transformed_context.iloc[positions].to_numpy(dtype=float)
        query_ids = ids.iloc[positions].to_numpy()
        neighbor_count = min(top_k + 1 if view == "train" else top_k, train_features.shape[0])
        distances, _ = model.kneighbors(query_features, n_neighbors=neighbor_count)
        for query_id, query_features_row, row_distances in zip(query_ids, query_features, distances, strict=True):
            boundary = np.nextafter(float(row_distances[-1]), float("inf"))
            candidate_distances, candidate_indexes = model.radius_neighbors(
                query_features_row.reshape(1, -1),
                radius=boundary,
                sort_results=False,
            )
            selected = []
            for distance, index in zip(candidate_distances[0], candidate_indexes[0], strict=True):
                reference_id = str(train_ids[index])
                if view == "train" and str(query_id) == reference_id:
                    continue
                selected.append((float(distance), _sortable_id(reference_id), reference_id))
            for distance, _, reference_id in sorted(selected)[:top_k]:
                relations.append(
                    {
                        "view": view,
                        "source_observation_id": str(query_id),
                        "target_observation_id": reference_id,
                        "distance": distance,
                        "source_split": view,
                        "target_split": "train",
                    }
                )
    return _symmetrized_relation(
        relations,
        source_column="source_observation_id",
        target_column="target_observation_id",
    )


def _symmetrized_relation(
    relations: list[dict[str, object]],
    *,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for row in relations:
        rows.append(row)
        reversed_row = row.copy()
        reversed_row[source_column] = row[target_column]
        reversed_row[target_column] = row[source_column]
        reversed_row["source_split"] = row["target_split"]
        reversed_row["target_split"] = row["source_split"]
        rows.append(reversed_row)
    relation = pd.DataFrame(rows)
    if relation.empty:
        return relation
    return (
        relation.sort_values(["view", source_column, "distance", target_column])
        .drop_duplicates(["view", source_column, target_column])
        .reset_index(drop=True)
    )


def validate_graph_artifacts(assigned: pd.DataFrame, artifacts: dict[str, GraphArtifact]) -> None:
    expected_edge_ids = set(assigned["id"].astype(str))
    split_by_edge = dict(zip(assigned["id"].astype(str), assigned[SPLIT_COLUMN].astype(str), strict=True))
    for artifact in artifacts.values():
        candidates = artifact.relations["candidate_pair"]
        if set(candidates["edge_id"]) != expected_edge_ids or candidates["edge_id"].duplicated().any():
            raise ValueError(f"{artifact.name} candidate edges do not preserve locked input rows")
        if not candidates["edge_id"].map(split_by_edge).equals(candidates["split"]):
            raise ValueError(f"{artifact.name} candidate edge splits drift from locked assignment")
        if (candidates["measured"] != 1).any() or (candidates["experiment_id"] == 18).any():
            raise ValueError(f"{artifact.name} violates measured-only or experiment exclusion policy")
        for name, source_columns in artifact.feature_sources.items():
            _assert_allowed_sources(source_columns, table_name=f"{artifact.name}:{name}")

    physical_features = set(artifacts[GRAPH_A].nodes["physical_target_site"].columns)
    if physical_features - {"node_id", *PHYSICAL_TARGET_KEY_FIELDS}:
        raise ValueError("Graph A physical target nodes contain predictive/context feature columns")
    _validate_inductive_similarity(
        artifacts[GRAPH_B].relations["sequence_similar_to"],
        split_by_node=_split_lookup(artifacts[GRAPH_B].nodes["sgRNA"]),
        source_column="source_sgrna_id",
        target_column="target_sgrna_id",
    )
    _validate_inductive_similarity(
        artifacts[GRAPH_C].relations["context_similar_to"],
        split_by_node=_split_lookup(artifacts[GRAPH_C].nodes["target_observation"]),
        source_column="source_observation_id",
        target_column="target_observation_id",
    )


def _validate_inductive_similarity(
    relation: pd.DataFrame,
    *,
    split_by_node: dict[str, str],
    source_column: str,
    target_column: str,
) -> None:
    for view in VIEWS:
        part = relation.loc[relation["view"] == view]
        for source, target in part[[source_column, target_column]].itertuples(
            index=False, name=None
        ):
            source_split = split_by_node[str(source)]
            target_split = split_by_node[str(target)]
            allowed = {"train"} if view == "train" else {"train", view}
            if source_split not in allowed or target_split not in allowed:
                raise ValueError(f"Similarity view '{view}' exposes a disallowed split")
            if view != "train" and source_split == view and target_split == view:
                raise ValueError(f"Similarity view '{view}' connects held-out nodes to each other")


def write_graph_artifacts(
    artifacts: dict[str, GraphArtifact],
    *,
    artifact_dir: str | Path,
    report_path: str | Path,
    split_id: str,
) -> tuple[Path, list[Path]]:
    base = Path(artifact_dir)
    written: list[Path] = []
    for graph_name, artifact in artifacts.items():
        graph_dir = base / graph_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        for node_type, table in artifact.nodes.items():
            path = graph_dir / f"nodes_{node_type}.parquet"
            table.to_parquet(path, index=False)
            written.append(path)
        for relation_type, table in artifact.relations.items():
            path = graph_dir / f"relation_{relation_type}.parquet"
            table.to_parquet(path, index=False)
            written.append(path)
        for feature_name, table in artifact.feature_tables.items():
            path = graph_dir / f"features_{feature_name}.parquet"
            table.to_parquet(path, index=False)
            written.append(path)
        manifest = {
            "graph_name": artifact.name,
            "description": artifact.description,
            "label_scheme": "scheme_a",
            "label_definition": "cleavage_freq > 1e-05",
            "split_id": split_id,
            "nodes": {name: int(table.shape[0]) for name, table in artifact.nodes.items()},
            "relations": {name: int(table.shape[0]) for name, table in artifact.relations.items()},
            "feature_tables": {name: int(table.shape[1] - 1) for name, table in artifact.feature_tables.items()},
            "feature_sources": artifact.feature_sources,
            "preprocessing": artifact.preprocessing,
            "metadata": artifact.metadata,
        }
        manifest_path = graph_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(manifest_path)
    report = _write_graph_report(artifacts, Path(report_path), split_id=split_id)
    return report, written


def _write_graph_report(artifacts: dict[str, GraphArtifact], report_path: Path, *, split_id: str) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = artifacts[GRAPH_A].relations["candidate_pair"]
    split_summary = (
        candidates.groupby("split", sort=False)
        .agg(rows=("edge_id", "size"), positives=("label", "sum"))
        .reset_index()
    )
    split_summary["negatives"] = split_summary["rows"] - split_summary["positives"]
    split_summary["positive_rate"] = split_summary["positives"] / split_summary["rows"]
    lines = [
        "# Sprint 3 Graph Schema Report",
        "",
        "Sprint 3 constructs graph datasets and leakage audits only. No GNN model performance is reported here.",
        "",
        "## Preserved Evaluation Contract",
        "",
        f"- Locked split: `{split_id}`.",
        "- Label: Scheme A, `cleavage_freq > 1e-5`.",
        "- Main graph universe is measured-only and excludes `experiment_id=18`.",
        "- Primary downstream comparison remains `xgboost_unweighted / F4` (test AUPRC `0.992522`).",
        "- Primary visibility policy: strict-inductive; held-out similarity is computed only against training references.",
        "",
        "## Candidate Edge Summary",
        "",
        _markdown_table(split_summary),
        "",
        "## Graph Artifacts",
        "",
        "| graph | node tables | relation tables | feature placement | role |",
        "| --- | --- | --- | --- | --- |",
    ]
    for graph_name in (GRAPH_A, GRAPH_B, GRAPH_C):
        artifact = artifacts[graph_name]
        node_counts = ", ".join(f"{name}={len(table)}" for name, table in artifact.nodes.items())
        relation_counts = ", ".join(f"{name}={len(table)}" for name, table in artifact.relations.items())
        placement = str(artifact.metadata.get("context_placement", artifact.metadata.get("inherits_feature_tables_from", "Graph A")))
        role = str(artifact.metadata.get("role", "primary_schema"))
        lines.append(f"| `{graph_name}` | {node_counts} | {relation_counts} | {placement} | {role} |")
    lines.extend(
        [
            "",
            "## Schema Decisions",
            "",
            "- Graph A represents genome-aware physical target loci; row-varying sequence/context is carried by candidate-edge feature tables.",
            "- Graph B inherits Graph A and adds guide-sequence similarity as a bounded control relation.",
            "- Graph C represents target observations keyed by source row `id`; context similarity is based on train-fitted F3/F4 context only.",
            "- Graph C is not merely Graph A with additional edges: the target semantics intentionally differ.",
            "- Large feature tables are generated under ignored `data/processed/graphs/sprint3/`; this report is the tracked Sprint 3 artifact.",
            "",
            "## Leakage Controls",
            "",
            "- Candidate labels are recomputed and validated against Scheme A.",
            "- Candidate edge membership exactly preserves the locked split assignment.",
            "- Candidate relations are stored with split labels; a later training loader must expose only `split=train` supervised edges during fitting.",
            "- Graph A physical target nodes contain metadata identity only, not variable F3/F4 context.",
            "- Graph B similarity uses guide sequence only.",
            "- Graph C context preprocessing is fitted on training observations only; validation/test observations connect only to training references.",
            "- Auxiliary relation rows are stored as visibility fragments: a validation/test view is the union of its rows with training-view rows.",
            "- No test metric or label is used for graph construction or schema selection.",
            "",
            "## Sprint 4 Handoff",
            "",
            "Sprint 4 may load these typed artifacts into PyTorch Geometric `HeteroData` after an explicit dependency decision. Train Graph A first, evaluate the primary context-enriched Graph C second, and use Graph B only as a bounded control run. Every downstream comparison must preserve the locked protocol and compare to `xgboost_unweighted / F4`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _keyed_features(keys: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    renamed = features.copy()
    renamed.columns = [f"{FEATURE_PREFIX}{column}" for column in renamed.columns]
    return pd.concat([pd.DataFrame({"record_id": keys.to_numpy()}), renamed.reset_index(drop=True)], axis=1)


def _assert_allowed_sources(source_columns: list[str], *, table_name: str = "feature table") -> None:
    forbidden = sorted(set(source_columns).intersection(FORBIDDEN_PREDICTIVE_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden predictive columns in {table_name}: {forbidden}")


def _imputer_statistics(preprocessor: TrainOnlyPreprocessor) -> dict[str, float]:
    imputer = preprocessor.pipeline.named_steps["imputer"]
    columns = preprocessor.feature_columns or []
    return {name: float(value) for name, value in zip(columns, imputer.statistics_, strict=True)}


def _one_hot_sequence(sequence: object, max_length: int) -> np.ndarray:
    bases = ("A", "C", "G", "T", "N")
    lookup = {base: index for index, base in enumerate(bases)}
    text = "" if pd.isna(sequence) else str(sequence).upper()
    output = np.zeros((max_length, len(bases)), dtype=np.float32)
    for position in range(max_length):
        base = text[position] if position < len(text) and text[position] in lookup else "N"
        output[position, lookup[base]] = 1.0
    return output.reshape(-1)


def _hamming_distance(left: object, right: object, max_length: int) -> int:
    a = "" if pd.isna(left) else str(left).upper()
    b = "" if pd.isna(right) else str(right).upper()
    return sum(
        (a[position] if position < len(a) else "N") != (b[position] if position < len(b) else "N")
        for position in range(max_length)
    )


def _sortable_id(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (2**31 - 1, value)


def _split_lookup(nodes: pd.DataFrame) -> dict[str, str]:
    return dict(zip(nodes["node_id"].astype(str), nodes["split"].astype(str), strict=True))


def _markdown_table(df: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(str(column) for column in df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for row in df.itertuples(index=False):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
