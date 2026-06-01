"""PyG materialization of validated Sprint 3 typed graph artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from crispr_gnn.features.tabular import FORBIDDEN_PREDICTIVE_COLUMNS
from crispr_gnn.graph.graph_schemas import (
    GRAPH_A,
    GRAPH_B,
    GRAPH_C,
    PHYSICAL_TARGET_KEY_FIELDS,
)


LABEL_SCHEME = "scheme_a"
LABEL_DEFINITION = "cleavage_freq > 1e-05"
SPLIT_ID = "sprint2_main_seed42"
VISIBILITY_POLICY = "strict_inductive_primary"
VIEWS = ("train", "val", "test")
VIEW_COMPOSITION = {
    "train": ("train",),
    "val": ("train", "val"),
    "test": ("train", "test"),
}
CANDIDATE_SPLIT_COUNTS = {"train": 8010, "val": 1734, "test": 1702}
CANONICAL_COUNTS = {
    GRAPH_A: {
        "nodes": {"sgRNA": 150, "physical_target_site": 9880},
        "relations": {"candidate_pair": 11446},
    },
    GRAPH_B: {
        "nodes": {"sgRNA": 150, "physical_target_site": 9880},
        "relations": {"candidate_pair": 11446, "sequence_similar_to": 1208},
    },
    GRAPH_C: {
        "nodes": {"sgRNA": 150, "target_observation": 11446},
        "relations": {"candidate_pair": 11446, "context_similar_to": 91754},
    },
}


@dataclass(frozen=True)
class MaterializedGraph:
    """Manifest and strict-inductive PyG views for one Sprint 3 schema."""

    graph_name: str
    manifest: dict[str, Any]
    views: dict[str, HeteroData]

    def view(self, split: str) -> HeteroData:
        if split not in self.views:
            raise ValueError(f"Unknown materialized graph view: {split}")
        return self.views[split]


class Sprint3HeteroDataLoader:
    """Load serialized Sprint 3 graph artifacts without rebuilding topology."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        expected_counts: Mapping[str, Mapping[str, Mapping[str, int]]] | None = None,
        expected_candidate_split_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.expected_counts = dict(expected_counts or CANONICAL_COUNTS)
        self.expected_candidate_split_counts = dict(
            expected_candidate_split_counts or CANDIDATE_SPLIT_COUNTS
        )

    def load_all(self) -> dict[str, MaterializedGraph]:
        return {graph_name: self.load(graph_name) for graph_name in (GRAPH_A, GRAPH_B, GRAPH_C)}

    def load(self, graph_name: str) -> MaterializedGraph:
        if graph_name not in self.expected_counts:
            raise ValueError(f"Unsupported or unexpected graph schema: {graph_name}")
        graph_dir = self.artifact_dir / graph_name
        manifest = self._read_manifest(graph_dir)
        self._validate_manifest(graph_name, manifest)
        nodes, relations, features = self._read_tables(graph_name, graph_dir, manifest)
        self._validate_tables(graph_name, nodes, relations, features, manifest)
        views = {
            view: self._materialize_view(graph_name, nodes, relations, features, manifest, view)
            for view in VIEWS
        }
        return MaterializedGraph(graph_name=graph_name, manifest=manifest, views=views)

    @staticmethod
    def _read_manifest(graph_dir: Path) -> dict[str, Any]:
        manifest_path = graph_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Graph manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _validate_manifest(self, graph_name: str, manifest: dict[str, Any]) -> None:
        if manifest.get("graph_name") != graph_name:
            raise ValueError(f"Graph manifest identity drift for {graph_name}")
        if manifest.get("label_scheme") != LABEL_SCHEME or manifest.get("label_definition") != LABEL_DEFINITION:
            raise ValueError(f"Graph manifest label contract drift for {graph_name}")
        if manifest.get("split_id") != SPLIT_ID:
            raise ValueError(f"Graph manifest split contract drift for {graph_name}")
        metadata = manifest.get("metadata", {})
        if metadata.get("visibility_policy") != VISIBILITY_POLICY:
            raise ValueError(f"Graph manifest visibility policy drift for {graph_name}")
        expected = self.expected_counts[graph_name]
        if manifest.get("nodes") != expected["nodes"] or manifest.get("relations") != expected["relations"]:
            raise ValueError(f"Graph manifest count drift for {graph_name}")
        if graph_name == GRAPH_A:
            self._validate_graph_a_manifest(metadata)
        elif graph_name == GRAPH_B:
            self._validate_graph_b_manifest(metadata)
        elif graph_name == GRAPH_C:
            self._validate_graph_c_manifest(metadata, manifest.get("preprocessing", {}))

    @staticmethod
    def _validate_graph_a_manifest(metadata: dict[str, Any]) -> None:
        if metadata.get("context_placement") != "candidate_pair_edge":
            raise ValueError("Graph A context placement drift")
        if metadata.get("target_key_fields") != list(PHYSICAL_TARGET_KEY_FIELDS):
            raise ValueError("Graph A physical target identity drift")
        if metadata.get("candidate_relation_visibility") != "filter_by_split_for_model_views":
            raise ValueError("Graph A candidate visibility drift")

    @staticmethod
    def _validate_graph_b_manifest(metadata: dict[str, Any]) -> None:
        if metadata.get("base_graph") != GRAPH_A or metadata.get("inherits_feature_tables_from") != GRAPH_A:
            raise ValueError("Graph B base-schema contract drift")
        if metadata.get("role") != "bounded_secondary_control":
            raise ValueError("Graph B bounded-control role drift")
        _assert_auxiliary_composition(metadata, graph_name="Graph B")

    @staticmethod
    def _validate_graph_c_manifest(metadata: dict[str, Any], preprocessing: dict[str, Any]) -> None:
        if metadata.get("context_placement") != "target_observation_node":
            raise ValueError("Graph C context placement drift")
        if metadata.get("target_observation_key") != "source row id":
            raise ValueError("Graph C observation identity drift")
        context_preprocessing = preprocessing.get("target_observation_context", {})
        if context_preprocessing.get("fit_scope") != "train_only":
            raise ValueError("Graph C context preprocessing scope drift")
        _assert_auxiliary_composition(metadata, graph_name="Graph C")

    def _read_tables(
        self,
        graph_name: str,
        graph_dir: Path,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        nodes = {
            name: _read_parquet(graph_dir / f"nodes_{name}.parquet")
            for name in manifest["nodes"]
        }
        relations = {
            name: _read_parquet(graph_dir / f"relation_{name}.parquet")
            for name in manifest["relations"]
        }
        feature_graph_dir = self.artifact_dir / GRAPH_A if graph_name == GRAPH_B else graph_dir
        feature_manifest = self._read_manifest(feature_graph_dir) if graph_name == GRAPH_B else manifest
        if graph_name == GRAPH_B:
            self._validate_manifest(GRAPH_A, feature_manifest)
        features = {
            name: _read_parquet(feature_graph_dir / f"features_{name}.parquet")
            for name in feature_manifest.get("feature_tables", {})
        }
        return nodes, relations, features

    def _validate_tables(
        self,
        graph_name: str,
        nodes: dict[str, pd.DataFrame],
        relations: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        manifest: dict[str, Any],
    ) -> None:
        for name, expected_count in manifest["nodes"].items():
            if len(nodes[name]) != expected_count:
                raise ValueError(f"Serialized node table count drift for {graph_name}:{name}")
        for name, expected_count in manifest["relations"].items():
            if len(relations[name]) != expected_count:
                raise ValueError(f"Serialized relation table count drift for {graph_name}:{name}")
        candidate = relations["candidate_pair"]
        _require_columns(
            candidate,
            {"edge_id", "source_sgrna_id", "label", "split", "measured", "experiment_id"},
            f"{graph_name}:candidate_pair",
        )
        counts = candidate["split"].value_counts().to_dict()
        if counts != self.expected_candidate_split_counts:
            raise ValueError(f"Candidate split counts drift for {graph_name}: {counts}")
        if (candidate["measured"] != 1).any() or (candidate["experiment_id"] == 18).any():
            raise ValueError(f"Candidate supervision universe drift for {graph_name}")
        if candidate["edge_id"].astype(str).duplicated().any():
            raise ValueError(f"Candidate source-row identity is not unique for {graph_name}")
        if graph_name in {GRAPH_B, GRAPH_C}:
            self._validate_candidate_contract_matches_graph_a(graph_name, candidate)
        if graph_name == GRAPH_C and not candidate["target_observation_id"].astype(str).equals(
            candidate["edge_id"].astype(str)
        ):
            raise ValueError("Graph C candidate destination no longer preserves source-row observation identity")
        feature_contract = (
            self._read_manifest(self.artifact_dir / GRAPH_A)
            if graph_name == GRAPH_B
            else manifest
        )
        expected_features = feature_contract.get("feature_tables", {})
        if set(features) != set(expected_features):
            raise ValueError(f"Serialized feature-table identity drift for {graph_name}")
        for name, table in features.items():
            _validate_feature_table(name, table)
            if len(table) != len(candidate) or len(_feature_columns(table)) != expected_features[name]:
                raise ValueError(f"Serialized feature-table shape drift for {graph_name}:{name}")
        if graph_name == GRAPH_A:
            allowed_target_columns = {"node_id", *PHYSICAL_TARGET_KEY_FIELDS}
            if set(nodes["physical_target_site"].columns) != allowed_target_columns:
                raise ValueError("Graph A physical targets contain unexpected feature or identity columns")
        if graph_name == GRAPH_B:
            self._validate_auxiliary_relation(
                relations["sequence_similar_to"],
                source_column="source_sgrna_id",
                target_column="target_sgrna_id",
                graph_name=graph_name,
            )
        if graph_name == GRAPH_C:
            self._validate_auxiliary_relation(
                relations["context_similar_to"],
                source_column="source_observation_id",
                target_column="target_observation_id",
                graph_name=graph_name,
            )

    def _validate_candidate_contract_matches_graph_a(
        self,
        graph_name: str,
        candidate: pd.DataFrame,
    ) -> None:
        graph_a_dir = self.artifact_dir / GRAPH_A
        graph_a_manifest = self._read_manifest(graph_a_dir)
        self._validate_manifest(GRAPH_A, graph_a_manifest)
        reference = _read_parquet(graph_a_dir / "relation_candidate_pair.parquet")
        contract_columns = [
            "edge_id",
            "source_sgrna_id",
            "label",
            "split",
            "measured",
            "experiment_id",
        ]
        if graph_name == GRAPH_B:
            contract_columns.append("target_node_id")
        _require_columns(reference, set(contract_columns), f"{GRAPH_A}:candidate_pair")
        expected = reference[contract_columns].copy()
        actual = candidate[contract_columns].copy()
        expected["edge_id"] = expected["edge_id"].astype(str)
        actual["edge_id"] = actual["edge_id"].astype(str)
        expected = expected.sort_values("edge_id").reset_index(drop=True)
        actual = actual.sort_values("edge_id").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"Candidate supervised contract drift relative to Graph A for {graph_name}")

    @staticmethod
    def _validate_auxiliary_relation(
        relation: pd.DataFrame,
        *,
        source_column: str,
        target_column: str,
        graph_name: str,
    ) -> None:
        _require_columns(
            relation,
            {"view", source_column, target_column, "distance", "source_split", "target_split"},
            graph_name,
        )
        for view, composition in VIEW_COMPOSITION.items():
            part = relation.loc[relation["view"] == view]
            visible_splits = set(part["source_split"]).union(part["target_split"])
            if not visible_splits.issubset(set(composition)):
                raise ValueError(f"Auxiliary relation view leakage for {graph_name}:{view}")
            if view != "train":
                held_out_to_held_out = (part["source_split"] == view) & (part["target_split"] == view)
                if held_out_to_held_out.any():
                    raise ValueError(f"Held-out auxiliary neighbors exposed for {graph_name}:{view}")

    def _materialize_view(
        self,
        graph_name: str,
        nodes: dict[str, pd.DataFrame],
        relations: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        manifest: dict[str, Any],
        view: str,
    ) -> HeteroData:
        composition = VIEW_COMPOSITION[view]
        candidate = relations["candidate_pair"].loc[
            relations["candidate_pair"]["split"].isin(composition)
        ].reset_index(drop=True)
        auxiliary_name = _auxiliary_relation_name(graph_name)
        auxiliary = None
        if auxiliary_name is not None:
            auxiliary = relations[auxiliary_name].loc[
                relations[auxiliary_name]["view"].isin(composition)
            ].reset_index(drop=True)
        visible_nodes = _visible_node_ids(graph_name, candidate, auxiliary)
        data = HeteroData()
        data.graph_name = graph_name
        data.view_name = view
        data.split_id = SPLIT_ID
        data.label_scheme = LABEL_SCHEME
        data.visibility_policy = VISIBILITY_POLICY
        data.manifest_counts = {
            "nodes": manifest["nodes"],
            "relations": manifest["relations"],
        }
        node_maps = self._materialize_nodes(data, graph_name, nodes, features, visible_nodes)
        self._materialize_candidate_edges(data, graph_name, candidate, features, node_maps, view)
        if auxiliary_name is not None and auxiliary is not None:
            self._materialize_auxiliary_edges(data, graph_name, auxiliary_name, auxiliary, node_maps)
        return data

    @staticmethod
    def _materialize_nodes(
        data: HeteroData,
        graph_name: str,
        nodes: dict[str, pd.DataFrame],
        features: dict[str, pd.DataFrame],
        visible_nodes: dict[str, set[str]],
    ) -> dict[str, dict[str, int]]:
        maps: dict[str, dict[str, int]] = {}
        for node_type, table in nodes.items():
            selected = table.loc[table["node_id"].astype(str).isin(visible_nodes[node_type])].reset_index(drop=True)
            node_ids = selected["node_id"].astype(str).tolist()
            maps[node_type] = {node_id: index for index, node_id in enumerate(node_ids)}
            storage = data[node_type]
            storage.num_nodes = len(node_ids)
            storage.audit_node_ids = node_ids
            if "split" in selected.columns:
                storage.audit_splits = selected["split"].astype(str).tolist()
            feature_table = selected
            if node_type == "target_observation":
                feature_table = _left_join_features(selected, features["target_observation_features"])
            feature_columns = _feature_columns(feature_table)
            if feature_columns:
                storage.x = _float_tensor(feature_table[feature_columns])
                storage.feature_names = feature_columns
        return maps

    @staticmethod
    def _materialize_candidate_edges(
        data: HeteroData,
        graph_name: str,
        candidate: pd.DataFrame,
        features: dict[str, pd.DataFrame],
        node_maps: dict[str, dict[str, int]],
        view: str,
    ) -> None:
        destination_type, destination_column = _candidate_destination(graph_name)
        edge_type = ("sgRNA", "candidate_pair", destination_type)
        storage = data[edge_type]
        storage.edge_index = _edge_index(
            candidate,
            source_column="source_sgrna_id",
            target_column=destination_column,
            source_map=node_maps["sgRNA"],
            target_map=node_maps[destination_type],
        )
        storage.edge_label = torch.tensor(candidate["label"].to_numpy(dtype=int), dtype=torch.float32)
        storage.supervision_mask = torch.tensor(candidate["split"].eq(view).to_numpy(), dtype=torch.bool)
        storage.audit_edge_ids = candidate["edge_id"].astype(str).tolist()
        storage.audit_splits = candidate["split"].astype(str).tolist()
        storage.audit_sgrna_ids = candidate["source_sgrna_id"].astype(str).tolist()
        if "target_node_id" in candidate.columns:
            storage.audit_genome = [str(nid).split("|")[0] for nid in candidate["target_node_id"].astype(str)]
        for feature_name, feature_table in features.items():
            if graph_name == GRAPH_C and feature_name == "target_observation_features":
                continue
            selected = _left_join_features(candidate.rename(columns={"edge_id": "node_id"}), feature_table)
            columns = _feature_columns(selected)
            storage[f"edge_attr_{feature_name.lower()}"] = _float_tensor(selected[columns])
            storage[f"feature_names_{feature_name.lower()}"] = columns

    @staticmethod
    def _materialize_auxiliary_edges(
        data: HeteroData,
        graph_name: str,
        auxiliary_name: str,
        auxiliary: pd.DataFrame,
        node_maps: dict[str, dict[str, int]],
    ) -> None:
        if graph_name == GRAPH_B:
            node_type = "sgRNA"
            source_column = "source_sgrna_id"
            target_column = "target_sgrna_id"
        else:
            node_type = "target_observation"
            source_column = "source_observation_id"
            target_column = "target_observation_id"
        storage = data[(node_type, auxiliary_name, node_type)]
        storage.edge_index = _edge_index(
            auxiliary,
            source_column=source_column,
            target_column=target_column,
            source_map=node_maps[node_type],
            target_map=node_maps[node_type],
        )
        storage.edge_attr = torch.tensor(auxiliary[["distance"]].to_numpy(dtype=float), dtype=torch.float32)
        storage.feature_names = ["distance"]
        storage.audit_fragments = auxiliary["view"].astype(str).tolist()


def validate_gcn_headline_config(config: Mapping[str, Any]) -> None:
    """Reject placeholder or non-contract headline GCN evaluation settings."""
    data = config.get("data", {})
    graph = config.get("graph", {})
    evaluation = config.get("evaluation", {})
    if not isinstance(data, Mapping) or not isinstance(graph, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("GCN config data, graph, and evaluation sections must be mappings")
    if data.get("label_scheme") != LABEL_SCHEME:
        raise ValueError("GCN headline config must use Scheme A")
    if data.get("split_id") != SPLIT_ID:
        raise ValueError("GCN headline config must use the locked guide-level split")
    if graph.get("visibility_policy") != VISIBILITY_POLICY:
        raise ValueError("GCN headline config must use strict-inductive visibility")
    if evaluation.get("protocol") != "headline_guide_level":
        raise ValueError("GCN headline config must declare headline guide-level evaluation")
    values = [value.lower() for value in _config_string_values(config)]
    if any("debug" in value or "random_edge" in value or "random-edge" in value for value in values):
        raise ValueError("Debug/random-edge configuration cannot be used as a GCN headline run")


def _assert_auxiliary_composition(metadata: dict[str, Any], *, graph_name: str) -> None:
    stored = metadata.get("auxiliary_relation_view_composition")
    expected = {key: list(value) for key, value in VIEW_COMPOSITION.items()}
    if stored != expected:
        raise ValueError(f"{graph_name} auxiliary visibility composition drift")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Graph table not found: {path}")
    return pd.read_parquet(path)


def _config_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [
            item
            for nested in value.values()
            for item in _config_string_values(nested)
        ]
    if isinstance(value, list):
        return [
            item
            for nested in value
            for item in _config_string_values(nested)
        ]
    return []


def _require_columns(table: pd.DataFrame, columns: set[str], table_name: str) -> None:
    missing = sorted(columns.difference(table.columns))
    if missing:
        raise ValueError(f"Serialized table columns drift for {table_name}: missing={missing}")


def _validate_feature_table(name: str, table: pd.DataFrame) -> None:
    _require_columns(table, {"record_id"}, f"feature table {name}")
    if table["record_id"].astype(str).duplicated().any():
        raise ValueError(f"Feature table identity is not unique: {name}")
    feature_columns = _feature_columns(table)
    if not feature_columns:
        raise ValueError(f"Feature table contains no predictive columns: {name}")
    sources = {column.removeprefix("feature__") for column in feature_columns}
    forbidden = sorted(sources.intersection(FORBIDDEN_PREDICTIVE_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden predictive columns in serialized feature table {name}: {forbidden}")


def _feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column.startswith("feature__")]


def _left_join_features(records: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["_record_id"] = records["node_id"].astype(str)
    keyed_features = feature_table.copy()
    keyed_features["_record_id"] = keyed_features["record_id"].astype(str)
    selected = records[["_record_id"]].merge(
        keyed_features.drop(columns=["record_id"]),
        on="_record_id",
        how="left",
        validate="one_to_one",
    )
    if selected[_feature_columns(selected)].isna().any().any():
        raise ValueError("Materialized feature rows are missing or contain NaN values")
    return selected


def _float_tensor(table: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(table.to_numpy(dtype=float), dtype=torch.float32)


def _candidate_destination(graph_name: str) -> tuple[str, str]:
    if graph_name in {GRAPH_A, GRAPH_B}:
        return "physical_target_site", "target_node_id"
    return "target_observation", "target_observation_id"


def _auxiliary_relation_name(graph_name: str) -> str | None:
    if graph_name == GRAPH_B:
        return "sequence_similar_to"
    if graph_name == GRAPH_C:
        return "context_similar_to"
    return None


def _visible_node_ids(
    graph_name: str,
    candidate: pd.DataFrame,
    auxiliary: pd.DataFrame | None,
) -> dict[str, set[str]]:
    destination_type, destination_column = _candidate_destination(graph_name)
    visible = {
        "sgRNA": set(candidate["source_sgrna_id"].astype(str)),
        destination_type: set(candidate[destination_column].astype(str)),
    }
    if graph_name == GRAPH_B and auxiliary is not None:
        visible["sgRNA"].update(auxiliary["source_sgrna_id"].astype(str))
        visible["sgRNA"].update(auxiliary["target_sgrna_id"].astype(str))
    if graph_name == GRAPH_C and auxiliary is not None:
        visible["target_observation"].update(auxiliary["source_observation_id"].astype(str))
        visible["target_observation"].update(auxiliary["target_observation_id"].astype(str))
    return visible


def _edge_index(
    relation: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    source_map: dict[str, int],
    target_map: dict[str, int],
) -> torch.Tensor:
    source = [source_map[str(value)] for value in relation[source_column]]
    target = [target_map[str(value)] for value in relation[target_column]]
    return torch.tensor([source, target], dtype=torch.long)
