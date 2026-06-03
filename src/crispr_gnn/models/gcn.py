"""Minimal Sprint 4 GCN edge classifiers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from crispr_gnn.graph.graph_schemas import GRAPH_A, GRAPH_B, GRAPH_C


GRAPH_A_EDGE_TYPE = ("sgRNA", "candidate_pair", "physical_target_site")
GRAPH_B_EDGE_TYPE = ("sgRNA", "candidate_pair", "physical_target_site")
GRAPH_B_SIMILARITY_EDGE_TYPE = ("sgRNA", "sequence_similar_to", "sgRNA")
GRAPH_C_EDGE_TYPE = ("sgRNA", "candidate_pair", "target_observation")
GRAPH_C_CONTEXT_EDGE_TYPE = ("target_observation", "context_similar_to", "target_observation")
TARGET_REPRESENTATION_POLICY = "zero_type_feature"
GRAPH_C_TARGET_REPRESENTATION_POLICY = "target_observation_context_encoder"


class GraphAEdgeGCN(nn.Module):
    """GCN link predictor over Graph A candidate guide-target relations.

    Graph A physical target nodes are intentionally featureless. This model
    gives them one shared learned type vector, not per-target ID embeddings, so
    target identity, coordinates, and row-varying context do not become hidden
    predictive tensors.
    """

    def __init__(
        self,
        *,
        sgrna_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if sgrna_input_dim <= 0:
            raise ValueError("sgrna_input_dim must be positive")
        if edge_input_dim < 0:
            raise ValueError("edge_input_dim cannot be negative")
        self.target_representation_policy = TARGET_REPRESENTATION_POLICY
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.physical_target_type = nn.Parameter(torch.zeros(1, hidden_dim))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        classifier_input_dim = hidden_dim * 4 + edge_input_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData, *, edge_feature_attrs: Sequence[str]) -> torch.Tensor:
        _validate_graph_a_data(data)
        edge_store = data[GRAPH_A_EDGE_TYPE]
        edge_index = edge_store.edge_index
        edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph A")
        x = self._initial_node_features(data)
        homogeneous_edge_index = _homogeneous_candidate_edge_index(
            edge_index=edge_index,
            sgrna_nodes=data["sgRNA"].num_nodes,
        )
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = norm(conv(x, homogeneous_edge_index).relu())
            x = self.dropout(x)
        source_index = edge_index[0]
        target_index = edge_index[1] + data["sgRNA"].num_nodes
        source = x[source_index]
        target = x[target_index]
        pair = torch.cat([source, target, source * target, torch.abs(source - target), edge_attr], dim=1)
        return self.edge_classifier(pair).squeeze(-1)

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        sgrna_x = data["sgRNA"].x
        guide_features = self.sgrna_encoder(sgrna_x)
        target_count = int(data["physical_target_site"].num_nodes)
        target_features = self.physical_target_type.expand(target_count, -1)
        return torch.cat([guide_features, target_features], dim=0)


class GraphBEdgeGCN(nn.Module):
    """GCN link predictor over Graph B candidate relations with guide-similarity topology.

    Graph B inherits Graph A's featureless physical targets and candidate edge features.
    The only structural addition is label-free guide-sequence similarity edges
    (sgRNA↔sgRNA) which participate in message passing. This isolates the topology
    contribution of guide similarity from Graph C's simultaneous topology and target
    semantics change.

    Risk note: without including sequence_similar_to edges in the homogeneous adjacency
    matrix passed to GCNConv, Graph B would be numerically identical to Graph A. The
    _homogeneous_graph_b_edge_index helper combines candidate and similarity edges,
    and an assertion confirms that similarity indices stay within the sgRNA range (no
    offset must be applied to sgRNA↔sgRNA edges).
    """

    def __init__(
        self,
        *,
        sgrna_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if sgrna_input_dim <= 0:
            raise ValueError("sgrna_input_dim must be positive")
        if edge_input_dim < 0:
            raise ValueError("edge_input_dim cannot be negative")
        self.target_representation_policy = TARGET_REPRESENTATION_POLICY
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.physical_target_type = nn.Parameter(torch.zeros(1, hidden_dim))
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        classifier_input_dim = hidden_dim * 4 + edge_input_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData, *, edge_feature_attrs: Sequence[str]) -> torch.Tensor:
        _validate_graph_b_data(data)
        edge_store = data[GRAPH_B_EDGE_TYPE]
        edge_index = edge_store.edge_index
        edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph B")
        x = self._initial_node_features(data)
        homogeneous_edge_index = _homogeneous_graph_b_edge_index(data)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = norm(conv(x, homogeneous_edge_index).relu())
            x = self.dropout(x)
        source_index = edge_index[0]
        target_index = edge_index[1] + data["sgRNA"].num_nodes
        source = x[source_index]
        target = x[target_index]
        pair = torch.cat([source, target, source * target, torch.abs(source - target), edge_attr], dim=1)
        return self.edge_classifier(pair).squeeze(-1)

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        sgrna_x = data["sgRNA"].x
        guide_features = self.sgrna_encoder(sgrna_x)
        target_count = int(data["physical_target_site"].num_nodes)
        target_features = self.physical_target_type.expand(target_count, -1)
        return torch.cat([guide_features, target_features], dim=0)


class GraphCEdgeGCN(nn.Module):
    """GCN link predictor over Graph C observation-level context targets.

    Graph C target observations carry the Sprint 3 train-preprocessed context
    tensor as node features. Context enters through this dedicated node encoder,
    not by copying row-varying context onto candidate edge tensors.
    """

    def __init__(
        self,
        *,
        sgrna_input_dim: int,
        target_observation_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if sgrna_input_dim <= 0:
            raise ValueError("sgrna_input_dim must be positive")
        if target_observation_input_dim <= 0:
            raise ValueError("target_observation_input_dim must be positive")
        if edge_input_dim < 0:
            raise ValueError("edge_input_dim cannot be negative")
        self.target_representation_policy = GRAPH_C_TARGET_REPRESENTATION_POLICY
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.target_observation_encoder = nn.Sequential(
            nn.Linear(target_observation_input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        classifier_input_dim = hidden_dim * 4 + edge_input_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData, *, edge_feature_attrs: Sequence[str]) -> torch.Tensor:
        _validate_graph_c_data(data)
        edge_store = data[GRAPH_C_EDGE_TYPE]
        edge_index = edge_store.edge_index
        edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph C")
        x = self._initial_node_features(data)
        homogeneous_edge_index = _homogeneous_graph_c_edge_index(data)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = norm(conv(x, homogeneous_edge_index).relu())
            x = self.dropout(x)
        source_index = edge_index[0]
        target_index = edge_index[1] + data["sgRNA"].num_nodes
        source = x[source_index]
        target = x[target_index]
        pair = torch.cat([source, target, source * target, torch.abs(source - target), edge_attr], dim=1)
        return self.edge_classifier(pair).squeeze(-1)

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        guide_features = self.sgrna_encoder(data["sgRNA"].x)
        target_features = self.target_observation_encoder(data["target_observation"].x)
        return torch.cat([guide_features, target_features], dim=0)


def graph_a_edge_feature_attrs(feature_sets: Sequence[str]) -> list[str]:
    attrs = []
    for feature_set in feature_sets:
        normalized = feature_set.lower()
        if normalized == "s1_pair":
            attrs.append("edge_attr_s1_pair")
        elif normalized in {"f1", "f2", "f3", "f4"}:
            attrs.append(f"edge_attr_{normalized}")
        elif normalized in {
            "s5f0_seq",
            "s5f1_mismatch",
            "s5f2_energy",
            "s5f3_experimental_epi",
            "s5f4_computed_agg",
            "s5f5_computed_pos",
        }:
            attrs.append(f"edge_attr_{normalized}")
        else:
            raise ValueError(f"Unsupported Graph A edge feature set: {feature_set}")
    return attrs


def graph_b_edge_feature_attrs(feature_sets: Sequence[str]) -> list[str]:
    attrs = []
    for feature_set in feature_sets:
        normalized = feature_set.lower()
        if normalized == "s1_pair":
            attrs.append("edge_attr_s1_pair")
        elif normalized in {"f1", "f2", "f3", "f4"}:
            attrs.append(f"edge_attr_{normalized}")
        else:
            raise ValueError(f"Unsupported Graph B edge feature set: {feature_set}")
    return attrs


def graph_b_feature_dimensions(data: HeteroData, edge_feature_attrs: Sequence[str]) -> tuple[int, int]:
    _validate_graph_b_data(data)
    sgrna_dim = int(data["sgRNA"].x.shape[1])
    edge_dim = int(_candidate_edge_features(data[GRAPH_B_EDGE_TYPE], edge_feature_attrs, graph_label="Graph B").shape[1])
    return sgrna_dim, edge_dim


def graph_c_edge_feature_attrs(feature_sets: Sequence[str]) -> list[str]:
    attrs = []
    for feature_set in feature_sets:
        normalized = feature_set.lower()
        if normalized in {"candidate_pair_features", "candidate_pair"}:
            attrs.append("edge_attr_candidate_pair_features")
        elif normalized == "s5f2_energy":
            attrs.append(f"edge_attr_{normalized}")
        else:
            raise ValueError(f"Unsupported Graph C edge feature set: {feature_set}")
    return attrs


def graph_a_feature_dimensions(data: HeteroData, edge_feature_attrs: Sequence[str]) -> tuple[int, int]:
    _validate_graph_a_data(data)
    sgrna_dim = int(data["sgRNA"].x.shape[1])
    edge_dim = int(_candidate_edge_features(data[GRAPH_A_EDGE_TYPE], edge_feature_attrs, graph_label="Graph A").shape[1])
    return sgrna_dim, edge_dim


def graph_c_feature_dimensions(data: HeteroData, edge_feature_attrs: Sequence[str]) -> tuple[int, int, int]:
    _validate_graph_c_data(data)
    sgrna_dim = int(data["sgRNA"].x.shape[1])
    target_dim = int(data["target_observation"].x.shape[1])
    edge_dim = int(_candidate_edge_features(data[GRAPH_C_EDGE_TYPE], edge_feature_attrs, graph_label="Graph C").shape[1])
    return sgrna_dim, target_dim, edge_dim


def _validate_graph_a_data(data: HeteroData) -> None:
    if getattr(data, "graph_name", None) != GRAPH_A:
        raise ValueError("GraphAEdgeGCN only supports Graph A materialized views")
    if "x" not in data["sgRNA"]:
        raise ValueError("Graph A sgRNA nodes must contain guide features")
    if "x" in data["physical_target_site"]:
        raise ValueError("Graph A physical target nodes must remain featureless")
    if GRAPH_A_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph A candidate edge type is missing")


def _validate_graph_b_data(data: HeteroData) -> None:
    if getattr(data, "graph_name", None) != GRAPH_B:
        raise ValueError("GraphBEdgeGCN only supports Graph B materialized views")
    if "x" not in data["sgRNA"]:
        raise ValueError("Graph B sgRNA nodes must contain guide features")
    if "x" in data["physical_target_site"]:
        raise ValueError("Graph B physical target nodes must remain featureless")
    if GRAPH_B_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph B candidate edge type is missing")
    if GRAPH_B_SIMILARITY_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph B sequence_similar_to edge type is missing")


def _validate_graph_c_data(data: HeteroData) -> None:
    if getattr(data, "graph_name", None) != GRAPH_C:
        raise ValueError("GraphCEdgeGCN only supports Graph C materialized views")
    if "x" not in data["sgRNA"]:
        raise ValueError("Graph C sgRNA nodes must contain guide features")
    if "x" not in data["target_observation"]:
        raise ValueError("Graph C target_observation nodes must contain context features")
    if "physical_target_site" in data.node_types:
        raise ValueError("Graph C must use target_observation nodes, not physical_target_site nodes")
    if GRAPH_C_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph C candidate edge type is missing")
    if GRAPH_C_CONTEXT_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph C context similarity edge type is missing")


def _candidate_edge_features(
    edge_store: object,
    edge_feature_attrs: Sequence[str],
    *,
    graph_label: str,
) -> torch.Tensor:
    tensors = []
    for attr in edge_feature_attrs:
        if attr not in edge_store:
            raise ValueError(f"{graph_label} candidate edge feature tensor is missing: {attr}")
        tensors.append(edge_store[attr])
    if not tensors:
        edge_count = int(edge_store.edge_index.shape[1])
        return edge_store.edge_index.new_zeros((edge_count, 0), dtype=torch.float32)
    return torch.cat(tensors, dim=1).float()


def _homogeneous_candidate_edge_index(*, edge_index: torch.Tensor, sgrna_nodes: int) -> torch.Tensor:
    source = edge_index[0]
    target = edge_index[1] + int(sgrna_nodes)
    forward = torch.stack([source, target], dim=0)
    reverse = torch.stack([target, source], dim=0)
    return torch.cat([forward, reverse], dim=1)


def _homogeneous_graph_b_edge_index(data: HeteroData) -> torch.Tensor:
    """Build homogeneous edge index combining candidate and guide-similarity edges.

    Node layout: sgRNA [0..sgrna_nodes-1], physical_target_site [sgrna_nodes..total-1].

    Candidate edges (sgRNA→physical_target): physical target indices are offset by sgrna_nodes.
    Similarity edges (sgRNA↔sgRNA): both indices are sgRNA nodes — NO offset applied.
    This distinction is critical; applying the candidate offset to similarity edges would
    produce out-of-range indices or corrupt the physical target node representations.
    """
    candidate = data[GRAPH_B_EDGE_TYPE].edge_index
    similarity = data[GRAPH_B_SIMILARITY_EDGE_TYPE].edge_index
    sgrna_nodes = int(data["sgRNA"].num_nodes)
    if similarity.numel() > 0:
        assert int(similarity.max()) < sgrna_nodes, (
            f"Graph B similarity edge index {int(similarity.max())} >= sgrna_nodes {sgrna_nodes}; "
            "no offset must be applied to sgRNA↔sgRNA edges"
        )
    cand_fwd = torch.stack([candidate[0], candidate[1] + sgrna_nodes], dim=0)
    cand_rev = torch.stack([candidate[1] + sgrna_nodes, candidate[0]], dim=0)
    sim_fwd = similarity
    sim_rev = torch.stack([similarity[1], similarity[0]], dim=0)
    return torch.cat([cand_fwd, cand_rev, sim_fwd, sim_rev], dim=1)


def _homogeneous_graph_c_edge_index(data: HeteroData) -> torch.Tensor:
    candidate = data[GRAPH_C_EDGE_TYPE].edge_index
    context = data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index
    sgrna_nodes = int(data["sgRNA"].num_nodes)
    candidate_forward = torch.stack([candidate[0], candidate[1] + sgrna_nodes], dim=0)
    candidate_reverse = torch.stack([candidate[1] + sgrna_nodes, candidate[0]], dim=0)
    context_forward = context + sgrna_nodes
    context_reverse = torch.stack([context[1] + sgrna_nodes, context[0] + sgrna_nodes], dim=0)
    return torch.cat([candidate_forward, candidate_reverse, context_forward, context_reverse], dim=1)
