"""Minimal Graph A GCN edge classifier for Sprint 4."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from crispr_gnn.graph.graph_schemas import GRAPH_A


GRAPH_A_EDGE_TYPE = ("sgRNA", "candidate_pair", "physical_target_site")
TARGET_REPRESENTATION_POLICY = "zero_type_feature"


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
        edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs)
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


def graph_a_edge_feature_attrs(feature_sets: Sequence[str]) -> list[str]:
    attrs = []
    for feature_set in feature_sets:
        normalized = feature_set.lower()
        if normalized == "s1_pair":
            attrs.append("edge_attr_s1_pair")
        elif normalized in {"f1", "f2", "f3", "f4"}:
            attrs.append(f"edge_attr_{normalized}")
        else:
            raise ValueError(f"Unsupported Graph A edge feature set: {feature_set}")
    return attrs


def graph_a_feature_dimensions(data: HeteroData, edge_feature_attrs: Sequence[str]) -> tuple[int, int]:
    _validate_graph_a_data(data)
    sgrna_dim = int(data["sgRNA"].x.shape[1])
    edge_dim = int(_candidate_edge_features(data[GRAPH_A_EDGE_TYPE], edge_feature_attrs).shape[1])
    return sgrna_dim, edge_dim


def _validate_graph_a_data(data: HeteroData) -> None:
    if getattr(data, "graph_name", None) != GRAPH_A:
        raise ValueError("GraphAEdgeGCN only supports Graph A materialized views")
    if "x" not in data["sgRNA"]:
        raise ValueError("Graph A sgRNA nodes must contain guide features")
    if "x" in data["physical_target_site"]:
        raise ValueError("Graph A physical target nodes must remain featureless")
    if GRAPH_A_EDGE_TYPE not in data.edge_types:
        raise ValueError("Graph A candidate edge type is missing")


def _candidate_edge_features(edge_store: object, edge_feature_attrs: Sequence[str]) -> torch.Tensor:
    tensors = []
    for attr in edge_feature_attrs:
        if attr not in edge_store:
            raise ValueError(f"Graph A candidate edge feature tensor is missing: {attr}")
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
