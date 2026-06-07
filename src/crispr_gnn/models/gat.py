"""Sprint 7 Graph A attention-based edge classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GATv2Conv

from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    TARGET_REPRESENTATION_POLICY,
    _candidate_edge_features,
    _validate_graph_a_data,
)


AttentionConvName = Literal["gat", "gatv2"]


class GraphAEdgeAttentionGNN(nn.Module):
    """Graph A link predictor with GAT/GATv2 message passing.

    Candidate edge features are still provided to the final edge classifier, as
    in ``GraphAEdgeGCN``. When ``edge_aware_attention`` is true, the same
    candidate edge feature vector is also passed to attention/message passing
    through PyG's ``edge_dim``/``edge_attr`` path. Reverse candidate edges reuse
    the forward edge feature vector; PyG-added self-loops receive a zero edge
    feature vector via ``fill_value=0.0``.
    """

    def __init__(
        self,
        *,
        conv_name: AttentionConvName,
        sgrna_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        attention_dropout: float | None = None,
        edge_aware_attention: bool = True,
        self_loop_edge_fill: float = 0.0,
        gatv2_share_weights: bool = False,
    ) -> None:
        super().__init__()
        if conv_name not in {"gat", "gatv2"}:
            raise ValueError("conv_name must be 'gat' or 'gatv2'")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if sgrna_input_dim <= 0:
            raise ValueError("sgrna_input_dim must be positive")
        if edge_input_dim <= 0 and edge_aware_attention:
            raise ValueError("edge-aware attention requires positive edge_input_dim")
        if heads < 1:
            raise ValueError("heads must be at least 1")
        if concat and hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads when concat=True")
        if self_loop_edge_fill != 0.0:
            raise ValueError("Sprint 7 edge-aware self-loop fill is frozen to 0.0")

        self.conv_name = conv_name
        self.target_representation_policy = TARGET_REPRESENTATION_POLICY
        self.edge_aware_attention = bool(edge_aware_attention)
        self.self_loop_edge_fill = float(self_loop_edge_fill)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.physical_target_type = nn.Parameter(torch.zeros(1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        conv_cls: type[GATConv] | type[GATv2Conv] = GATConv if conv_name == "gat" else GATv2Conv
        edge_dim = edge_input_dim if self.edge_aware_attention else None
        attention_p = dropout if attention_dropout is None else attention_dropout
        out_channels = hidden_dim // heads if concat else hidden_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            kwargs = {}
            if conv_name == "gatv2":
                kwargs["share_weights"] = gatv2_share_weights
            self.convs.append(
                conv_cls(
                    hidden_dim,
                    out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=attention_p,
                    add_self_loops=True,
                    edge_dim=edge_dim,
                    fill_value=self.self_loop_edge_fill,
                    **kwargs,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        classifier_input_dim = hidden_dim * 4 + edge_input_dim
        self.edge_classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        data: HeteroData,
        *,
        edge_feature_attrs: Sequence[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        _validate_graph_a_data(data)
        edge_store = data[GRAPH_A_EDGE_TYPE]
        candidate_edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph A")
        x = self._initial_node_features(data)
        attention_edge_index, attention_edge_attr = graph_a_attention_edge_tensors(
            data,
            edge_feature_attrs=edge_feature_attrs,
        )
        conv_edge_attr = attention_edge_attr if self.edge_aware_attention else None
        attention_records: list[dict[str, torch.Tensor]] = []
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            if return_attention:
                x, weights = conv(
                    x,
                    attention_edge_index,
                    edge_attr=conv_edge_attr,
                    return_attention_weights=True,
                )
                returned_edge_index, alpha = weights
                attention_records.append(
                    {
                        "layer": torch.tensor(layer, dtype=torch.long, device=alpha.device),
                        "edge_index": returned_edge_index.detach(),
                        "alpha": alpha.detach(),
                    }
                )
            else:
                x = conv(x, attention_edge_index, edge_attr=conv_edge_attr)
            x = norm(x.relu())
            x = self.dropout(x)

        candidate_edge_index = edge_store.edge_index
        source_index = candidate_edge_index[0]
        target_index = candidate_edge_index[1] + data["sgRNA"].num_nodes
        source = x[source_index]
        target = x[target_index]
        pair = torch.cat(
            [source, target, source * target, torch.abs(source - target), candidate_edge_attr],
            dim=1,
        )
        logits = self.edge_classifier(pair).squeeze(-1)
        if return_attention:
            return logits, attention_records
        return logits

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        guide_features = self.sgrna_encoder(data["sgRNA"].x)
        target_count = int(data["physical_target_site"].num_nodes)
        target_features = self.physical_target_type.expand(target_count, -1)
        return torch.cat([guide_features, target_features], dim=0)


class GraphAEdgeGAT(GraphAEdgeAttentionGNN):
    """Graph A GATConv edge classifier."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(conv_name="gat", **kwargs)


class GraphAEdgeGATv2(GraphAEdgeAttentionGNN):
    """Graph A GATv2Conv edge classifier."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(conv_name="gatv2", **kwargs)


def graph_a_attention_edge_tensors(
    data: HeteroData,
    *,
    edge_feature_attrs: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bidirectional candidate edges and duplicated candidate edge attrs."""
    _validate_graph_a_data(data)
    edge_store = data[GRAPH_A_EDGE_TYPE]
    edge_index = edge_store.edge_index
    edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph A")
    source = edge_index[0]
    target = edge_index[1] + int(data["sgRNA"].num_nodes)
    forward = torch.stack([source, target], dim=0)
    reverse = torch.stack([target, source], dim=0)
    attention_edge_index = torch.cat([forward, reverse], dim=1)
    attention_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return attention_edge_index, attention_edge_attr
