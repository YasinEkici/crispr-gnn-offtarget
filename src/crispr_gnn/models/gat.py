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
    GRAPH_B_EDGE_TYPE,
    GRAPH_B_SIMILARITY_EDGE_TYPE,
    GRAPH_C_CONTEXT_EDGE_TYPE,
    GRAPH_C_EDGE_TYPE,
    GRAPH_C_TARGET_REPRESENTATION_POLICY,
    TARGET_REPRESENTATION_POLICY,
    _candidate_edge_features,
    _validate_graph_a_data,
    _validate_graph_b_data,
    _validate_graph_c_data,
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


class GraphBEdgeGATv2(nn.Module):
    """Graph B GATv2 link predictor with edge-aware candidate message passing.

    Graph B keeps physical targets featureless and adds only label-free
    guide-similarity topology. Candidate edges carry S5F2-style edge features
    into both GATv2 attention and the final edge classifier. Similarity edges
    are topology-only in this Sprint 7B design, so they receive zero edge_attr
    vectors when passed to ``GATv2Conv``.
    """

    def __init__(
        self,
        *,
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
        _validate_attention_init(
            sgrna_input_dim=sgrna_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            concat=concat,
            edge_aware_attention=edge_aware_attention,
            self_loop_edge_fill=self_loop_edge_fill,
        )
        self.target_representation_policy = TARGET_REPRESENTATION_POLICY
        self.edge_aware_attention = bool(edge_aware_attention)
        self.self_loop_edge_fill = float(self_loop_edge_fill)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.physical_target_type = nn.Parameter(torch.zeros(1, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.convs, self.norms = _gatv2_layers(
            hidden_dim=hidden_dim,
            edge_dim=edge_input_dim if self.edge_aware_attention else None,
            num_layers=num_layers,
            heads=heads,
            concat=concat,
            dropout=dropout if attention_dropout is None else attention_dropout,
            self_loop_edge_fill=self_loop_edge_fill,
            gatv2_share_weights=gatv2_share_weights,
        )
        self.edge_classifier = _edge_classifier(hidden_dim=hidden_dim, edge_input_dim=edge_input_dim, dropout=dropout)

    def forward(
        self,
        data: HeteroData,
        *,
        edge_feature_attrs: Sequence[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        _validate_graph_b_data(data)
        edge_store = data[GRAPH_B_EDGE_TYPE]
        candidate_edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph B")
        x = self._initial_node_features(data)
        attention_edge_index, attention_edge_attr = graph_b_attention_edge_tensors(
            data,
            edge_feature_attrs=edge_feature_attrs,
        )
        x, attention_records = _apply_attention_layers(
            x,
            self.convs,
            self.norms,
            self.dropout,
            attention_edge_index=attention_edge_index,
            attention_edge_attr=attention_edge_attr if self.edge_aware_attention else None,
            return_attention=return_attention,
        )
        edge_index = edge_store.edge_index
        source_index = edge_index[0]
        target_index = edge_index[1] + data["sgRNA"].num_nodes
        logits = _classify_candidate_edges(
            x,
            source_index=source_index,
            target_index=target_index,
            candidate_edge_attr=candidate_edge_attr,
            edge_classifier=self.edge_classifier,
        )
        if return_attention:
            return logits, attention_records
        return logits

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        guide_features = self.sgrna_encoder(data["sgRNA"].x)
        target_count = int(data["physical_target_site"].num_nodes)
        target_features = self.physical_target_type.expand(target_count, -1)
        return torch.cat([guide_features, target_features], dim=0)


class GraphCEdgeGATv2(nn.Module):
    """Graph C GATv2 link predictor over target-observation context nodes.

    Candidate edges carry S5F2-style edge features into GATv2 attention and the
    final classifier. Context-similarity edges remain topology/context-neighbor
    links and receive zero edge_attr vectors in the collapsed homogeneous GATv2
    view.
    """

    def __init__(
        self,
        *,
        sgrna_input_dim: int,
        target_observation_input_dim: int,
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
        _validate_attention_init(
            sgrna_input_dim=sgrna_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            concat=concat,
            edge_aware_attention=edge_aware_attention,
            self_loop_edge_fill=self_loop_edge_fill,
        )
        if target_observation_input_dim <= 0:
            raise ValueError("target_observation_input_dim must be positive")
        self.target_representation_policy = GRAPH_C_TARGET_REPRESENTATION_POLICY
        self.edge_aware_attention = bool(edge_aware_attention)
        self.self_loop_edge_fill = float(self_loop_edge_fill)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.target_observation_encoder = nn.Sequential(
            nn.Linear(target_observation_input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.convs, self.norms = _gatv2_layers(
            hidden_dim=hidden_dim,
            edge_dim=edge_input_dim if self.edge_aware_attention else None,
            num_layers=num_layers,
            heads=heads,
            concat=concat,
            dropout=dropout if attention_dropout is None else attention_dropout,
            self_loop_edge_fill=self_loop_edge_fill,
            gatv2_share_weights=gatv2_share_weights,
        )
        self.edge_classifier = _edge_classifier(hidden_dim=hidden_dim, edge_input_dim=edge_input_dim, dropout=dropout)

    def forward(
        self,
        data: HeteroData,
        *,
        edge_feature_attrs: Sequence[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        _validate_graph_c_data(data)
        edge_store = data[GRAPH_C_EDGE_TYPE]
        candidate_edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph C")
        x = self._initial_node_features(data)
        attention_edge_index, attention_edge_attr = graph_c_attention_edge_tensors(
            data,
            edge_feature_attrs=edge_feature_attrs,
        )
        x, attention_records = _apply_attention_layers(
            x,
            self.convs,
            self.norms,
            self.dropout,
            attention_edge_index=attention_edge_index,
            attention_edge_attr=attention_edge_attr if self.edge_aware_attention else None,
            return_attention=return_attention,
        )
        edge_index = edge_store.edge_index
        source_index = edge_index[0]
        target_index = edge_index[1] + data["sgRNA"].num_nodes
        logits = _classify_candidate_edges(
            x,
            source_index=source_index,
            target_index=target_index,
            candidate_edge_attr=candidate_edge_attr,
            edge_classifier=self.edge_classifier,
        )
        if return_attention:
            return logits, attention_records
        return logits

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        guide_features = self.sgrna_encoder(data["sgRNA"].x)
        target_features = self.target_observation_encoder(data["target_observation"].x)
        return torch.cat([guide_features, target_features], dim=0)


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


def graph_b_attention_edge_tensors(
    data: HeteroData,
    *,
    edge_feature_attrs: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_graph_b_data(data)
    edge_store = data[GRAPH_B_EDGE_TYPE]
    candidate = edge_store.edge_index
    candidate_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph B")
    sgrna_nodes = int(data["sgRNA"].num_nodes)
    candidate_forward = torch.stack([candidate[0], candidate[1] + sgrna_nodes], dim=0)
    candidate_reverse = torch.stack([candidate[1] + sgrna_nodes, candidate[0]], dim=0)
    similarity = data[GRAPH_B_SIMILARITY_EDGE_TYPE].edge_index
    if similarity.numel() > 0 and int(similarity.max()) >= sgrna_nodes:
        raise ValueError("Graph B similarity edge indices must remain within sgRNA nodes")
    similarity_reverse = torch.stack([similarity[1], similarity[0]], dim=0)
    zero_similarity_attr = candidate_attr.new_zeros((similarity.shape[1] * 2, candidate_attr.shape[1]))
    attention_edge_index = torch.cat([candidate_forward, candidate_reverse, similarity, similarity_reverse], dim=1)
    attention_edge_attr = torch.cat([candidate_attr, candidate_attr, zero_similarity_attr], dim=0)
    return attention_edge_index, attention_edge_attr


def graph_c_attention_edge_tensors(
    data: HeteroData,
    *,
    edge_feature_attrs: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_graph_c_data(data)
    edge_store = data[GRAPH_C_EDGE_TYPE]
    candidate = edge_store.edge_index
    candidate_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph C")
    sgrna_nodes = int(data["sgRNA"].num_nodes)
    candidate_forward = torch.stack([candidate[0], candidate[1] + sgrna_nodes], dim=0)
    candidate_reverse = torch.stack([candidate[1] + sgrna_nodes, candidate[0]], dim=0)
    context = data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index
    context_forward = context + sgrna_nodes
    context_reverse = torch.stack([context[1] + sgrna_nodes, context[0] + sgrna_nodes], dim=0)
    zero_context_attr = candidate_attr.new_zeros((context.shape[1] * 2, candidate_attr.shape[1]))
    attention_edge_index = torch.cat(
        [candidate_forward, candidate_reverse, context_forward, context_reverse],
        dim=1,
    )
    attention_edge_attr = torch.cat([candidate_attr, candidate_attr, zero_context_attr], dim=0)
    return attention_edge_index, attention_edge_attr


def _validate_attention_init(
    *,
    sgrna_input_dim: int,
    edge_input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    concat: bool,
    edge_aware_attention: bool,
    self_loop_edge_fill: float,
) -> None:
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
        raise ValueError("Sprint 7B edge-aware self-loop fill is frozen to 0.0")


def _gatv2_layers(
    *,
    hidden_dim: int,
    edge_dim: int | None,
    num_layers: int,
    heads: int,
    concat: bool,
    dropout: float,
    self_loop_edge_fill: float,
    gatv2_share_weights: bool,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    out_channels = hidden_dim // heads if concat else hidden_dim
    convs = nn.ModuleList()
    norms = nn.ModuleList()
    for _ in range(num_layers):
        convs.append(
            GATv2Conv(
                hidden_dim,
                out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=edge_dim,
                fill_value=self_loop_edge_fill,
                share_weights=gatv2_share_weights,
            )
        )
        norms.append(nn.LayerNorm(hidden_dim))
    return convs, norms


def _apply_attention_layers(
    x: torch.Tensor,
    convs: nn.ModuleList,
    norms: nn.ModuleList,
    dropout: nn.Dropout,
    *,
    attention_edge_index: torch.Tensor,
    attention_edge_attr: torch.Tensor | None,
    return_attention: bool,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    attention_records: list[dict[str, torch.Tensor]] = []
    for layer, (conv, norm) in enumerate(zip(convs, norms, strict=True)):
        if return_attention:
            x, weights = conv(
                x,
                attention_edge_index,
                edge_attr=attention_edge_attr,
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
            x = conv(x, attention_edge_index, edge_attr=attention_edge_attr)
        x = norm(x.relu())
        x = dropout(x)
    return x, attention_records


def _edge_classifier(*, hidden_dim: int, edge_input_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim * 4 + edge_input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _classify_candidate_edges(
    x: torch.Tensor,
    *,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    candidate_edge_attr: torch.Tensor,
    edge_classifier: nn.Module,
) -> torch.Tensor:
    source = x[source_index]
    target = x[target_index]
    pair = torch.cat(
        [source, target, source * target, torch.abs(source - target), candidate_edge_attr],
        dim=1,
    )
    return edge_classifier(pair).squeeze(-1)
