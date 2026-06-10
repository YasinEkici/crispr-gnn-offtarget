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
from crispr_gnn.models.target_context_encoder import (
    UNIFIED_SHALLOW_CONTEXT_ENCODER,
    build_target_context_encoder,
    target_context_encoder_parameter_count,
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


CONTEXT_EDGE_INTERACTION_NONE = "none"
CONTEXT_EDGE_INTERACTION_FILM = "film"
CONTEXT_EDGE_INTERACTION_MLP = "mlp"
SUPPORTED_CONTEXT_EDGE_INTERACTIONS = {
    CONTEXT_EDGE_INTERACTION_NONE,
    CONTEXT_EDGE_INTERACTION_FILM,
    CONTEXT_EDGE_INTERACTION_MLP,
}


class ContextEdgeInteractionHead(nn.Module):
    """Sprint 8A head-only interaction between the target-context embedding and the
    candidate ``S5F2_energy`` edge features.

    Applied in the Graph C edge-classifier head only; the frozen GATv2
    attention/message passing is unchanged. The candidate edge features are first
    embedded (``edge_input_dim -> interaction_edge_dim``) and then either
    FiLM-modulated by the (post-GATv2) target-context embedding or combined with it
    through a small interaction MLP. Output dim is ``interaction_edge_dim``.
    """

    def __init__(
        self,
        *,
        interaction: str,
        edge_input_dim: int,
        context_dim: int,
        interaction_edge_dim: int,
    ) -> None:
        super().__init__()
        if interaction not in {CONTEXT_EDGE_INTERACTION_FILM, CONTEXT_EDGE_INTERACTION_MLP}:
            raise ValueError("ContextEdgeInteractionHead requires interaction 'film' or 'mlp'")
        if edge_input_dim <= 0:
            raise ValueError("ContextEdgeInteractionHead requires positive edge_input_dim")
        if interaction_edge_dim <= 0:
            raise ValueError("interaction_edge_dim must be positive")
        self.interaction = str(interaction)
        self.interaction_edge_dim = int(interaction_edge_dim)
        self.edge_encoder = nn.Linear(edge_input_dim, interaction_edge_dim)
        if self.interaction == CONTEXT_EDGE_INTERACTION_FILM:
            self.film_generator = nn.Linear(context_dim, 2 * interaction_edge_dim)
        else:
            self.context_proj = nn.Linear(context_dim, interaction_edge_dim)
            self.mlp = nn.Sequential(
                nn.Linear(2 * interaction_edge_dim + context_dim, interaction_edge_dim),
                nn.ReLU(),
            )

    def film_params(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-edge FiLM scale/shift ``(gamma, beta)`` from the context embedding."""
        if self.interaction != CONTEXT_EDGE_INTERACTION_FILM:
            raise RuntimeError("film_params is only defined for the FiLM interaction")
        gamma, beta = self.film_generator(context).chunk(2, dim=-1)
        return gamma, beta

    def forward(self, candidate_edge_attr: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        edge_embed = self.edge_encoder(candidate_edge_attr)
        if self.interaction == CONTEXT_EDGE_INTERACTION_FILM:
            gamma, beta = self.film_params(context)
            return gamma * edge_embed + beta
        proj = self.context_proj(context)
        fused = torch.cat([edge_embed, context, edge_embed * proj], dim=1)
        return self.mlp(fused)


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
        drop_context_similarity_edges: bool = False,
        edge_blind_candidate_attention: bool = False,
        mask_target_observation_features: bool = False,
        target_observation_mask_indices: Sequence[int] | None = None,
        target_context_encoder_type: str = UNIFIED_SHALLOW_CONTEXT_ENCODER,
        target_context_feature_names: Sequence[str] | None = None,
        context_edge_interaction: str = CONTEXT_EDGE_INTERACTION_NONE,
        interaction_edge_dim: int = 64,
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
        self.drop_context_similarity_edges = bool(drop_context_similarity_edges)
        self.edge_blind_candidate_attention = bool(edge_blind_candidate_attention)
        self.mask_target_observation_features = bool(mask_target_observation_features)
        self.target_context_encoder_type = str(target_context_encoder_type)
        self.target_context_feature_names = tuple(str(name) for name in (target_context_feature_names or ()))
        self.target_observation_mask_indices = tuple(int(index) for index in (target_observation_mask_indices or ()))
        if self.mask_target_observation_features and self.target_observation_mask_indices:
            raise ValueError("Use either full target-observation masking or column-index masking, not both")
        if self.target_observation_mask_indices:
            if min(self.target_observation_mask_indices) < 0:
                raise ValueError("target_observation_mask_indices cannot contain negative indexes")
            if max(self.target_observation_mask_indices) >= target_observation_input_dim:
                raise ValueError("target_observation_mask_indices exceed target_observation_input_dim")
        self.sgrna_encoder = nn.Sequential(nn.Linear(sgrna_input_dim, hidden_dim), nn.ReLU())
        self.target_observation_encoder = build_target_context_encoder(
            encoder_type=self.target_context_encoder_type,
            input_dim=target_observation_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            feature_names=self.target_context_feature_names or None,
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
        # Sprint 8A head-only context-edge interaction. Default 'none' reproduces
        # the Sprint 7F path exactly; the interaction modules are built only when
        # enabled so the 'none' parameter set / state_dict is unchanged.
        self.context_edge_interaction = str(context_edge_interaction)
        if self.context_edge_interaction not in SUPPORTED_CONTEXT_EDGE_INTERACTIONS:
            allowed = sorted(SUPPORTED_CONTEXT_EDGE_INTERACTIONS)
            raise ValueError(
                f"Unsupported context_edge_interaction '{context_edge_interaction}'. Allowed: {allowed}"
            )
        self.interaction_edge_dim = int(interaction_edge_dim)
        self.context_edge_interaction_head: ContextEdgeInteractionHead | None = None
        self.interaction_edge_classifier: nn.Sequential | None = None
        if self.context_edge_interaction != CONTEXT_EDGE_INTERACTION_NONE:
            if self.interaction_edge_dim <= 0:
                raise ValueError("interaction_edge_dim must be positive when context_edge_interaction is enabled")
            self.context_edge_interaction_head = ContextEdgeInteractionHead(
                interaction=self.context_edge_interaction,
                edge_input_dim=edge_input_dim,
                context_dim=hidden_dim,
                interaction_edge_dim=self.interaction_edge_dim,
            )
            self.interaction_edge_classifier = _edge_classifier(
                hidden_dim=hidden_dim, edge_input_dim=self.interaction_edge_dim, dropout=dropout
            )

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
            include_context_edges=not self.drop_context_similarity_edges,
            zero_candidate_attention_attr=self.edge_blind_candidate_attention,
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
        if self.context_edge_interaction == CONTEXT_EDGE_INTERACTION_NONE:
            logits = _classify_candidate_edges(
                x,
                source_index=source_index,
                target_index=target_index,
                candidate_edge_attr=candidate_edge_attr,
                edge_classifier=self.edge_classifier,
            )
        else:
            logits = self._classify_with_interaction(
                x,
                source_index=source_index,
                target_index=target_index,
                candidate_edge_attr=candidate_edge_attr,
            )
        if return_attention:
            return logits, attention_records
        return logits

    def _classify_with_interaction(
        self,
        x: torch.Tensor,
        *,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        candidate_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-C-specific edge classification with a head-only context-edge interaction.

        Mirrors ``_classify_candidate_edges`` but replaces the raw candidate edge
        features with the interaction vector (the post-GATv2 target embedding is the
        interaction context). The shared helper is left untouched.
        """
        if self.context_edge_interaction_head is None or self.interaction_edge_classifier is None:
            raise RuntimeError("context-edge interaction head is not enabled")
        source = x[source_index]
        target = x[target_index]
        interaction_vector = self.context_edge_interaction_head(candidate_edge_attr, target)
        pair = torch.cat(
            [source, target, source * target, torch.abs(source - target), interaction_vector],
            dim=1,
        )
        return self.interaction_edge_classifier(pair).squeeze(-1)

    def _initial_node_features(self, data: HeteroData) -> torch.Tensor:
        guide_features = self.sgrna_encoder(data["sgRNA"].x)
        target_x = data["target_observation"].x
        if self.mask_target_observation_features:
            target_x = torch.zeros_like(target_x)
        elif self.target_observation_mask_indices:
            target_x = target_x.clone()
            mask_index = torch.as_tensor(self.target_observation_mask_indices, dtype=torch.long, device=target_x.device)
            target_x[:, mask_index] = 0.0
        target_features = self.target_observation_encoder(target_x)
        return torch.cat([guide_features, target_features], dim=0)

    def target_context_encoder_activation_summary(self, target_x: torch.Tensor) -> list[dict[str, object]]:
        """Return interpretation-only target encoder activation summaries."""
        encoder = self.target_observation_encoder
        if hasattr(encoder, "activation_summary"):
            rows = encoder.activation_summary(target_x)  # type: ignore[attr-defined]
        else:
            with torch.no_grad():
                values = encoder(target_x).detach().float().cpu()
            rows = [
                {
                    "target_context_encoder_type": self.target_context_encoder_type,
                    "target_context_family": "all_target_observation_features",
                    "input_columns": int(target_x.shape[1]),
                    "branch_dim": int(values.shape[1]),
                    "activation_mean": float(values.mean()),
                    "activation_std": float(values.std(unbiased=False)),
                    "activation_l2_mean": float(values.norm(dim=1).mean()),
                }
            ]
        parameter_count = target_context_encoder_parameter_count(encoder)
        for row in rows:
            row["target_context_encoder_parameter_count"] = parameter_count
        return rows

    def context_edge_interaction_summary(
        self,
        data: HeteroData,
        *,
        edge_feature_attrs: Sequence[str],
        split: str = "test",
    ) -> list[dict[str, object]]:
        """Return interpretation-only context-edge interaction summaries for one split.

        Reproduces the frozen GATv2 propagation read-only (no module is modified) to
        obtain the post-GATv2 target embedding used as the interaction context, then
        summarizes the FiLM scale/shift (FiLM only) and the interaction-vector norm.
        Returns an empty list when the interaction is disabled.
        """
        if self.context_edge_interaction == CONTEXT_EDGE_INTERACTION_NONE:
            return []
        head = self.context_edge_interaction_head
        if head is None:
            return []
        _validate_graph_c_data(data)
        edge_store = data[GRAPH_C_EDGE_TYPE]
        candidate_edge_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph C")
        with torch.no_grad():
            x = self._initial_node_features(data)
            attention_edge_index, attention_edge_attr = graph_c_attention_edge_tensors(
                data,
                edge_feature_attrs=edge_feature_attrs,
                include_context_edges=not self.drop_context_similarity_edges,
                zero_candidate_attention_attr=self.edge_blind_candidate_attention,
            )
            x, _ = _apply_attention_layers(
                x,
                self.convs,
                self.norms,
                self.dropout,
                attention_edge_index=attention_edge_index,
                attention_edge_attr=attention_edge_attr if self.edge_aware_attention else None,
                return_attention=False,
            )
            target_index = edge_store.edge_index[1] + int(data["sgRNA"].num_nodes)
            context = x[target_index]
            edge_embed = head.edge_encoder(candidate_edge_attr)
            interaction_vector = head(candidate_edge_attr, context)
            row: dict[str, object] = {
                "split": split,
                "context_edge_interaction": self.context_edge_interaction,
                "interaction_edge_dim": int(self.interaction_edge_dim),
                "candidate_edges": int(candidate_edge_attr.shape[0]),
                "edge_embed_l2_mean": float(edge_embed.norm(dim=1).mean()),
                "interaction_vector_l2_mean": float(interaction_vector.norm(dim=1).mean()),
                "classifier_candidate_edge_attr_abs_sum": float(candidate_edge_attr.abs().sum()),
                "film_gamma_mean": None,
                "film_gamma_std": None,
                "film_beta_mean": None,
                "film_beta_std": None,
            }
            if self.context_edge_interaction == CONTEXT_EDGE_INTERACTION_FILM:
                gamma, beta = head.film_params(context)
                row["film_gamma_mean"] = float(gamma.mean())
                row["film_gamma_std"] = float(gamma.std(unbiased=False))
                row["film_beta_mean"] = float(beta.mean())
                row["film_beta_std"] = float(beta.std(unbiased=False))
        return [row]


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
    include_context_edges: bool = True,
    zero_candidate_attention_attr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_graph_c_data(data)
    edge_store = data[GRAPH_C_EDGE_TYPE]
    candidate = edge_store.edge_index
    candidate_attr = _candidate_edge_features(edge_store, edge_feature_attrs, graph_label="Graph C")
    if zero_candidate_attention_attr:
        candidate_attention_attr = torch.zeros_like(candidate_attr)
    else:
        candidate_attention_attr = candidate_attr
    sgrna_nodes = int(data["sgRNA"].num_nodes)
    candidate_forward = torch.stack([candidate[0], candidate[1] + sgrna_nodes], dim=0)
    candidate_reverse = torch.stack([candidate[1] + sgrna_nodes, candidate[0]], dim=0)
    if not include_context_edges:
        attention_edge_index = torch.cat([candidate_forward, candidate_reverse], dim=1)
        attention_edge_attr = torch.cat([candidate_attention_attr, candidate_attention_attr], dim=0)
        return attention_edge_index, attention_edge_attr
    context = data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index
    context_forward = context + sgrna_nodes
    context_reverse = torch.stack([context[1] + sgrna_nodes, context[0] + sgrna_nodes], dim=0)
    zero_context_attr = candidate_attr.new_zeros((context.shape[1] * 2, candidate_attr.shape[1]))
    attention_edge_index = torch.cat(
        [candidate_forward, candidate_reverse, context_forward, context_reverse],
        dim=1,
    )
    attention_edge_attr = torch.cat([candidate_attention_attr, candidate_attention_attr, zero_context_attr], dim=0)
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
