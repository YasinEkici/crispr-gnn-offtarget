"""Model definitions."""

from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
)

__all__ = [
    "GRAPH_A_EDGE_TYPE",
    "TARGET_REPRESENTATION_POLICY",
    "GraphAEdgeGCN",
    "graph_a_edge_feature_attrs",
    "graph_a_feature_dimensions",
]
