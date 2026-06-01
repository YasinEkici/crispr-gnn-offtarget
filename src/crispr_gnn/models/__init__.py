"""Model definitions."""

from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    GRAPH_C_EDGE_TYPE,
    GRAPH_C_TARGET_REPRESENTATION_POLICY,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    GraphCEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
    graph_c_edge_feature_attrs,
    graph_c_feature_dimensions,
)

__all__ = [
    "GRAPH_A_EDGE_TYPE",
    "GRAPH_C_EDGE_TYPE",
    "GRAPH_C_TARGET_REPRESENTATION_POLICY",
    "TARGET_REPRESENTATION_POLICY",
    "GraphAEdgeGCN",
    "GraphCEdgeGCN",
    "graph_a_edge_feature_attrs",
    "graph_a_feature_dimensions",
    "graph_c_edge_feature_attrs",
    "graph_c_feature_dimensions",
]
