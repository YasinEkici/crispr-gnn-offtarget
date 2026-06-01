import torch
from torch_geometric.data import HeteroData

from crispr_gnn.graph.graph_schemas import GRAPH_A, GRAPH_B, GRAPH_C
from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    GRAPH_B_EDGE_TYPE,
    GRAPH_B_SIMILARITY_EDGE_TYPE,
    GRAPH_C_CONTEXT_EDGE_TYPE,
    GRAPH_C_EDGE_TYPE,
    GRAPH_C_TARGET_REPRESENTATION_POLICY,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    GraphBEdgeGCN,
    GraphCEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
    graph_b_edge_feature_attrs,
    graph_b_feature_dimensions,
    graph_c_edge_feature_attrs,
    graph_c_feature_dimensions,
)


def test_graph_a_gcn_forward_outputs_one_logit_per_candidate_edge() -> None:
    data = _tiny_graph_a_view()
    attrs = graph_a_edge_feature_attrs(["s1_pair", "f1"])
    sgrna_dim, edge_dim = graph_a_feature_dimensions(data, attrs)
    model = GraphAEdgeGCN(
        sgrna_input_dim=sgrna_dim,
        edge_input_dim=edge_dim,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    logits = model(data, edge_feature_attrs=attrs)

    assert logits.shape == (3,)
    assert model.target_representation_policy == TARGET_REPRESENTATION_POLICY
    assert model.physical_target_type.shape == (1, 8)
    assert not any("target_id" in name for name, _ in model.named_parameters())


def test_graph_a_gcn_rejects_physical_target_features() -> None:
    data = _tiny_graph_a_view()
    data["physical_target_site"].x = torch.ones((2, 3), dtype=torch.float32)
    model = GraphAEdgeGCN(sgrna_input_dim=4, edge_input_dim=3, hidden_dim=8, num_layers=1)

    try:
        model(data, edge_feature_attrs=["edge_attr_s1_pair", "edge_attr_f1"])
    except ValueError as exc:
        assert "featureless" in str(exc)
    else:
        raise AssertionError("Graph A model accepted physical target features")


def test_graph_b_gcn_forward_outputs_one_logit_per_candidate_edge() -> None:
    data = _tiny_graph_b_view()
    attrs = graph_b_edge_feature_attrs(["s1_pair", "f1"])
    sgrna_dim, edge_dim = graph_b_feature_dimensions(data, attrs)
    model = GraphBEdgeGCN(
        sgrna_input_dim=sgrna_dim,
        edge_input_dim=edge_dim,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    logits = model(data, edge_feature_attrs=attrs)

    assert logits.shape == (3,)
    assert model.target_representation_policy == TARGET_REPRESENTATION_POLICY
    assert model.physical_target_type.shape == (1, 8)
    assert not any("target_id" in name for name, _ in model.named_parameters())


def test_graph_b_gcn_similarity_edges_affect_forward_pass() -> None:
    """Verify that sequence_similar_to edges actually change the model output.

    GraphBEdgeGCN and GraphAEdgeGCN differ only in that Graph B passes candidate
    AND similarity edges through GCNConv. With the same initial weights and the
    same candidate edges, adding similarity edges changes degree normalization and
    neighborhood aggregation — so logits must not be identical.
    """
    data_b = _tiny_graph_b_view()
    data_a = _tiny_graph_a_view()
    attrs_b = graph_b_edge_feature_attrs(["s1_pair", "f1"])
    attrs_a = graph_a_edge_feature_attrs(["s1_pair", "f1"])
    sgrna_dim_b, edge_dim_b = graph_b_feature_dimensions(data_b, attrs_b)
    sgrna_dim_a, edge_dim_a = graph_a_feature_dimensions(data_a, attrs_a)
    assert sgrna_dim_b == sgrna_dim_a and edge_dim_b == edge_dim_a

    torch.manual_seed(0)
    model_b = GraphBEdgeGCN(sgrna_input_dim=sgrna_dim_b, edge_input_dim=edge_dim_b, hidden_dim=8, num_layers=1, dropout=0.0)
    torch.manual_seed(0)
    model_a = GraphAEdgeGCN(sgrna_input_dim=sgrna_dim_a, edge_input_dim=edge_dim_a, hidden_dim=8, num_layers=1, dropout=0.0)

    with torch.no_grad():
        logits_b = model_b(data_b, edge_feature_attrs=attrs_b)
        logits_a = model_a(data_a, edge_feature_attrs=attrs_a)

    assert not torch.allclose(logits_b, logits_a), (
        "Graph B logits must differ from Graph A — similarity edges must be used in message passing"
    )


def test_graph_b_gcn_rejects_physical_target_features() -> None:
    data = _tiny_graph_b_view()
    data["physical_target_site"].x = torch.ones((2, 3), dtype=torch.float32)
    model = GraphBEdgeGCN(sgrna_input_dim=4, edge_input_dim=3, hidden_dim=8, num_layers=1)

    try:
        model(data, edge_feature_attrs=["edge_attr_s1_pair", "edge_attr_f1"])
    except ValueError as exc:
        assert "featureless" in str(exc)
    else:
        raise AssertionError("Graph B model accepted physical target features")


def test_graph_b_gcn_rejects_missing_similarity_edges() -> None:
    data = _tiny_graph_b_view()
    del data[GRAPH_B_SIMILARITY_EDGE_TYPE]
    model = GraphBEdgeGCN(sgrna_input_dim=4, edge_input_dim=3, hidden_dim=8, num_layers=1)

    try:
        model(data, edge_feature_attrs=["edge_attr_s1_pair", "edge_attr_f1"])
    except ValueError as exc:
        assert "sequence_similar_to" in str(exc)
    else:
        raise AssertionError("Graph B model accepted missing similarity edges")


def test_graph_c_gcn_forward_uses_target_observation_context_encoder() -> None:
    data = _tiny_graph_c_view()
    attrs = graph_c_edge_feature_attrs(["candidate_pair_features"])
    sgrna_dim, target_dim, edge_dim = graph_c_feature_dimensions(data, attrs)
    model = GraphCEdgeGCN(
        sgrna_input_dim=sgrna_dim,
        target_observation_input_dim=target_dim,
        edge_input_dim=edge_dim,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    logits = model(data, edge_feature_attrs=attrs)

    assert logits.shape == (3,)
    assert model.target_representation_policy == GRAPH_C_TARGET_REPRESENTATION_POLICY
    assert hasattr(model, "target_observation_encoder")
    assert not any("physical_target" in name for name, _ in model.named_parameters())


def test_graph_c_gcn_rejects_missing_target_observation_features() -> None:
    data = _tiny_graph_c_view()
    del data["target_observation"].x
    model = GraphCEdgeGCN(
        sgrna_input_dim=4,
        target_observation_input_dim=5,
        edge_input_dim=3,
        hidden_dim=8,
        num_layers=1,
    )

    try:
        model(data, edge_feature_attrs=["edge_attr_candidate_pair_features"])
    except ValueError as exc:
        assert "context features" in str(exc)
    else:
        raise AssertionError("Graph C model accepted missing target_observation context features")


def _tiny_graph_a_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = GRAPH_A
    data["sgRNA"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    data["physical_target_site"].num_nodes = 2
    edge_store = data[GRAPH_A_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s1_pair = torch.ones((3, 2), dtype=torch.float32)
    edge_store.edge_attr_f1 = torch.zeros((3, 1), dtype=torch.float32)
    return data


def _tiny_graph_b_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = GRAPH_B
    data["sgRNA"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    data["physical_target_site"].num_nodes = 2
    edge_store = data[GRAPH_B_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s1_pair = torch.ones((3, 2), dtype=torch.float32)
    edge_store.edge_attr_f1 = torch.zeros((3, 1), dtype=torch.float32)
    sim_store = data[GRAPH_B_SIMILARITY_EDGE_TYPE]
    sim_store.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    sim_store.edge_attr = torch.ones((1, 1), dtype=torch.float32)
    return data


def _tiny_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = GRAPH_C
    data["sgRNA"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    data["target_observation"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.2, 0.3],
            [0.0, 1.0, 0.0, 0.4, 0.5],
            [0.0, 0.0, 1.0, 0.6, 0.7],
        ],
        dtype=torch.float32,
    )
    edge_store = data[GRAPH_C_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_candidate_pair_features = torch.ones((3, 3), dtype=torch.float32)
    context_store = data[GRAPH_C_CONTEXT_EDGE_TYPE]
    context_store.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    context_store.edge_attr = torch.ones((2, 1), dtype=torch.float32)
    return data
