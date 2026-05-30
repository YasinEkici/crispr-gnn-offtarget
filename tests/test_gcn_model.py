import torch
from torch_geometric.data import HeteroData

from crispr_gnn.graph.graph_schemas import GRAPH_A
from crispr_gnn.models.gcn import (
    GRAPH_A_EDGE_TYPE,
    TARGET_REPRESENTATION_POLICY,
    GraphAEdgeGCN,
    graph_a_edge_feature_attrs,
    graph_a_feature_dimensions,
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
