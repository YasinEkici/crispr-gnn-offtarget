import torch
from torch_geometric.data import HeteroData

from crispr_gnn.graph.graph_schemas import GRAPH_B, GRAPH_C
from crispr_gnn.models.gat import (
    GraphBEdgeGATv2,
    GraphCEdgeGATv2,
    graph_b_attention_edge_tensors,
    graph_c_attention_edge_tensors,
)
from crispr_gnn.models.gcn import (
    GRAPH_B_EDGE_TYPE,
    GRAPH_B_SIMILARITY_EDGE_TYPE,
    GRAPH_C_CONTEXT_EDGE_TYPE,
    GRAPH_C_EDGE_TYPE,
    graph_b_edge_feature_attrs,
    graph_c_edge_feature_attrs,
)


def test_graph_b_gatv2_edge_tensors_zero_fill_similarity_edges() -> None:
    data = _tiny_graph_b_view()
    attrs = graph_b_edge_feature_attrs(["s5f2_energy"])

    edge_index, edge_attr = graph_b_attention_edge_tensors(data, edge_feature_attrs=attrs)

    assert edge_index.shape == (2, 8)
    assert edge_attr.shape == (8, 4)
    torch.testing.assert_close(edge_attr[:3], data[GRAPH_B_EDGE_TYPE].edge_attr_s5f2_energy)
    torch.testing.assert_close(edge_attr[3:6], data[GRAPH_B_EDGE_TYPE].edge_attr_s5f2_energy)
    torch.testing.assert_close(edge_attr[6:], torch.zeros((2, 4), dtype=torch.float32))


def test_graph_b_gatv2_forward_outputs_logits_and_attention() -> None:
    data = _tiny_graph_b_view()
    attrs = graph_b_edge_feature_attrs(["s5f2_energy"])
    model = GraphBEdgeGATv2(
        sgrna_input_dim=4,
        edge_input_dim=4,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        attention_dropout=0.0,
    )

    logits, attention_records = model(data, edge_feature_attrs=attrs, return_attention=True)

    assert logits.shape == (3,)
    assert model.edge_aware_attention is True
    assert model.convs[0].edge_dim == 4
    assert attention_records[0]["alpha"].shape[1] == 2


def test_graph_c_gatv2_edge_tensors_zero_fill_context_edges() -> None:
    data = _tiny_graph_c_view()
    attrs = graph_c_edge_feature_attrs(["s5f2_energy"])

    edge_index, edge_attr = graph_c_attention_edge_tensors(data, edge_feature_attrs=attrs)

    assert edge_index.shape == (2, 10)
    assert edge_attr.shape == (10, 4)
    torch.testing.assert_close(edge_attr[:3], data[GRAPH_C_EDGE_TYPE].edge_attr_s5f2_energy)
    torch.testing.assert_close(edge_attr[3:6], data[GRAPH_C_EDGE_TYPE].edge_attr_s5f2_energy)
    torch.testing.assert_close(edge_attr[6:], torch.zeros((4, 4), dtype=torch.float32))


def test_graph_c_gatv2_forward_outputs_logits_and_attention() -> None:
    data = _tiny_graph_c_view()
    attrs = graph_c_edge_feature_attrs(["s5f2_energy"])
    model = GraphCEdgeGATv2(
        sgrna_input_dim=4,
        target_observation_input_dim=5,
        edge_input_dim=4,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        attention_dropout=0.0,
    )

    logits, attention_records = model(data, edge_feature_attrs=attrs, return_attention=True)

    assert logits.shape == (3,)
    assert model.edge_aware_attention is True
    assert model.convs[0].edge_dim == 4
    assert attention_records[0]["alpha"].shape[1] == 2


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
    edge_store.edge_attr_s5f2_energy = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    sim_store = data[GRAPH_B_SIMILARITY_EDGE_TYPE]
    sim_store.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
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
    edge_store.edge_attr_s5f2_energy = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    context_store = data[GRAPH_C_CONTEXT_EDGE_TYPE]
    context_store.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data
