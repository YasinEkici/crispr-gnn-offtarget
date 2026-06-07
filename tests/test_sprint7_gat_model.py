import pytest
import torch
from torch_geometric.data import HeteroData

from crispr_gnn.graph.graph_schemas import GRAPH_A
from crispr_gnn.models.gat import GraphAEdgeGAT, GraphAEdgeGATv2, graph_a_attention_edge_tensors
from crispr_gnn.models.gcn import GRAPH_A_EDGE_TYPE, TARGET_REPRESENTATION_POLICY, graph_a_edge_feature_attrs


def test_graph_a_gat_edge_tensors_duplicate_forward_features_for_reverse_edges() -> None:
    data = _tiny_graph_a_view()
    attrs = graph_a_edge_feature_attrs(["s5f2_energy"])

    edge_index, edge_attr = graph_a_attention_edge_tensors(data, edge_feature_attrs=attrs)

    assert edge_index.shape == (2, 6)
    assert edge_attr.shape == (6, 4)
    torch.testing.assert_close(edge_attr[:3], data[GRAPH_A_EDGE_TYPE].edge_attr_s5f2_energy)
    torch.testing.assert_close(edge_attr[3:], data[GRAPH_A_EDGE_TYPE].edge_attr_s5f2_energy)
    assert edge_index[:, :3].tolist() == [[0, 1, 0], [2, 3, 3]]
    assert edge_index[:, 3:].tolist() == [[2, 3, 3], [0, 1, 0]]


@pytest.mark.parametrize("model_cls", [GraphAEdgeGAT, GraphAEdgeGATv2])
def test_graph_a_edge_aware_attention_model_forward_outputs_one_logit_per_candidate_edge(model_cls) -> None:
    data = _tiny_graph_a_view()
    attrs = graph_a_edge_feature_attrs(["s5f2_energy"])
    model = model_cls(
        sgrna_input_dim=4,
        edge_input_dim=4,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        concat=True,
        dropout=0.0,
        attention_dropout=0.0,
        edge_aware_attention=True,
        self_loop_edge_fill=0.0,
    )

    logits, attention_records = model(data, edge_feature_attrs=attrs, return_attention=True)

    assert logits.shape == (3,)
    assert model.target_representation_policy == TARGET_REPRESENTATION_POLICY
    assert model.edge_aware_attention is True
    assert model.convs[0].edge_dim == 4
    assert model.convs[0].fill_value == 0.0
    assert attention_records
    assert attention_records[0]["alpha"].shape[1] == 2


def test_graph_a_edge_blind_attention_control_omits_edge_dim() -> None:
    data = _tiny_graph_a_view()
    attrs = graph_a_edge_feature_attrs(["s5f2_energy"])
    model = GraphAEdgeGAT(
        sgrna_input_dim=4,
        edge_input_dim=4,
        hidden_dim=8,
        num_layers=1,
        heads=2,
        dropout=0.0,
        edge_aware_attention=False,
        self_loop_edge_fill=0.0,
    )

    logits = model(data, edge_feature_attrs=attrs)

    assert logits.shape == (3,)
    assert model.edge_aware_attention is False
    assert model.convs[0].edge_dim is None


def test_graph_a_gat_rejects_nonzero_self_loop_edge_fill() -> None:
    with pytest.raises(ValueError, match="self-loop fill is frozen to 0.0"):
        GraphAEdgeGAT(
            sgrna_input_dim=4,
            edge_input_dim=4,
            hidden_dim=8,
            num_layers=1,
            heads=2,
            self_loop_edge_fill=1.0,
        )


def test_graph_a_gat_rejects_physical_target_features() -> None:
    data = _tiny_graph_a_view()
    data["physical_target_site"].x = torch.ones((2, 3), dtype=torch.float32)
    attrs = graph_a_edge_feature_attrs(["s5f2_energy"])
    model = GraphAEdgeGAT(sgrna_input_dim=4, edge_input_dim=4, hidden_dim=8, num_layers=1, heads=2)

    with pytest.raises(ValueError, match="featureless"):
        model(data, edge_feature_attrs=attrs)


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
    edge_store.edge_attr_s5f2_energy = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    return data
