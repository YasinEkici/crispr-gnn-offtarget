"""Sprint 8A Slice 1: target-context encoder delta tests (family gate + regularized branch).

Scope is the encoder only (`target_context_encoder.py`). The context-edge
interaction head (gat.py) and config dispatch (training/gcn.py) are later slices.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.models.gat import (
    ContextEdgeInteractionHead,
    GraphCEdgeGATv2,
)
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from crispr_gnn.models.target_context_encoder import (
    EXPERIMENTAL_EMPHASIS_BRANCH_DIMS,
    FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
    UNIFIED_DEEP_CONTEXT_ENCODER,
    UNIFIED_SHALLOW_CONTEXT_ENCODER,
    FamilyAwareTargetContextEncoder,
    build_target_context_encoder,
    target_context_encoder_parameter_count,
)

EDGE_ATTRS = ["edge_attr_s5f2_energy"]


def _target_context_feature_names() -> list[str]:
    names = [
        f"feature__target_pos_{position:02d}_{base}"
        for position in range(23)
        for base in ("A", "C", "G", "T", "N")
    ]
    names.extend(f"feature__{name}" for name in EXPERIMENTAL_EPIGENETIC_FEATURES)
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        for suffix in ("mean", "std", "min", "max", "center", "pam_proximal_mean"):
            names.append(f"feature__{feature}_{suffix}")
        names.append(f"feature__{feature}_missing")
    return names


def _encoder(**overrides) -> FamilyAwareTargetContextEncoder:
    kwargs = dict(
        input_dim=212,
        hidden_dim=128,
        feature_names=_target_context_feature_names(),
        branch_dims=EXPERIMENTAL_EMPHASIS_BRANCH_DIMS,
        dropout=0.0,
        encoder_type=FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
    )
    kwargs.update(overrides)
    return FamilyAwareTargetContextEncoder(**kwargs)


def test_family_gate_forward_and_weights() -> None:
    torch.manual_seed(0)
    encoder = _encoder(family_gate=True, gate_reduction=4).eval()
    x = torch.randn(5, 212)

    output = encoder(x)
    assert output.shape == (5, 128)

    summary = encoder.activation_summary(x)
    assert len(summary) == 4  # 4-row family structure preserved
    assert all(row["family_gate_enabled"] for row in summary)
    for row in summary:
        weight = row["family_gate_weight_mean"]
        assert 0.0 < weight < 1.0  # sigmoid gate

    # gate produces exactly one scalar per family
    gates = encoder._gate_weights(encoder.branch_outputs(x))
    assert gates.shape == (5, 4)

    # gating adds the small excitation MLP only
    base_params = target_context_encoder_parameter_count(_encoder())
    gated_params = target_context_encoder_parameter_count(encoder)
    assert gated_params > base_params


def test_family_gate_off_reproduces_base_exactly() -> None:
    torch.manual_seed(0)
    base = _encoder().eval()
    off = _encoder(
        family_gate=False,
        experimental_branch_bottleneck=None,
        experimental_branch_feature_dropout=0.0,
    ).eval()
    off.load_state_dict(base.state_dict())  # identical structure when all flags OFF

    assert off.gate is None
    x = torch.randn(7, 212)
    assert torch.allclose(base(x), off(x), atol=1e-6)
    assert target_context_encoder_parameter_count(base) == target_context_encoder_parameter_count(off)


def test_regularized_experimental_branch_wiring() -> None:
    encoder = _encoder(experimental_branch_bottleneck=4, experimental_branch_feature_dropout=0.3)
    branch = encoder.branches["experimental_epigenetic"]
    linears = [layer for layer in branch if isinstance(layer, torch.nn.Linear)]
    dropouts = [layer for layer in branch if isinstance(layer, torch.nn.Dropout)]

    assert len(dropouts) == 1 and dropouts[0].p == pytest.approx(0.3)
    assert [lin.in_features for lin in linears] == [6, 4]  # 6 -> 4 -> 48
    assert [lin.out_features for lin in linears] == [4, 48]
    assert encoder(torch.randn(3, 212)).shape == (3, 128)

    # non-experimental branches stay the plain single-linear form
    seq_branch = encoder.branches["target_sequence_one_hot"]
    seq_linears = [layer for layer in seq_branch if isinstance(layer, torch.nn.Linear)]
    assert [lin.in_features for lin in seq_linears] == [115]


def test_experimental_feature_dropout_is_train_only() -> None:
    torch.manual_seed(0)
    encoder = _encoder(experimental_branch_feature_dropout=0.5)
    x = torch.randn(8, 212)

    encoder.train()
    assert not torch.allclose(encoder(x), encoder(x))  # stochastic in train mode

    encoder.eval()
    assert torch.allclose(encoder(x), encoder(x))  # deterministic in eval mode


def test_builder_rejects_sprint8a_flags_on_unified_encoder() -> None:
    with pytest.raises(ValueError, match="family-aware"):
        build_target_context_encoder(
            encoder_type=UNIFIED_DEEP_CONTEXT_ENCODER,
            input_dim=212,
            hidden_dim=128,
            dropout=0.0,
            feature_names=_target_context_feature_names(),
            family_gate=True,
        )


def test_builder_defaults_produce_ungated_encoder() -> None:
    encoder = build_target_context_encoder(
        encoder_type=FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
        input_dim=212,
        hidden_dim=128,
        dropout=0.0,
        feature_names=_target_context_feature_names(),
    )
    assert isinstance(encoder, FamilyAwareTargetContextEncoder)
    assert encoder.gate is None
    assert encoder.experimental_branch_bottleneck is None


# --- Slice 2: context-edge interaction head (gat.py / GraphCEdgeGATv2) ---


def _graph_c_model(**overrides) -> GraphCEdgeGATv2:
    kwargs = dict(
        sgrna_input_dim=4,
        target_observation_input_dim=212,
        edge_input_dim=2,
        hidden_dim=16,
        num_layers=1,
        heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_context_similarity_edges=True,
        target_context_encoder_type=UNIFIED_SHALLOW_CONTEXT_ENCODER,
    )
    kwargs.update(overrides)
    return GraphCEdgeGATv2(**kwargs)


def _fake_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_c_context_observation"
    data["sgRNA"].x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    data["target_observation"].x = torch.arange(636, dtype=torch.float32).reshape(3, 212) + 1.0
    data["target_observation"].feature_names = _target_context_feature_names()
    edge_store = data[GRAPH_C_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s5f2_energy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data


def test_context_edge_interaction_none_matches_base_and_builds_no_modules() -> None:
    torch.manual_seed(0)
    data = _fake_graph_c_view()
    base = _graph_c_model().eval()
    none_model = _graph_c_model(context_edge_interaction="none").eval()
    none_model.load_state_dict(base.state_dict())  # identical structure when interaction is OFF

    assert none_model.context_edge_interaction_head is None
    assert none_model.interaction_edge_classifier is None
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in none_model.parameters())
    assert torch.allclose(base(data, edge_feature_attrs=EDGE_ATTRS), none_model(data, edge_feature_attrs=EDGE_ATTRS))


def test_context_edge_interaction_does_not_change_message_passing() -> None:
    """Frozen-message-passing assertion: attention records are independent of the head."""
    torch.manual_seed(0)
    data = _fake_graph_c_view()
    none_model = _graph_c_model(context_edge_interaction="none").eval()
    film_model = _graph_c_model(context_edge_interaction="film", interaction_edge_dim=8).eval()
    # Share the message-passing submodules; only the head differs.
    for name in ("sgrna_encoder", "target_observation_encoder", "convs", "norms"):
        getattr(film_model, name).load_state_dict(getattr(none_model, name).state_dict())

    _, none_attn = none_model(data, edge_feature_attrs=EDGE_ATTRS, return_attention=True)
    _, film_attn = film_model(data, edge_feature_attrs=EDGE_ATTRS, return_attention=True)

    assert len(none_attn) == len(film_attn) == 1
    for left, right in zip(none_attn, film_attn, strict=True):
        assert torch.equal(left["edge_index"], right["edge_index"])
        assert torch.allclose(left["alpha"], right["alpha"], atol=1e-6)


def test_film_head_forward_and_summary() -> None:
    torch.manual_seed(0)
    data = _fake_graph_c_view()
    model = _graph_c_model(context_edge_interaction="film", interaction_edge_dim=8).eval()

    logits = model(data, edge_feature_attrs=EDGE_ATTRS)
    assert logits.shape == (3,)
    # classifier input = hidden*4 + interaction_edge_dim
    assert model.interaction_edge_classifier[0].in_features == 16 * 4 + 8

    summary = model.context_edge_interaction_summary(data, edge_feature_attrs=EDGE_ATTRS, split="test")
    assert len(summary) == 1
    row = summary[0]
    assert row["context_edge_interaction"] == "film"
    assert row["interaction_edge_dim"] == 8
    assert row["film_gamma_mean"] is not None and row["film_beta_mean"] is not None
    assert row["classifier_candidate_edge_attr_abs_sum"] > 0.0


def test_mlp_head_forward_and_summary() -> None:
    torch.manual_seed(0)
    data = _fake_graph_c_view()
    model = _graph_c_model(context_edge_interaction="mlp", interaction_edge_dim=8).eval()

    logits = model(data, edge_feature_attrs=EDGE_ATTRS)
    assert logits.shape == (3,)
    assert model.interaction_edge_classifier[0].in_features == 16 * 4 + 8

    summary = model.context_edge_interaction_summary(data, edge_feature_attrs=EDGE_ATTRS)
    assert summary[0]["context_edge_interaction"] == "mlp"
    assert summary[0]["film_gamma_mean"] is None  # no FiLM params for the MLP head
    assert summary[0]["interaction_vector_l2_mean"] >= 0.0


def test_invalid_context_edge_interaction_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported context_edge_interaction"):
        _graph_c_model(context_edge_interaction="bogus")


def test_context_edge_interaction_summary_empty_for_none() -> None:
    data = _fake_graph_c_view()
    model = _graph_c_model(context_edge_interaction="none").eval()
    assert model.context_edge_interaction_summary(data, edge_feature_attrs=EDGE_ATTRS) == []


def test_context_edge_interaction_head_film_params_shape() -> None:
    head = ContextEdgeInteractionHead(
        interaction="film", edge_input_dim=5, context_dim=16, interaction_edge_dim=8
    )
    edge = torch.randn(4, 5)
    context = torch.randn(4, 16)
    gamma, beta = head.film_params(context)
    assert gamma.shape == (4, 8) and beta.shape == (4, 8)
    assert head(edge, context).shape == (4, 8)
