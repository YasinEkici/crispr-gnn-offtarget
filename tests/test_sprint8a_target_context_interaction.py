"""Sprint 8A Slice 1: target-context encoder delta tests (family gate + regularized branch).

Scope is the encoder only (`target_context_encoder.py`). The context-edge
interaction head (gat.py) and config dispatch (training/gcn.py) are later slices.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.models.target_context_encoder import (
    EXPERIMENTAL_EMPHASIS_BRANCH_DIMS,
    FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
    UNIFIED_DEEP_CONTEXT_ENCODER,
    FamilyAwareTargetContextEncoder,
    build_target_context_encoder,
    target_context_encoder_parameter_count,
)


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
