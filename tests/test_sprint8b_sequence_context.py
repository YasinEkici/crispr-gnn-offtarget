"""Sprint 8B Slice 1: sequence-context encoder + S1 reconstruction + leakage audit.

Scope is the encoder module only (`sequence_context_encoder.py`). Graph dispatch
and late-fusion wiring are later slices.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.features.sequence import build_sequence_pair_encoding
from crispr_gnn.models.sequence_context_encoder import (
    S1_NUM_CHANNELS,
    S1_NUM_POSITIONS,
    SequenceContextEncoder,
    build_s1_pair_for_edges,
    build_s1_pair_from_onehot,
    resolve_s1_onehot_indices,
    sequence_input_audit,
)


def _guide_feature_names() -> list[str]:
    return [f"feature__guide_pos_{p:02d}_{b}" for p in range(23) for b in ("A", "C", "G", "T", "N")]


def _target_feature_names() -> list[str]:
    # Mirrors the Graph C target_observation 212-column layout: 115 target one-hot + non-sequence context.
    names = [f"feature__target_pos_{p:02d}_{b}" for p in range(23) for b in ("A", "C", "G", "T", "N")]
    names.extend(f"feature__{name}" for name in EXPERIMENTAL_EPIGENETIC_FEATURES)
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        for suffix in ("mean", "std", "min", "max", "center", "pam_proximal_mean"):
            names.append(f"feature__{feature}_{suffix}")
        names.append(f"feature__{feature}_missing")
    return names


def _onehot_from_sequence(seq: str, length: int = 23) -> torch.Tensor:
    bases = "ACGTN"
    out = torch.zeros(length * 5, dtype=torch.float32)
    for pos in range(length):
        base = seq[pos] if pos < len(seq) else "N"
        base = base if base in bases else "N"
        out[pos * 5 + bases.index(base)] = 1.0
    return out


def test_s1_reconstruction_matches_sprint2_builder() -> None:
    df = pd.DataFrame(
        {
            "grna_target_sequence": ["ACGTACGTACGTACGTACGTACG", "ACGTACGTACGT"],  # 2nd is short -> N padding
            "target_sequence": ["ACGTTCGTACGAACGTACGTACG", "TGCATGCATGCA"],
        }
    )
    sprint2 = build_sequence_pair_encoding(df).encoded  # (2, 23, 11)
    sprint2_t = torch.from_numpy(sprint2)
    guide_onehot = sprint2_t[:, :, 0:5].reshape(2, 115)
    target_onehot = sprint2_t[:, :, 5:10].reshape(2, 115)

    rebuilt = build_s1_pair_from_onehot(guide_onehot, target_onehot)
    assert rebuilt.shape == (2, S1_NUM_POSITIONS, S1_NUM_CHANNELS)
    # Reconstruction reproduces the Sprint 2 S1 tensor exactly, incl. the mismatch channel.
    assert torch.allclose(rebuilt, sprint2_t)


def test_build_s1_pair_layout_and_mismatch() -> None:
    guide = torch.stack([_onehot_from_sequence("A" * 23), _onehot_from_sequence("A" * 23)])
    target = torch.stack([_onehot_from_sequence("G" * 23), _onehot_from_sequence("A" * 22 + "N")])
    s1 = build_s1_pair_from_onehot(guide, target)
    assert s1.shape == (2, 23, 11)
    # row0: all A vs all G -> every position mismatch
    assert torch.allclose(s1[0, :, 10], torch.ones(23))
    # row1: all A vs all A except last N -> no mismatch anywhere (A==A; last is N->0)
    assert torch.allclose(s1[1, :, 10], torch.zeros(23))
    # guide channels (0-4) and target channels (5-9) carry the one-hots
    assert s1[0, 0, 0] == 1.0 and s1[0, 0, 5 + 2] == 1.0  # guide A, target G


def test_encoder_forward_shape_and_determinism() -> None:
    torch.manual_seed(0)
    encoder = SequenceContextEncoder(embed_dim=64).eval()
    s1 = torch.rand(7, S1_NUM_POSITIONS, S1_NUM_CHANNELS)
    out = encoder(s1)
    assert out.shape == (7, 64)
    assert torch.allclose(encoder(s1), out)  # deterministic in eval


def test_encoder_rejects_wrong_shape() -> None:
    encoder = SequenceContextEncoder()
    with pytest.raises(ValueError, match="expects"):
        encoder(torch.rand(3, 23, 9))  # wrong channel count


def test_sequence_input_audit_isolates_sequence_only() -> None:
    target_names = _target_feature_names()
    assert len(target_names) == 212
    # resolve selects exactly the 115 target one-hot columns out of 212 (no epi/nucleosome).
    idx = resolve_s1_onehot_indices(target_names, kind="target")
    assert len(idx) == 115
    audit = sequence_input_audit(guide_feature_names=_guide_feature_names(), target_feature_names=target_names)
    assert audit["guide_onehot_columns"] == 115 and audit["target_onehot_columns"] == 115
    assert audit["channels"] == 11


def test_resolve_raises_on_incomplete_onehot() -> None:
    truncated = _guide_feature_names()[:-5]  # drop the last position's bases
    with pytest.raises(ValueError, match="incomplete"):
        resolve_s1_onehot_indices(truncated, kind="guide")


def test_build_s1_pair_for_edges_gathers_per_edge() -> None:
    guide_names = _guide_feature_names()
    target_names = _target_feature_names()
    # node 0 guide = all A, node 1 guide = all C
    guide_node_x = torch.stack([_onehot_from_sequence("A" * 23), _onehot_from_sequence("C" * 23)])
    # target nodes carry 212 cols; set the target one-hot portion, leave context zero
    t0 = torch.zeros(212)
    t0[: 115] = _onehot_from_sequence("A" * 23)  # target node 0 = all A (matches guide A -> no mismatch)
    t1 = torch.zeros(212)
    t1[: 115] = _onehot_from_sequence("G" * 23)  # target node 1 = all G
    t2 = torch.zeros(212)
    t2[: 115] = _onehot_from_sequence("T" * 23)
    target_node_x = torch.stack([t0, t1, t2])
    edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)  # (g0->t0, g1->t1, g0->t2)

    s1 = build_s1_pair_for_edges(
        guide_node_x=guide_node_x,
        guide_feature_names=guide_names,
        target_node_x=target_node_x,
        target_feature_names=target_names,
        edge_index=edge_index,
    )
    assert s1.shape == (3, 23, 11)
    assert torch.allclose(s1[0, :, 10], torch.zeros(23))  # g0(A) vs t0(A) -> no mismatch
    assert torch.allclose(s1[1, :, 10], torch.ones(23))   # g1(C) vs t1(G) -> mismatch
    assert torch.allclose(s1[2, :, 10], torch.ones(23))   # g0(A) vs t2(T) -> mismatch
