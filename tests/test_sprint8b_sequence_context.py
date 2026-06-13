"""Sprint 8B Slice 1: sequence-context encoder + S1 reconstruction + leakage audit.

Scope is the encoder module only (`sequence_context_encoder.py`). Graph dispatch
and late-fusion wiring are later slices.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES
from crispr_gnn.features.sequence import build_sequence_pair_encoding
from crispr_gnn.models.gat import GraphCEdgeGATv2, GraphCSequenceOnlyClassifier
from crispr_gnn.models.gcn import GRAPH_C_CONTEXT_EDGE_TYPE, GRAPH_C_EDGE_TYPE
from crispr_gnn.models.sequence_context_encoder import (
    S1_NUM_CHANNELS,
    S1_NUM_POSITIONS,
    SequenceContextEncoder,
    build_s1_pair_for_edges,
    build_s1_pair_from_onehot,
    resolve_s1_onehot_indices,
    sequence_input_audit,
)
from crispr_gnn.training.gcn import _build_model, _result_row, gcn_run_config_from_mapping


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


def _sequence_graph_c_view() -> HeteroData:
    data = HeteroData()
    data.graph_name = "graph_c_context_observation"
    data["sgRNA"].x = torch.stack([_onehot_from_sequence("A" * 23), _onehot_from_sequence("C" * 23)])
    data["sgRNA"].feature_names = _guide_feature_names()
    target_rows = []
    for sequence in ("A" * 23, "G" * 23, "T" * 23):
        row = torch.zeros(212)
        row[:115] = _onehot_from_sequence(sequence)
        target_rows.append(row)
    data["target_observation"].x = torch.stack(target_rows)
    data["target_observation"].feature_names = _target_feature_names()
    edge_store = data[GRAPH_C_EDGE_TYPE]
    edge_store.edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]], dtype=torch.long)
    edge_store.edge_label = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    edge_store.supervision_mask = torch.tensor([True, True, True])
    edge_store.edge_attr_s5f2_energy = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)
    data[GRAPH_C_CONTEXT_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return data


def _sequence_graph_c_config(*, sequence_mode: str, context_edge_interaction: str = "none") -> dict:
    return {
        "sprint": "sprint8b",
        "seed": 42,
        "data": {"split_id": "sprint2_main_seed42", "label_scheme": "scheme_a"},
        "graph": {"schema": "graph_c_context_observation", "visibility_policy": "strict_inductive_primary"},
        "features": {"edge_feature_sets": ["s5f2_energy"], "feature_set": "S5F2_energy"},
        "model": {
            "name": f"s8b_{sequence_mode}",
            "architecture": "gatv2",
            "hidden_dim": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "target_node_representation": "target_observation_context_encoder",
            "attention": {
                "heads": 2,
                "concat": True,
                "dropout": 0.0,
                "edge_aware": True,
                "drop_context_similarity_edges": True,
            },
            "context_edge_interaction": context_edge_interaction,
            "interaction_edge_dim": 8,
            "sequence_context_encoder": {
                "mode": sequence_mode,
                "embed_dim": 5,
                "conv_channels": 4,
                "lstm_hidden": 3,
                "dropout": 0.0,
            },
        },
        "training": {"loss": "weighted_bce"},
    }


def test_config_maps_sprint8b_sequence_context_keys() -> None:
    parsed = gcn_run_config_from_mapping(
        _sequence_graph_c_config(sequence_mode="late_fusion", context_edge_interaction="film")
    )
    assert parsed.sequence_context_mode == "late_fusion"
    assert parsed.sequence_embed_dim == 5
    assert parsed.sequence_conv_channels == 4
    assert parsed.sequence_lstm_hidden == 3
    assert parsed.sequence_dropout == 0.0


def test_config_rejects_sequence_only_with_context_edge_interaction() -> None:
    with pytest.raises(ValueError, match="sequence_only"):
        gcn_run_config_from_mapping(
            _sequence_graph_c_config(sequence_mode="sequence_only", context_edge_interaction="film")
        )


def test_build_model_dispatches_sequence_only_path() -> None:
    view = _sequence_graph_c_view()
    config = gcn_run_config_from_mapping(_sequence_graph_c_config(sequence_mode="sequence_only"))
    model, edge_dim = _build_model(view, config, ["edge_attr_s5f2_energy"])
    assert isinstance(model, GraphCSequenceOnlyClassifier)
    assert edge_dim == 2
    assert model.sequence_input_audit_summary()["policy"].startswith("guide one-hot")
    logits = model.eval()(view, edge_feature_attrs=["edge_attr_s5f2_energy"])
    assert logits.shape == (3,)


def test_late_fusion_appends_zero_sequence_without_changing_context_prefix() -> None:
    torch.manual_seed(0)
    view = _sequence_graph_c_view()
    model = GraphCEdgeGATv2(
        sgrna_input_dim=115,
        target_observation_input_dim=212,
        edge_input_dim=2,
        hidden_dim=16,
        num_layers=1,
        heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        drop_context_similarity_edges=True,
        context_edge_interaction="film",
        interaction_edge_dim=8,
        sequence_context_mode="late_fusion",
        guide_feature_names=_guide_feature_names(),
        target_context_feature_names=_target_feature_names(),
        sequence_embed_dim=5,
        sequence_conv_channels=4,
        sequence_lstm_hidden=3,
        sequence_dropout=0.0,
    ).eval()

    context_only = model.classifier_input_snapshot(
        view,
        edge_feature_attrs=["edge_attr_s5f2_energy"],
        include_sequence_embedding=False,
    )
    zero_sequence = model.classifier_input_snapshot(
        view,
        edge_feature_attrs=["edge_attr_s5f2_energy"],
        zero_sequence_embedding=True,
    )
    assert zero_sequence.shape[1] == context_only.shape[1] + 5
    assert torch.allclose(zero_sequence[:, : context_only.shape[1]], context_only)
    assert torch.allclose(zero_sequence[:, context_only.shape[1] :], torch.zeros(3, 5))
    assert model.interaction_edge_classifier is not None
    assert model.interaction_edge_classifier[0].in_features == 16 * 4 + 8 + 5


def test_build_model_dispatches_late_fusion_path() -> None:
    view = _sequence_graph_c_view()
    config = gcn_run_config_from_mapping(
        _sequence_graph_c_config(sequence_mode="late_fusion", context_edge_interaction="film")
    )
    model, _edge_dim = _build_model(view, config, ["edge_attr_s5f2_energy"])
    assert isinstance(model, GraphCEdgeGATv2)
    assert model.sequence_context_mode == "late_fusion"
    assert model.sequence_encoder is not None
    assert model.context_edge_interaction == "film"


def test_sequence_only_result_row_does_not_report_candidate_edge_features() -> None:
    config = gcn_run_config_from_mapping(_sequence_graph_c_config(sequence_mode="sequence_only"))

    class _FakeMaterialized:
        manifest = {"graph_name": "graph_c_context_observation", "split_id": "sprint2_main_seed42"}

    row = _result_row(
        config=config,
        materialized=_FakeMaterialized(),
        val_labels=torch.tensor([0, 1, 1]).numpy(),
        val_scores=torch.tensor([0.2, 0.8, 0.9]).numpy(),
        test_labels=torch.tensor([0, 1, 1]).numpy(),
        test_scores=torch.tensor([0.1, 0.7, 0.95]).numpy(),
        threshold=0.5,
        threshold_policy="validation_max_f1",
        best_epoch=1,
        epochs_ran=1,
        best_val_auprc=1.0,
        edge_dim=2,
        parameter_count=100,
        active_parameter_count=100,
        pos_weight=1.0,
    )
    assert row["sequence_context_mode"] == "sequence_only"
    assert row["edge_feature_sets"] == ""
    assert row["edge_feature_columns"] == 0
    assert row["target_context_encoder_type"] is None
    assert "no context/message-passing/candidate-edge-feature signal" in row["notes"]
