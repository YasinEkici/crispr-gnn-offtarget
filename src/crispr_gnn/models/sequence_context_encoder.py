"""Sprint 8B sequence-context encoder (CRISPR-Net / CRISPR-IP-adapted, from scratch).

The encoder is a small 1D-Conv (local mismatch/identity features) + BiLSTM
(positional features) over the aligned sgRNA/target ``S1`` pair, producing a
fixed-dimension ``seq_embed`` per candidate edge. It is **adapted** from CRISPR-Net
(Inception-conv + BiLSTM) and CRISPR-IP (CNN + BiLSTM + attention); attention is
deferred to limit scope/overtuning. This is **not a reproduction**: the data
(measured-only Mak/crisprSQL), split (guide-disjoint ``sprint2_main_seed42``),
label (``scheme_a``), and primary metric (AUPRC at ~90% positive prevalence) all
differ from the source papers.

Sprint 8B Slice-0 decision (a): the ``S1`` pair is reconstructed deterministically
from the frozen Graph C one-hot node features (guide one-hot on ``sgRNA`` nodes +
target one-hot on ``target_observation`` nodes + a computed mismatch channel),
**not** from a raw-data edge-id join. The channel layout matches the Sprint 2
``build_sequence_pair_encoding`` 23x11 contract (5 guide + 5 target + 1 mismatch).
The sequence branch carries sequence-only signal; ``sequence_input_audit`` proves
no energy/epigenetic/context column is used.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from torch import nn

from crispr_gnn.features.sequence import BASE_TO_INDEX, BASES


S1_NUM_POSITIONS = 23
S1_NUM_BASES = len(BASES)  # 5 (A, C, G, T, N)
S1_NUM_CHANNELS = 2 * S1_NUM_BASES + 1  # 11: guide one-hot, target one-hot, mismatch
_N_INDEX = BASE_TO_INDEX["N"]

_GUIDE_POS_RE = re.compile(r"^guide_pos_(\d{2})_([ACGTN])$")
_TARGET_POS_RE = re.compile(r"^target_pos_(\d{2})_([ACGTN])$")
# Tokens that must never appear in a column selected as sequence input (leakage guard).
_NONSEQUENCE_TOKENS = (
    "energy",
    "epigen",
    "mnase",
    "drip",
    "dnase",
    "h3k",
    "nucleosome",
    "_missing",
    "_mean",
    "_std",
    "_center",
    "pam_proximal",
)


def _normalize(name: object) -> str:
    return str(name).removeprefix("feature__")


def resolve_s1_onehot_indices(feature_names: Sequence[str], *, kind: str) -> tuple[int, ...]:
    """Return column indexes for the guide/target one-hot, ordered by (position, base).

    Selecting in canonical (position, base) order makes the downstream reshape
    correct regardless of the original column order, and validates that the full
    23x5 one-hot is present.
    """
    if kind not in {"guide", "target"}:
        raise ValueError("kind must be 'guide' or 'target'")
    regex = _GUIDE_POS_RE if kind == "guide" else _TARGET_POS_RE
    found: dict[tuple[int, int], int] = {}
    for index, name in enumerate(feature_names):
        match = regex.match(_normalize(name))
        if match:
            position = int(match.group(1))
            base_index = BASE_TO_INDEX[match.group(2)]
            found[(position, base_index)] = index
    expected = [(position, base) for position in range(S1_NUM_POSITIONS) for base in range(S1_NUM_BASES)]
    missing = [key for key in expected if key not in found]
    if missing:
        raise ValueError(
            f"{kind} one-hot is incomplete: missing {len(missing)} (position, base) entries, e.g. {missing[:3]}"
        )
    return tuple(found[key] for key in expected)


def sequence_input_audit(
    *,
    guide_feature_names: Sequence[str],
    target_feature_names: Sequence[str],
) -> dict[str, object]:
    """Validate and document that the S1 input is sequence-only (no leakage).

    Raises if the guide/target one-hot cannot be resolved or if any selected column
    carries a non-sequence (energy/epigenetic/nucleosome/context) token.
    """
    guide_indices = resolve_s1_onehot_indices(guide_feature_names, kind="guide")
    target_indices = resolve_s1_onehot_indices(target_feature_names, kind="target")
    selected = [_normalize(guide_feature_names[i]) for i in guide_indices]
    selected += [_normalize(target_feature_names[i]) for i in target_indices]
    leaked = [name for name in selected if any(token in name.lower() for token in _NONSEQUENCE_TOKENS)]
    if leaked:
        raise ValueError(f"Sequence-input leakage: non-sequence columns selected: {leaked[:5]}")
    return {
        "representation": "S1_sequence_pair_from_graph_c_onehot",
        "positions": S1_NUM_POSITIONS,
        "channels": S1_NUM_CHANNELS,
        "guide_onehot_columns": len(guide_indices),
        "target_onehot_columns": len(target_indices),
        "mismatch_channel": 1,
        "policy": "guide one-hot + target one-hot + aligned mismatch; sequence-only (no energy/epigenetic/context)",
        "source": "reconstructed from frozen Graph C node one-hot features (no raw-data join)",
    }


def build_s1_pair_from_onehot(guide_onehot: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Build the (N, 23, 11) S1 tensor from per-row guide/target one-hot (N, 115) each.

    Channels: ``[guide A,C,G,T,N | target A,C,G,T,N | mismatch]``. The mismatch
    channel follows the Sprint 2 convention: 1 where guide and target bases differ
    and neither is N (and both positions carry a base).
    """
    flat = S1_NUM_POSITIONS * S1_NUM_BASES
    if guide_onehot.shape[-1] != flat or target_onehot.shape[-1] != flat:
        raise ValueError(f"guide/target one-hot must have {flat} columns (23x5), got {guide_onehot.shape[-1]} / {target_onehot.shape[-1]}")
    if guide_onehot.shape[0] != target_onehot.shape[0]:
        raise ValueError("guide and target one-hot must have the same number of rows")
    rows = guide_onehot.shape[0]
    guide = guide_onehot.reshape(rows, S1_NUM_POSITIONS, S1_NUM_BASES).float()
    target = target_onehot.reshape(rows, S1_NUM_POSITIONS, S1_NUM_BASES).float()
    has_guide = guide.sum(dim=-1) > 0
    has_target = target.sum(dim=-1) > 0
    guide_base = guide.argmax(dim=-1)
    target_base = target.argmax(dim=-1)
    mismatch = (
        has_guide
        & has_target
        & (guide_base != _N_INDEX)
        & (target_base != _N_INDEX)
        & (guide_base != target_base)
    ).float().unsqueeze(-1)
    return torch.cat([guide, target, mismatch], dim=-1)


def build_s1_pair_for_edges(
    *,
    guide_node_x: torch.Tensor,
    guide_feature_names: Sequence[str],
    target_node_x: torch.Tensor,
    target_feature_names: Sequence[str],
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the per-candidate-edge S1 tensor from Graph C node one-hot features.

    ``edge_index[0]`` indexes ``sgRNA`` nodes (guide), ``edge_index[1]`` indexes
    ``target_observation`` nodes (target). Runs the sequence-input audit first so a
    leakage attempt fails loudly.
    """
    sequence_input_audit(guide_feature_names=guide_feature_names, target_feature_names=target_feature_names)
    guide_idx = torch.as_tensor(resolve_s1_onehot_indices(guide_feature_names, kind="guide"), dtype=torch.long)
    target_idx = torch.as_tensor(resolve_s1_onehot_indices(target_feature_names, kind="target"), dtype=torch.long)
    source = edge_index[0]
    target = edge_index[1]
    guide_onehot = guide_node_x[source][:, guide_idx]
    target_onehot = target_node_x[target][:, target_idx]
    return build_s1_pair_from_onehot(guide_onehot, target_onehot)


class SequenceContextEncoder(nn.Module):
    """CRISPR-Net/CRISPR-IP-adapted Conv + BiLSTM encoder over the S1 pair.

    Input: ``(N, 23, 11)`` S1 tensor. Output: ``(N, embed_dim)`` sequence embedding.
    Defaults are Slice-1 starting values; Sprint 8B config (Slice 3) may override.
    """

    def __init__(
        self,
        *,
        in_channels: int = S1_NUM_CHANNELS,
        seq_length: int = S1_NUM_POSITIONS,
        conv_channels: int = 32,
        conv_kernel: int = 3,
        lstm_hidden: int = 32,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or seq_length <= 0 or embed_dim <= 0:
            raise ValueError("in_channels, seq_length, embed_dim must be positive")
        self.in_channels = int(in_channels)
        self.seq_length = int(seq_length)
        self.embed_dim = int(embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.project = nn.Sequential(
            nn.Linear(2 * lstm_hidden, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, s1: torch.Tensor) -> torch.Tensor:
        if s1.dim() != 3 or s1.shape[1] != self.seq_length or s1.shape[2] != self.in_channels:
            raise ValueError(
                f"SequenceContextEncoder expects (N, {self.seq_length}, {self.in_channels}); got {tuple(s1.shape)}"
            )
        x = s1.transpose(1, 2)  # (N, channels, length) for Conv1d
        x = self.conv(x)
        x = x.transpose(1, 2)  # (N, length, conv_channels) for BiLSTM
        sequence_out, _ = self.bilstm(x)
        pooled = sequence_out.mean(dim=1)  # order-robust fixed-dim aggregation over positions
        return self.project(pooled)
