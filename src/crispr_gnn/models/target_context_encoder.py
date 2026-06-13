"""Target-observation context encoders for Sprint 7F / Sprint 8A Graph C experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from crispr_gnn.features.target_context import (
    EXPERIMENTAL_EPIGENETIC_FAMILY,
    TARGET_CONTEXT_FAMILY_ORDER,
    target_context_feature_family,
    validate_target_context_feature_names,
)


UNIFIED_SHALLOW_CONTEXT_ENCODER = "unified_shallow"
UNIFIED_DEEP_CONTEXT_ENCODER = "unified_deep"
FAMILY_AWARE_CONTEXT_ENCODER = "family_aware"
FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER = "family_aware_experimental_emphasis"

SUPPORTED_TARGET_CONTEXT_ENCODERS = {
    UNIFIED_SHALLOW_CONTEXT_ENCODER,
    UNIFIED_DEEP_CONTEXT_ENCODER,
    FAMILY_AWARE_CONTEXT_ENCODER,
    FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
}

FAMILY_AWARE_BRANCH_DIMS = {
    "target_sequence_one_hot": 32,
    "experimental_epigenetic": 32,
    "computed_nucleosome_aggregates": 48,
    "computed_nucleosome_missingness": 16,
}

EXPERIMENTAL_EMPHASIS_BRANCH_DIMS = {
    "target_sequence_one_hot": 24,
    "experimental_epigenetic": 48,
    "computed_nucleosome_aggregates": 40,
    "computed_nucleosome_missingness": 16,
}


def build_target_context_encoder(
    *,
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    feature_names: Sequence[str] | None = None,
    family_gate: bool = False,
    gate_reduction: int = 4,
    experimental_branch_bottleneck: int | None = None,
    experimental_branch_feature_dropout: float = 0.0,
) -> nn.Module:
    """Build the predeclared Sprint 7F / Sprint 8A target-observation encoder.

    The Sprint 8A options (``family_gate``, ``experimental_branch_*``) default to
    OFF so the returned encoder reproduces the Sprint 7F path exactly. They are
    only valid for family-aware encoders; passing them with a unified encoder is
    rejected to avoid silently ignoring a misconfiguration.
    """
    normalized = str(encoder_type)
    if normalized not in SUPPORTED_TARGET_CONTEXT_ENCODERS:
        allowed = sorted(SUPPORTED_TARGET_CONTEXT_ENCODERS)
        raise ValueError(f"Unsupported target context encoder '{encoder_type}'. Allowed: {allowed}")
    is_family_aware = normalized in {
        FAMILY_AWARE_CONTEXT_ENCODER,
        FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER,
    }
    if not is_family_aware and (
        family_gate
        or experimental_branch_bottleneck is not None
        or experimental_branch_feature_dropout
    ):
        raise ValueError(
            "family_gate / experimental_branch_* options require a family-aware target context "
            f"encoder, not '{encoder_type}'"
        )
    if normalized == UNIFIED_SHALLOW_CONTEXT_ENCODER:
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
    if normalized == UNIFIED_DEEP_CONTEXT_ENCODER:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    branch_dims = (
        EXPERIMENTAL_EMPHASIS_BRANCH_DIMS
        if normalized == FAMILY_AWARE_EXPERIMENTAL_EMPHASIS_CONTEXT_ENCODER
        else FAMILY_AWARE_BRANCH_DIMS
    )
    if feature_names is None:
        raise ValueError("Family-aware target context encoders require target_observation feature_names")
    return FamilyAwareTargetContextEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        feature_names=feature_names,
        branch_dims=branch_dims,
        dropout=dropout,
        encoder_type=normalized,
        family_gate=family_gate,
        gate_reduction=gate_reduction,
        experimental_branch_bottleneck=experimental_branch_bottleneck,
        experimental_branch_feature_dropout=experimental_branch_feature_dropout,
    )


def target_context_family_indices(feature_names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Return stable target-observation column indexes by canonical feature family."""
    validate_target_context_feature_names(feature_names)
    grouped: dict[str, list[int]] = {family: [] for family in TARGET_CONTEXT_FAMILY_ORDER}
    for index, name in enumerate(feature_names):
        family = target_context_feature_family(name)
        if family in grouped:
            grouped[family].append(index)
    return {family: tuple(indexes) for family, indexes in grouped.items()}


class FamilyAwareTargetContextEncoder(nn.Module):
    """Encode Graph C target-observation features through predeclared family branches."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        feature_names: Sequence[str],
        branch_dims: dict[str, int],
        dropout: float,
        encoder_type: str,
        family_gate: bool = False,
        gate_reduction: int = 4,
        experimental_branch_bottleneck: int | None = None,
        experimental_branch_feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim != len(feature_names):
            raise ValueError("target_observation feature_names length must match input_dim")
        if gate_reduction < 1:
            raise ValueError("gate_reduction must be >= 1")
        if experimental_branch_bottleneck is not None and experimental_branch_bottleneck < 1:
            raise ValueError("experimental_branch_bottleneck must be >= 1 when set")
        if not 0.0 <= float(experimental_branch_feature_dropout) < 1.0:
            raise ValueError("experimental_branch_feature_dropout must be in [0, 1)")
        self.encoder_type = str(encoder_type)
        self.feature_names = tuple(str(name) for name in feature_names)
        self.family_indices = target_context_family_indices(self.feature_names)
        self.branch_dims = {family: int(branch_dims[family]) for family in TARGET_CONTEXT_FAMILY_ORDER}
        self.family_gate = bool(family_gate)
        self.gate_reduction = int(gate_reduction)
        self.experimental_branch_bottleneck = (
            int(experimental_branch_bottleneck) if experimental_branch_bottleneck is not None else None
        )
        self.experimental_branch_feature_dropout = float(experimental_branch_feature_dropout)
        self.branches = nn.ModuleDict(
            {family: self._build_branch(family) for family in TARGET_CONTEXT_FAMILY_ORDER}
        )
        fused_dim = sum(self.branch_dims.values())
        # Sprint 8A SENET-style learned family gate (predeclared variant: excitation
        # over the full concatenated branch output -> one scalar gate per family).
        self.gate: nn.Module | None = None
        if self.family_gate:
            gate_hidden = max(1, fused_dim // self.gate_reduction)
            self.gate = nn.Sequential(
                nn.Linear(fused_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, len(TARGET_CONTEXT_FAMILY_ORDER)),
                nn.Sigmoid(),
            )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
        )

    def _build_branch(self, family: str) -> nn.Module:
        in_dim = len(self.family_indices[family])
        out_dim = self.branch_dims[family]
        is_experimental = family == EXPERIMENTAL_EPIGENETIC_FAMILY
        layers: list[nn.Module] = []
        if is_experimental and self.experimental_branch_feature_dropout > 0.0:
            # Input feature-dropout on the brittle experimental epigenetic family;
            # nn.Dropout is train-mode-only and identity in eval.
            layers.append(nn.Dropout(self.experimental_branch_feature_dropout))
        if is_experimental and self.experimental_branch_bottleneck is not None:
            bottleneck = self.experimental_branch_bottleneck
            layers.extend(
                [
                    nn.Linear(in_dim, bottleneck),
                    nn.LayerNorm(bottleneck),
                    nn.ReLU(),
                    nn.Linear(bottleneck, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                ]
            )
        else:
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = self.branch_outputs(x)
        if self.gate is not None:
            gates = self._gate_weights(branch_outputs)
            branch_outputs = {
                family: branch_outputs[family] * gates[:, position : position + 1]
                for position, family in enumerate(TARGET_CONTEXT_FAMILY_ORDER)
            }
        fused = torch.cat([branch_outputs[family] for family in TARGET_CONTEXT_FAMILY_ORDER], dim=1)
        return self.fusion(fused)

    def branch_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for family in TARGET_CONTEXT_FAMILY_ORDER:
            index = torch.as_tensor(self.family_indices[family], dtype=torch.long, device=x.device)
            outputs[family] = self.branches[family](x[:, index])
        return outputs

    def _gate_weights(self, branch_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-family gate scalars of shape (num_nodes, num_families)."""
        if self.gate is None:
            raise RuntimeError("family gate is not enabled")
        concat = torch.cat([branch_outputs[family] for family in TARGET_CONTEXT_FAMILY_ORDER], dim=1)
        return self.gate(concat)

    def activation_summary(self, x: torch.Tensor) -> list[dict[str, object]]:
        """Return interpretation-only branch activation (and gate) summaries for one split.

        Branch activations are reported BEFORE gating to preserve the Sprint 7F
        semantics; the learned family gate weight is reported as an extra column.
        """
        with torch.no_grad():
            outputs = self.branch_outputs(x)
            gates = self._gate_weights(outputs) if self.gate is not None else None
        rows: list[dict[str, object]] = []
        for position, family in enumerate(TARGET_CONTEXT_FAMILY_ORDER):
            values = outputs[family].detach().float().cpu()
            gate_weight_mean = (
                float(gates[:, position].detach().float().cpu().mean()) if gates is not None else 1.0
            )
            rows.append(
                {
                    "target_context_encoder_type": self.encoder_type,
                    "target_context_family": family,
                    "input_columns": int(len(self.family_indices[family])),
                    "branch_dim": int(self.branch_dims[family]),
                    "activation_mean": float(values.mean()),
                    "activation_std": float(values.std(unbiased=False)),
                    "activation_l2_mean": float(values.norm(dim=1).mean()),
                    "family_gate_enabled": bool(self.gate is not None),
                    "family_gate_weight_mean": gate_weight_mean,
                }
            )
        return rows


def target_context_encoder_parameter_count(encoder: nn.Module) -> int:
    """Count only target-observation encoder parameters."""
    return int(sum(parameter.numel() for parameter in encoder.parameters()))
