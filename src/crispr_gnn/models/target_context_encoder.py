"""Target-observation context encoders for Sprint 7F Graph C experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from crispr_gnn.features.target_context import (
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
) -> nn.Module:
    """Build the predeclared Sprint 7F target-observation encoder."""
    normalized = str(encoder_type)
    if normalized not in SUPPORTED_TARGET_CONTEXT_ENCODERS:
        allowed = sorted(SUPPORTED_TARGET_CONTEXT_ENCODERS)
        raise ValueError(f"Unsupported target context encoder '{encoder_type}'. Allowed: {allowed}")
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
    ) -> None:
        super().__init__()
        if input_dim != len(feature_names):
            raise ValueError("target_observation feature_names length must match input_dim")
        self.encoder_type = str(encoder_type)
        self.feature_names = tuple(str(name) for name in feature_names)
        self.family_indices = target_context_family_indices(self.feature_names)
        self.branch_dims = {family: int(branch_dims[family]) for family in TARGET_CONTEXT_FAMILY_ORDER}
        self.branches = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.Linear(len(self.family_indices[family]), self.branch_dims[family]),
                    nn.LayerNorm(self.branch_dims[family]),
                    nn.ReLU(),
                )
                for family in TARGET_CONTEXT_FAMILY_ORDER
            }
        )
        fused_dim = sum(self.branch_dims.values())
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = self.branch_outputs(x)
        fused = torch.cat([branch_outputs[family] for family in TARGET_CONTEXT_FAMILY_ORDER], dim=1)
        return self.fusion(fused)

    def branch_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for family in TARGET_CONTEXT_FAMILY_ORDER:
            index = torch.as_tensor(self.family_indices[family], dtype=torch.long, device=x.device)
            outputs[family] = self.branches[family](x[:, index])
        return outputs

    def activation_summary(self, x: torch.Tensor) -> list[dict[str, object]]:
        """Return interpretation-only branch activation summaries for one split."""
        with torch.no_grad():
            outputs = self.branch_outputs(x)
        rows: list[dict[str, object]] = []
        for family in TARGET_CONTEXT_FAMILY_ORDER:
            values = outputs[family].detach().float().cpu()
            rows.append(
                {
                    "target_context_encoder_type": self.encoder_type,
                    "target_context_family": family,
                    "input_columns": int(len(self.family_indices[family])),
                    "branch_dim": int(self.branch_dims[family]),
                    "activation_mean": float(values.mean()),
                    "activation_std": float(values.std(unbiased=False)),
                    "activation_l2_mean": float(values.norm(dim=1).mean()),
                }
            )
        return rows


def target_context_encoder_parameter_count(encoder: nn.Module) -> int:
    """Count only target-observation encoder parameters."""
    return int(sum(parameter.numel() for parameter in encoder.parameters()))
