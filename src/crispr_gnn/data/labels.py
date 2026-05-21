"""Label helpers for CRISPR off-target prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass


SCHEME_A_THRESHOLD = 1e-5
SCHEME_C_THRESHOLD = 1e-3


@dataclass(frozen=True)
class LabelScheme:
    name: str
    description: str
    threshold: float | None = None


LABEL_SCHEMES = {
    "scheme_a": LabelScheme(
        name="scheme_a",
        description="Paper-aligned binary label: cleavage_freq > 1e-5.",
        threshold=SCHEME_A_THRESHOLD,
    ),
    "scheme_b": LabelScheme(
        name="scheme_b",
        description="Paper comparison label based on transformed CA > -4.",
    ),
    "scheme_c": LabelScheme(
        name="scheme_c",
        description="High-confidence binary label: cleavage_freq > 1e-3.",
        threshold=SCHEME_C_THRESHOLD,
    ),
}


def is_label_eligible(value: float | int | None) -> bool:
    """Return True when a cleavage frequency can be used as a supervised label."""
    if value is None:
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(numeric_value)


def label_from_cleavage_freq(value: float | int | None, threshold: float = SCHEME_A_THRESHOLD) -> int:
    """Return a binary label for a cleavage frequency threshold.

    Missing or NaN cleavage frequencies are label-ineligible and must be filtered
    before supervised training/evaluation rather than silently mapped to zero.
    """
    if not is_label_eligible(value):
        raise ValueError("cleavage_freq is missing or NaN and cannot be converted to a binary label")
    return int(float(value) > threshold)


def labels_from_cleavage_freq(values: list[float | int | None], threshold: float = SCHEME_A_THRESHOLD) -> list[int]:
    """Vector-friendly wrapper that keeps the dependency surface small for Sprint 0."""
    return [label_from_cleavage_freq(value, threshold=threshold) for value in values]


def get_label_scheme(name: str) -> LabelScheme:
    try:
        return LABEL_SCHEMES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(LABEL_SCHEMES))
        raise ValueError(f"Unknown label scheme '{name}'. Valid schemes: {valid}") from exc
