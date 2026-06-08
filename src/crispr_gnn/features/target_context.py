"""Target-observation feature family helpers for Sprint 7E."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES


TARGET_SEQUENCE_FAMILY = "target_sequence_one_hot"
EXPERIMENTAL_EPIGENETIC_FAMILY = "experimental_epigenetic"
COMPUTED_NUCLEOSOME_AGGREGATE_FAMILY = "computed_nucleosome_aggregates"
COMPUTED_NUCLEOSOME_MISSINGNESS_FAMILY = "computed_nucleosome_missingness"
ALL_NONSEQUENCE_CONTEXT_FAMILY = "all_nonsequence_context"

TARGET_CONTEXT_FAMILY_ORDER = (
    TARGET_SEQUENCE_FAMILY,
    EXPERIMENTAL_EPIGENETIC_FAMILY,
    COMPUTED_NUCLEOSOME_AGGREGATE_FAMILY,
    COMPUTED_NUCLEOSOME_MISSINGNESS_FAMILY,
)

TARGET_CONTEXT_EXPECTED_COUNTS = {
    TARGET_SEQUENCE_FAMILY: 115,
    EXPERIMENTAL_EPIGENETIC_FAMILY: 6,
    COMPUTED_NUCLEOSOME_AGGREGATE_FAMILY: 78,
    COMPUTED_NUCLEOSOME_MISSINGNESS_FAMILY: 13,
}

TARGET_CONTEXT_GROUP_ALIASES = {
    ALL_NONSEQUENCE_CONTEXT_FAMILY: (
        EXPERIMENTAL_EPIGENETIC_FAMILY,
        COMPUTED_NUCLEOSOME_AGGREGATE_FAMILY,
        COMPUTED_NUCLEOSOME_MISSINGNESS_FAMILY,
    ),
}

_TARGET_SEQUENCE_RE = re.compile(r"^target_pos_\d{2}_[ACGTN]$")


def normalize_target_context_feature_name(feature_name: str) -> str:
    """Remove the serialized feature prefix used by graph artifacts."""
    return str(feature_name).removeprefix("feature__")


def target_context_feature_family(feature_name: str) -> str:
    """Classify one Graph C target-observation feature column."""
    name = normalize_target_context_feature_name(feature_name)
    if _TARGET_SEQUENCE_RE.match(name):
        return TARGET_SEQUENCE_FAMILY
    if name in EXPERIMENTAL_EPIGENETIC_FEATURES:
        return EXPERIMENTAL_EPIGENETIC_FAMILY
    for computed_feature in COMPUTED_NUCLEOSOME_FEATURES:
        if name == f"{computed_feature}_missing":
            return COMPUTED_NUCLEOSOME_MISSINGNESS_FAMILY
        if name.startswith(f"{computed_feature}_"):
            return COMPUTED_NUCLEOSOME_AGGREGATE_FAMILY
    return "unknown"


def expand_target_context_families(families: Iterable[str]) -> tuple[str, ...]:
    """Expand shorthand group names to canonical target-context families."""
    expanded: list[str] = []
    for family in families:
        normalized = str(family)
        if normalized in TARGET_CONTEXT_GROUP_ALIASES:
            expanded.extend(TARGET_CONTEXT_GROUP_ALIASES[normalized])
        else:
            expanded.append(normalized)
    return tuple(dict.fromkeys(expanded))


def target_context_family_counts(feature_names: Sequence[str]) -> dict[str, int]:
    counts = Counter(target_context_feature_family(name) for name in feature_names)
    return {family: int(counts.get(family, 0)) for family in (*TARGET_CONTEXT_FAMILY_ORDER, "unknown")}


def validate_target_context_feature_names(feature_names: Sequence[str]) -> None:
    """Assert the canonical Sprint 7E target-observation feature-family contract."""
    counts = target_context_family_counts(feature_names)
    expected_total = sum(TARGET_CONTEXT_EXPECTED_COUNTS.values())
    if len(feature_names) != expected_total:
        raise ValueError(f"Expected {expected_total} target-observation features, found {len(feature_names)}")
    for family, expected in TARGET_CONTEXT_EXPECTED_COUNTS.items():
        actual = counts.get(family, 0)
        if actual != expected:
            raise ValueError(f"Target-observation feature count drift for {family}: expected {expected}, found {actual}")
    if counts.get("unknown", 0):
        unknown = [
            name
            for name in feature_names
            if target_context_feature_family(name) == "unknown"
        ]
        raise ValueError(f"Unknown target-observation feature columns: {unknown[:10]}")


def target_context_mask_indices(feature_names: Sequence[str], families: Iterable[str]) -> tuple[int, ...]:
    """Return stable column indexes for one or more target-context families."""
    expanded = expand_target_context_families(families)
    invalid = sorted(set(expanded).difference(TARGET_CONTEXT_FAMILY_ORDER))
    if invalid:
        raise ValueError(f"Unknown target-observation mask families: {invalid}")
    indexes = tuple(
        index
        for index, name in enumerate(feature_names)
        if target_context_feature_family(name) in expanded
    )
    if not indexes:
        raise ValueError(f"No target-observation columns matched families: {list(families)}")
    return indexes
