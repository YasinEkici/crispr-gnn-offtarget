"""Sprint 5 Graph A feature-ablation builders.

Sprint 5 keeps Graph A topology fixed and changes only candidate-pair edge
features. S5F0 is intentionally stricter than Sprint 4's S1_pair: it contains
guide and target one-hot sequence channels only, without the mismatch channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from crispr_gnn.data.parsers import parse_numeric_array_result
from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.features.tabular import (
    FORBIDDEN_PREDICTIVE_COLUMNS,
    TrainOnlyPreprocessor,
    build_binding_energy_features,
    build_computed_nucleosome_features,
    build_experimental_epigenetic_features,
    build_sequence_mismatch_features,
)


Sprint5FeatureSetName = Literal[
    "S5F0_seq",
    "S5F1_mismatch",
    "S5F2_energy",
    "S5F3_experimental_epi",
    "S5F4_computed_agg",
    "S5F5_computed_pos",
]
SPRINT5_FEATURE_SET_ORDER: tuple[Sprint5FeatureSetName, ...] = (
    "S5F0_seq",
    "S5F1_mismatch",
    "S5F2_energy",
    "S5F3_experimental_epi",
    "S5F4_computed_agg",
    "S5F5_computed_pos",
)
SPRINT5_FEATURE_SET_ALIASES = {name.lower(): name for name in SPRINT5_FEATURE_SET_ORDER}


@dataclass(frozen=True)
class Sprint5FeatureSetSpec:
    name: Sprint5FeatureSetName
    description: str
    families: tuple[str, ...]


SPRINT5_FEATURE_SET_SPECS: dict[Sprint5FeatureSetName, Sprint5FeatureSetSpec] = {
    "S5F0_seq": Sprint5FeatureSetSpec(
        name="S5F0_seq",
        description="Guide and target sequence one-hot channels only; no explicit mismatch channel.",
        families=("sequence_one_hot",),
    ),
    "S5F1_mismatch": Sprint5FeatureSetSpec(
        name="S5F1_mismatch",
        description="S5F0 plus engineered sequence and mismatch-position features.",
        families=("sequence_one_hot", "sequence_summary", "mismatch_position"),
    ),
    "S5F2_energy": Sprint5FeatureSetSpec(
        name="S5F2_energy",
        description="S5F1 plus binding-energy scalar features.",
        families=("sequence_one_hot", "sequence_summary", "mismatch_position", "binding_energy"),
    ),
    "S5F3_experimental_epi": Sprint5FeatureSetSpec(
        name="S5F3_experimental_epi",
        description="S5F2 plus experimental epigenetic scalar features.",
        families=("sequence_one_hot", "sequence_summary", "mismatch_position", "binding_energy", "experimental_epigenetic"),
    ),
    "S5F4_computed_agg": Sprint5FeatureSetSpec(
        name="S5F4_computed_agg",
        description="S5F3 plus computed nucleosome aggregate and missingness features.",
        families=(
            "sequence_one_hot",
            "sequence_summary",
            "mismatch_position",
            "binding_energy",
            "experimental_epigenetic",
            "computed_nucleosome_aggregates",
            "computed_nucleosome_missingness",
        ),
    ),
    "S5F5_computed_pos": Sprint5FeatureSetSpec(
        name="S5F5_computed_pos",
        description="S5F4 plus position-resolved computed nucleosome vectors.",
        families=(
            "sequence_one_hot",
            "sequence_summary",
            "mismatch_position",
            "binding_energy",
            "experimental_epigenetic",
            "computed_nucleosome_aggregates",
            "computed_nucleosome_missingness",
            "computed_nucleosome_position_resolved",
        ),
    ),
}


def normalize_sprint5_feature_set(value: str) -> Sprint5FeatureSetName:
    normalized = value.strip().lower()
    if normalized not in SPRINT5_FEATURE_SET_ALIASES:
        raise ValueError(f"Unknown Sprint 5 feature set: {value}")
    return SPRINT5_FEATURE_SET_ALIASES[normalized]


def build_sequence_only_pair_features(df: pd.DataFrame, max_length: int = 23) -> pd.DataFrame:
    required = {"grna_target_sequence", "target_sequence"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required sequence columns: {missing}")
    rows = []
    for guide, target in zip(df["grna_target_sequence"], df["target_sequence"], strict=True):
        rows.append(np.concatenate([_one_hot_sequence(guide, max_length), _one_hot_sequence(target, max_length)]))
    columns = [
        *[f"guide_pos_{position:02d}_{base}" for position in range(max_length) for base in ("A", "C", "G", "T", "N")],
        *[f"target_pos_{position:02d}_{base}" for position in range(max_length) for base in ("A", "C", "G", "T", "N")],
    ]
    return pd.DataFrame(np.vstack(rows), columns=columns, index=df.index)


def build_computed_nucleosome_position_features(df: pd.DataFrame, max_length: int = 23) -> pd.DataFrame:
    missing = sorted(set(COMPUTED_NUCLEOSOME_FEATURES).difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing computed nucleosome columns: {missing}")
    parts = []
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        parsed = df[feature].map(lambda value: _fixed_length_array(value, max_length))
        values = np.vstack(parsed.to_numpy())
        columns = [f"{feature}_pos_{position:02d}" for position in range(max_length)]
        parts.append(pd.DataFrame(values, columns=columns, index=df.index))
    return pd.concat(parts, axis=1)


def build_sprint5_feature_set(
    df: pd.DataFrame,
    feature_set: Sprint5FeatureSetName | str,
    *,
    max_length: int = 23,
) -> pd.DataFrame:
    feature_set = normalize_sprint5_feature_set(str(feature_set))
    parts = [build_sequence_only_pair_features(df, max_length=max_length)]
    if feature_set in {
        "S5F1_mismatch",
        "S5F2_energy",
        "S5F3_experimental_epi",
        "S5F4_computed_agg",
        "S5F5_computed_pos",
    }:
        parts.append(build_sequence_mismatch_features(df, max_positions=max_length))
    if feature_set in {"S5F2_energy", "S5F3_experimental_epi", "S5F4_computed_agg", "S5F5_computed_pos"}:
        parts.append(build_binding_energy_features(df))
    if feature_set in {"S5F3_experimental_epi", "S5F4_computed_agg", "S5F5_computed_pos"}:
        parts.append(build_experimental_epigenetic_features(df))
    if feature_set in {"S5F4_computed_agg", "S5F5_computed_pos"}:
        parts.append(build_computed_nucleosome_features(df))
    if feature_set == "S5F5_computed_pos":
        parts.append(build_computed_nucleosome_position_features(df, max_length=max_length))

    features = pd.concat(parts, axis=1)
    leaked = sorted(set(features.columns).intersection(FORBIDDEN_PREDICTIVE_COLUMNS))
    if leaked:
        raise ValueError(f"Forbidden columns leaked into Sprint 5 features: {leaked}")
    return features


def build_sprint5_feature_tables(
    assigned: pd.DataFrame,
    *,
    max_length: int = 23,
    scale: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, object]]:
    required = {"id", "split"}
    missing = sorted(required.difference(assigned.columns))
    if missing:
        raise ValueError(f"Assigned rows are missing required columns: {missing}")
    ids = assigned["id"].astype(str).reset_index(drop=True)
    train_mask = assigned["split"].eq("train")
    if not train_mask.any():
        raise ValueError("Sprint 5 feature preprocessing requires at least one train row")

    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, list[str]] = {}
    preprocessing: dict[str, object] = {}
    for feature_set in SPRINT5_FEATURE_SET_ORDER:
        raw = build_sprint5_feature_set(assigned, feature_set, max_length=max_length)
        preprocessor = TrainOnlyPreprocessor(scale=scale).fit(raw.loc[train_mask])
        transformed = preprocessor.transform(raw)
        tables[feature_set] = _keyed_features(ids, transformed.reset_index(drop=True))
        sources[feature_set] = transformed.columns.tolist()
        preprocessing[feature_set] = {
            "fit_scope": "train_only",
            "imputation": "median",
            "scaling": "standard" if scale else False,
            "feature_columns": transformed.columns.tolist(),
            "median_values": _imputer_statistics(preprocessor),
        }
    return tables, sources, preprocessing


def _one_hot_sequence(sequence: object, max_length: int) -> np.ndarray:
    bases = ("A", "C", "G", "T", "N")
    lookup = {base: index for index, base in enumerate(bases)}
    text = "" if pd.isna(sequence) else str(sequence).upper()
    output = np.zeros((max_length, len(bases)), dtype=np.float32)
    for position in range(max_length):
        base = text[position] if position < len(text) and text[position] in lookup else "N"
        output[position, lookup[base]] = 1.0
    return output.reshape(-1)


def _fixed_length_array(value: object, max_length: int) -> np.ndarray:
    result = parse_numeric_array_result(value)
    if not result.is_valid or result.values is None:
        return np.full(max_length, np.nan, dtype=float)
    values = np.asarray(result.values, dtype=float)
    if values.shape[0] != max_length:
        raise ValueError(f"Computed nucleosome vector length must be {max_length}; got {values.shape[0]}")
    return values


def _keyed_features(keys: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    renamed = features.copy()
    renamed.columns = [f"feature__{column}" for column in renamed.columns]
    return pd.concat([pd.DataFrame({"record_id": keys.to_numpy()}), renamed.reset_index(drop=True)], axis=1)


def _imputer_statistics(preprocessor: TrainOnlyPreprocessor) -> dict[str, float]:
    imputer = preprocessor.pipeline.named_steps["imputer"]
    columns = preprocessor.feature_columns or []
    return {name: float(value) for name, value in zip(columns, imputer.statistics_, strict=True)}
