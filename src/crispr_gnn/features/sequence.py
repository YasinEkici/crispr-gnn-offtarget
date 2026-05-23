"""Sequence-only inputs for Sprint 2 neural baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crispr_gnn.features.tabular import FORBIDDEN_PREDICTIVE_COLUMNS


SEQUENCE_REPRESENTATION = "S1_sequence_pair"
SEQUENCE_FEATURE_SET = "S1"
SEQUENCE_SOURCE_COLUMNS = ("grna_target_sequence", "target_sequence")
BASES = ("A", "C", "G", "T", "N")
BASE_TO_INDEX = {base: index for index, base in enumerate(BASES)}


@dataclass(frozen=True)
class SequenceEncodingResult:
    encoded: np.ndarray
    row_index: np.ndarray
    audit: pd.DataFrame


def build_sequence_pair_encoding(df: pd.DataFrame, *, max_length: int = 23) -> SequenceEncodingResult:
    missing = sorted(set(SEQUENCE_SOURCE_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required sequence columns: {missing}")
    forbidden = sorted(set(SEQUENCE_SOURCE_COLUMNS).intersection(FORBIDDEN_PREDICTIVE_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden columns configured as sequence inputs: {forbidden}")

    encoded = np.zeros((df.shape[0], max_length, 11), dtype=np.float32)
    for row_position, (_, row) in enumerate(df.iterrows()):
        guide = _clean_sequence(row["grna_target_sequence"])
        target = _clean_sequence(row["target_sequence"])
        for position in range(max_length):
            guide_base = _base_at(guide, position)
            target_base = _base_at(target, position)
            encoded[row_position, position, BASE_TO_INDEX[guide_base]] = 1.0
            encoded[row_position, position, len(BASES) + BASE_TO_INDEX[target_base]] = 1.0
            if guide_base != "N" and target_base != "N" and guide_base != target_base:
                encoded[row_position, position, 10] = 1.0

    return SequenceEncodingResult(
        encoded=encoded,
        row_index=df.index.to_numpy(),
        audit=sequence_input_audit(max_length=max_length),
    )


def sequence_input_audit(*, max_length: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "input_representation": SEQUENCE_REPRESENTATION,
                "feature_set": SEQUENCE_FEATURE_SET,
                "source_column": column,
                "is_forbidden": column in FORBIDDEN_PREDICTIVE_COLUMNS,
                "max_length": max_length,
                "channels": 11,
                "policy": "one-hot guide bases, one-hot target bases, and aligned mismatch channel",
            }
            for column in SEQUENCE_SOURCE_COLUMNS
        ]
    )


def _clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(base if base in BASE_TO_INDEX else "N" for base in str(value).upper())


def _base_at(sequence: str, position: int) -> str:
    if position >= len(sequence):
        return "N"
    return sequence[position]
