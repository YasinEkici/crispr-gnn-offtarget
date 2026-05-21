"""Parsers for raw Mak 2022 dataset fields."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ArrayParseStatus = Literal["valid", "missing", "malformed_length", "non_numeric"]


@dataclass(frozen=True)
class ArrayParseResult:
    status: ArrayParseStatus
    values: tuple[float, ...] | None = None
    message: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


def is_missing_value(value: object) -> bool:
    """Return True for scalar missing values used by pandas/pyarrow."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "<na>", "null"}:
        return True
    return False


def parse_numeric_array(value: object, expected_length: int = 23) -> tuple[float, ...] | None:
    """Parse a numeric array field, returning None for missing values.

    The Mak 2022 computed nucleosome features are stored as string-formatted
    arrays such as ``"[0.1 0.2 ...]"`` with whitespace and occasional newlines.
    Comma-separated values are also accepted for tests and future robustness.
    """
    result = parse_numeric_array_result(value, expected_length=expected_length)
    if result.status == "missing":
        return None
    if result.values is None:
        raise ValueError(result.message)
    return result.values


def parse_numeric_array_result(value: object, expected_length: int = 23) -> ArrayParseResult:
    """Parse a numeric array field and classify missing/malformed cases."""
    if is_missing_value(value):
        return ArrayParseResult(status="missing", message="value is missing")

    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.replace(",", " ")
    tokens = text.split()

    if len(tokens) != expected_length:
        return ArrayParseResult(
            status="malformed_length",
            message=f"expected {expected_length} numeric values, found {len(tokens)}",
        )

    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as exc:
        return ArrayParseResult(status="non_numeric", message=str(exc))

    if any(math.isnan(value) for value in values):
        return ArrayParseResult(status="non_numeric", message="array contains NaN")

    return ArrayParseResult(status="valid", values=values)
