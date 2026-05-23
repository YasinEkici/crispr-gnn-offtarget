"""Tabular Sprint 2 feature-set builders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crispr_gnn.data.parsers import parse_numeric_array_result
from crispr_gnn.data.schemas import BINDING_ENERGY_FEATURES, COMPUTED_NUCLEOSOME_FEATURES, EXPERIMENTAL_EPIGENETIC_FEATURES


FeatureSetName = Literal["F1", "F2", "F3", "F4"]
FEATURE_SET_ORDER: tuple[FeatureSetName, ...] = ("F1", "F2", "F3", "F4")
RAW_ID_COLUMNS = {
    "id",
    "experiment_id",
    "target_chr",
    "target_start",
    "target_end",
    "target_strand",
    "target_geneid",
    "grna_target_id",
    "grna_target_chr",
    "grna_target_start",
    "grna_target_end",
    "grna_target_strand",
    "epigenetics_ids",
    "cell_line",
    "genome",
}


@dataclass(frozen=True)
class FeatureSetSpec:
    name: FeatureSetName
    description: str
    families: tuple[str, ...]


FEATURE_SET_SPECS = {
    "F1": FeatureSetSpec(
        name="F1",
        description="Sequence and mismatch engineered numeric features.",
        families=("sequence_summary", "mismatch_position"),
    ),
    "F2": FeatureSetSpec(
        name="F2",
        description="F1 plus binding-energy scalar features.",
        families=("sequence_summary", "mismatch_position", "binding_energy"),
    ),
    "F3": FeatureSetSpec(
        name="F3",
        description="F2 plus experimental epigenetic scalar features.",
        families=("sequence_summary", "mismatch_position", "binding_energy", "experimental_epigenetic"),
    ),
    "F4": FeatureSetSpec(
        name="F4",
        description="F3 plus aggregated computed nucleosome features and missingness indicators.",
        families=(
            "sequence_summary",
            "mismatch_position",
            "binding_energy",
            "experimental_epigenetic",
            "computed_nucleosome_aggregates",
            "computed_nucleosome_missingness",
        ),
    ),
}


class TrainOnlyPreprocessor:
    """Fit numeric imputation/scaling on train rows only, then transform all splits."""

    def __init__(self, scale: bool = True) -> None:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        self.pipeline = Pipeline(steps)
        self.feature_columns: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "TrainOnlyPreprocessor":
        self.feature_columns = list(features.columns)
        self.pipeline.fit(features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fit before transform")
        missing = sorted(set(self.feature_columns).difference(features.columns))
        extra = sorted(set(features.columns).difference(self.feature_columns))
        if missing or extra:
            raise ValueError(f"Feature columns differ from fit columns; missing={missing}, extra={extra}")
        transformed = self.pipeline.transform(features[self.feature_columns])
        return pd.DataFrame(transformed, columns=self.feature_columns, index=features.index)


def gc_fraction(sequence: object) -> float:
    text = "" if pd.isna(sequence) else str(sequence).upper()
    bases = [base for base in text if base in {"A", "C", "G", "T"}]
    if not bases:
        return np.nan
    return float(sum(base in {"G", "C"} for base in bases) / len(bases))


def _sequence_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper()


def build_sequence_mismatch_features(df: pd.DataFrame, max_positions: int = 23) -> pd.DataFrame:
    required = {"grna_target_sequence", "target_sequence"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required sequence columns: {missing}")

    guides = df["grna_target_sequence"].map(_sequence_text)
    targets = df["target_sequence"].map(_sequence_text)
    out = pd.DataFrame(index=df.index)
    out["guide_length"] = guides.str.len()
    out["target_length"] = targets.str.len()
    out["length_delta"] = out["target_length"] - out["guide_length"]
    out["guide_gc_fraction"] = guides.map(gc_fraction)
    out["target_gc_fraction"] = targets.map(gc_fraction)
    out["target_pam_gc_fraction"] = targets.str[-3:].map(gc_fraction)
    out["target_protospacer_gc_fraction"] = targets.str[:20].map(gc_fraction)

    mismatch_counts: list[int] = []
    aligned_lengths: list[int] = []
    mismatch_positions = {f"mismatch_pos_{position:02d}": [] for position in range(max_positions)}
    for guide, target in zip(guides, targets, strict=True):
        aligned = min(len(guide), len(target), max_positions)
        mismatches = 0
        for position in range(max_positions):
            is_mismatch = position < aligned and guide[position] != target[position]
            mismatch_positions[f"mismatch_pos_{position:02d}"].append(int(is_mismatch))
            mismatches += int(is_mismatch)
        mismatch_counts.append(mismatches)
        aligned_lengths.append(aligned)

    out["aligned_length"] = aligned_lengths
    out["mismatch_count"] = mismatch_counts
    out["mismatch_rate"] = out["mismatch_count"] / out["aligned_length"].replace(0, np.nan)
    for name, values in mismatch_positions.items():
        out[name] = values
    return out


def build_binding_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    return _require_numeric_columns(df, BINDING_ENERGY_FEATURES)


def build_experimental_epigenetic_features(df: pd.DataFrame) -> pd.DataFrame:
    return _require_numeric_columns(df, EXPERIMENTAL_EPIGENETIC_FEATURES)


def _computed_aggregate_for_value(value: object) -> dict[str, float]:
    result = parse_numeric_array_result(value)
    if not result.is_valid or result.values is None:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "center": np.nan,
            "pam_proximal_mean": np.nan,
            "missing": 1.0,
        }
    arr = np.asarray(result.values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "center": float(arr[len(arr) // 2]),
        "pam_proximal_mean": float(arr[-3:].mean()),
        "missing": 0.0,
    }


def build_computed_nucleosome_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(COMPUTED_NUCLEOSOME_FEATURES).difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing computed nucleosome columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        parsed = df[feature].map(_computed_aggregate_for_value)
        aggregates = pd.DataFrame(parsed.tolist(), index=df.index)
        for aggregate_name in ["mean", "std", "min", "max", "center", "pam_proximal_mean"]:
            out[f"{feature}_{aggregate_name}"] = aggregates[aggregate_name]
        out[f"{feature}_missing"] = aggregates["missing"]
    return out


def build_feature_set(df: pd.DataFrame, feature_set: FeatureSetName) -> pd.DataFrame:
    if feature_set not in FEATURE_SET_SPECS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    parts = [build_sequence_mismatch_features(df)]
    if feature_set in {"F2", "F3", "F4"}:
        parts.append(build_binding_energy_features(df))
    if feature_set in {"F3", "F4"}:
        parts.append(build_experimental_epigenetic_features(df))
    if feature_set == "F4":
        parts.append(build_computed_nucleosome_features(df))

    features = pd.concat(parts, axis=1)
    leaked = sorted(set(features.columns).intersection(RAW_ID_COLUMNS))
    if leaked:
        raise ValueError(f"Raw identifier columns leaked into features: {leaked}")
    return features


def feature_catalog_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_set in FEATURE_SET_ORDER:
        spec = FEATURE_SET_SPECS[feature_set]
        rows.append(
            {
                "feature_set": feature_set,
                "description": spec.description,
                "families": ", ".join(spec.families),
            }
        )
    return rows


def summarize_feature_sets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set in FEATURE_SET_ORDER:
        features = build_feature_set(df, feature_set)
        rows.append(
            {
                "feature_set": feature_set,
                "columns": int(features.shape[1]),
                "rows": int(features.shape[0]),
                "rows_with_missing": int(features.isna().any(axis=1).sum()),
                "columns_with_missing": int(features.isna().any(axis=0).sum()),
                "total_missing_values": int(features.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def write_feature_catalog(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    catalog_path = output_path / "sprint2_feature_catalog.md"
    summary_path = output_path / "sprint2_feature_summary.csv"

    summary = summarize_feature_sets(df)
    lines = [
        "# Sprint 2 Feature Catalog",
        "",
        "Feature matrices are generated from the locked Sprint 2 split rows.",
        "Raw identifiers, genome labels, cell-line labels, and coordinate fields are not predictive features.",
        "F4 computed nucleosome aggregates use missing values plus explicit missingness indicators; imputation is fit on train rows during model training.",
        "",
        "## Feature Sets",
        "",
    ]
    for row in feature_catalog_rows():
        lines.extend(
            [
                f"### {row['feature_set']}",
                "",
                str(row["description"]),
                "",
                f"Families: {row['families']}",
                "",
            ]
        )
    lines.extend(["## Summary", "", _markdown_table(summary), ""])
    catalog_path.write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    return catalog_path, summary_path


def _require_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required feature columns: {missing}")
    return df[columns].apply(pd.to_numeric, errors="coerce")


def feature_summary_json(df: pd.DataFrame) -> str:
    return json.dumps(summarize_feature_sets(df).to_dict(orient="records"), indent=2)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
