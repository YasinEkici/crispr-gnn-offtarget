"""Guide-level split helpers for supervised CRISPR off-target baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crispr_gnn.data.labels import SCHEME_A_THRESHOLD
from crispr_gnn.data.schemas import GENOME_CANDIDATE_FIELDS, GUIDE_KEY


SPLIT_ID = "sprint2_main_seed42"
SPLIT_COLUMN = "split"
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class GuideSplitConfig:
    split_id: str = SPLIT_ID
    seed: int = 42
    guide_column: str = GUIDE_KEY
    label_threshold: float = SCHEME_A_THRESHOLD
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    exclude_experiment_id: int | None = 18
    search_iterations: int = 50_000


@dataclass(frozen=True)
class GuideSplit:
    config: GuideSplitConfig
    guides: dict[str, list[str]]
    score: float

    def to_manifest(self, summary: pd.DataFrame) -> dict[str, Any]:
        return {
            "split_id": self.config.split_id,
            "label_scheme": "scheme_a",
            "label_definition": f"cleavage_freq > {self.config.label_threshold:g}",
            "guide_column": self.config.guide_column,
            "main_universe": {
                "label_eligible": True,
                "measured": 1,
                "excluded_experiment_id": self.config.exclude_experiment_id,
            },
            "random_seed": self.config.seed,
            "fractions": {
                "train": self.config.train_fraction,
                "val": self.config.val_fraction,
                "test": self.config.test_fraction,
            },
            "score": self.score,
            "guides": self.guides,
            "summary": summary.to_dict(orient="records"),
        }


def validate_split_fractions(config: GuideSplitConfig) -> None:
    total = config.train_fraction + config.val_fraction + config.test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}")
    if min(config.train_fraction, config.val_fraction, config.test_fraction) <= 0:
        raise ValueError("Split fractions must all be positive")


def add_scheme_a_labels(df: pd.DataFrame, threshold: float = SCHEME_A_THRESHOLD) -> pd.DataFrame:
    if "cleavage_freq" not in df.columns:
        raise ValueError("Dataframe is missing required column: cleavage_freq")
    labeled = df.copy()
    cleavage = pd.to_numeric(labeled["cleavage_freq"], errors="coerce")
    labeled = labeled.loc[cleavage.notna()].copy()
    labeled[LABEL_COLUMN] = (cleavage.loc[labeled.index] > threshold).astype(int)
    return labeled


def main_clean_frame(df: pd.DataFrame, config: GuideSplitConfig) -> pd.DataFrame:
    required = {config.guide_column, "measured", "experiment_id", "cleavage_freq"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required split columns: {missing}")

    labeled = add_scheme_a_labels(df, threshold=config.label_threshold)
    if config.exclude_experiment_id is not None:
        labeled = labeled.loc[labeled["experiment_id"] != config.exclude_experiment_id].copy()
    return labeled


def split_basis_frame(df: pd.DataFrame, config: GuideSplitConfig) -> pd.DataFrame:
    clean = main_clean_frame(df, config)
    measured = clean.loc[clean["measured"] == 1].copy()
    if measured.empty:
        raise ValueError("No measured=1 rows available for guide-level splitting")
    return measured


def _guide_summary(split_basis: pd.DataFrame, guide_column: str) -> pd.DataFrame:
    summary = (
        split_basis.groupby(guide_column, dropna=False)
        .agg(rows=(LABEL_COLUMN, "size"), positives=(LABEL_COLUMN, "sum"))
        .reset_index()
    )
    summary["negatives"] = summary["rows"] - summary["positives"]
    return summary


def _score_candidate(candidate: pd.DataFrame, targets: dict[str, float], fractions: dict[str, float]) -> float:
    score = 0.0
    for split_name, fraction in fractions.items():
        part = candidate.loc[candidate[SPLIT_COLUMN] == split_name]
        if part.empty:
            return float("inf")
        rows = float(part["rows"].sum())
        positives = float(part["positives"].sum())
        negatives = float(part["negatives"].sum())
        if positives <= 0 or negatives <= 0:
            return float("inf")
        max_guide_share = float(part["rows"].max() / rows)
        score += abs(rows - targets["rows"] * fraction) / targets["rows"]
        score += 0.75 * abs(positives - targets["positives"] * fraction) / max(targets["positives"], 1.0)
        score += 0.75 * abs(negatives - targets["negatives"] * fraction) / max(targets["negatives"], 1.0)
        score += 0.10 * max_guide_share
    return score


def build_guide_split(df: pd.DataFrame, config: GuideSplitConfig | None = None) -> tuple[GuideSplit, pd.DataFrame]:
    config = config or GuideSplitConfig()
    validate_split_fractions(config)

    split_basis = split_basis_frame(df, config)
    guide_summary = _guide_summary(split_basis, config.guide_column)
    if guide_summary.shape[0] < 3:
        raise ValueError("At least three guides are required for train/val/test guide-level splitting")

    fractions = {"train": config.train_fraction, "val": config.val_fraction, "test": config.test_fraction}
    targets = {
        "rows": float(guide_summary["rows"].sum()),
        "positives": float(guide_summary["positives"].sum()),
        "negatives": float(guide_summary["negatives"].sum()),
    }
    rng = np.random.default_rng(config.seed)
    split_names = np.array(["train", "val", "test"])
    probabilities = np.array([config.train_fraction, config.val_fraction, config.test_fraction])

    best: pd.DataFrame | None = None
    best_score = float("inf")
    for _ in range(config.search_iterations):
        candidate = guide_summary.copy()
        candidate[SPLIT_COLUMN] = rng.choice(split_names, size=len(candidate), p=probabilities)
        score = _score_candidate(candidate, targets, fractions)
        if score < best_score:
            best = candidate
            best_score = score

    if best is None or not np.isfinite(best_score):
        raise ValueError("Could not find a valid guide split with nonzero positives and negatives in every split")

    guides = {
        split_name: sorted(best.loc[best[SPLIT_COLUMN] == split_name, config.guide_column].astype(str).tolist())
        for split_name in ["train", "val", "test"]
    }
    split = GuideSplit(config=config, guides=guides, score=float(best_score))
    summary = summarize_split(split_basis, split)
    validate_guide_split(split_basis, split)
    return split, summary


def assign_measured_splits(df: pd.DataFrame, split: GuideSplit) -> pd.DataFrame:
    basis = split_basis_frame(df, split.config)
    guide_to_split = {guide: name for name, guides in split.guides.items() for guide in guides}
    assigned = basis.copy()
    assigned[SPLIT_COLUMN] = assigned[split.config.guide_column].astype(str).map(guide_to_split)
    assigned = assigned.loc[assigned[SPLIT_COLUMN].notna()].copy()
    return assigned


def summarize_split(df: pd.DataFrame, split: GuideSplit) -> pd.DataFrame:
    assigned = df.copy()
    guide_to_split = {guide: name for name, guides in split.guides.items() for guide in guides}
    assigned[SPLIT_COLUMN] = assigned[split.config.guide_column].astype(str).map(guide_to_split)
    assigned = assigned.loc[assigned[SPLIT_COLUMN].notna()].copy()

    genome_column = next((name for name in GENOME_CANDIDATE_FIELDS if name in assigned.columns), None)
    rows: list[dict[str, Any]] = []
    for split_name in ["train", "val", "test"]:
        part = assigned.loc[assigned[SPLIT_COLUMN] == split_name]
        guide_counts = part.groupby(split.config.guide_column).size()
        row = {
            "split": split_name,
            "rows": int(part.shape[0]),
            "guides": int(part[split.config.guide_column].nunique()),
            "positives": int(part[LABEL_COLUMN].sum()),
            "negatives": int(part.shape[0] - part[LABEL_COLUMN].sum()),
            "positive_rate": float(part[LABEL_COLUMN].mean()) if not part.empty else 0.0,
            "measured_1_rows": int((part["measured"] == 1).sum()),
            "measured_0_rows": int((part["measured"] == 0).sum()),
            "experiment_18_rows": int((part["experiment_id"] == 18).sum()),
            "largest_guide_rows": int(guide_counts.max()) if not guide_counts.empty else 0,
            "largest_guide_share": float(guide_counts.max() / part.shape[0]) if not guide_counts.empty else 0.0,
        }
        if genome_column is not None:
            row["genome_counts"] = json.dumps(part[genome_column].value_counts(dropna=False).to_dict(), sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_guide_split(df: pd.DataFrame, split: GuideSplit) -> None:
    guide_sets = {name: set(guides) for name, guides in split.guides.items()}
    if guide_sets["train"] & guide_sets["val"] or guide_sets["train"] & guide_sets["test"] or guide_sets["val"] & guide_sets["test"]:
        raise ValueError("Guide-level split contains overlapping guide IDs")

    assigned = assign_measured_splits(df, split)
    for split_name in ["train", "val", "test"]:
        part = assigned.loc[assigned[SPLIT_COLUMN] == split_name]
        if part.empty:
            raise ValueError(f"Split '{split_name}' has no rows")
        if (part["measured"] != 1).any():
            raise ValueError(f"Split '{split_name}' contains non-measured rows")
        if (part["experiment_id"] == 18).any():
            raise ValueError(f"Split '{split_name}' contains experiment_id=18 rows")
        positives = int(part[LABEL_COLUMN].sum())
        negatives = int(part.shape[0] - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(f"Split '{split_name}' must contain both positive and negative labels")


def write_split_artifacts(split: GuideSplit, summary: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "sprint2_guides.json"
    summary_path = output_path / "sprint2_split_summary.csv"

    manifest = split.to_manifest(summary)
    manifest["config"] = asdict(split.config)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.to_csv(summary_path, index=False)
    return manifest_path, summary_path
