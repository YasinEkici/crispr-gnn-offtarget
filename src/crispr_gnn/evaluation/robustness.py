"""Sprint 9 robustness layer — prediction registry and metric replay (Slice 1).

Loads the saved per-row predictions of the frozen Sprint 7F/8A/8B runs and each
model's validation-selected classification threshold, then replays full-split
metrics with the canonical :func:`binary_classification_metrics` so the Sprint 9
uncertainty layer is anchored to the exact frozen numbers. The guide-cluster
bootstrap (Slice 3) and paired-difference bootstrap (Slice 4) build on this
registry; the XGBoost ``F4`` entry is added in Slice 2.

Discipline (see ``docs/exec-plans/active/009-sprint9-robustness.md`` and
``docs/literature/sprint9-deep-research.pdf``):

- Thresholds are **read** from each run's comparison CSV (``threshold`` column,
  validation-selected) and are **never recomputed from test**.
- The guide cluster key is :data:`GUIDE_KEY` (``grna_target_id``). It is loaded here
  but resampled only in Slice 3 — rows are never the resampling unit.
- AUPRC uses the same ``average_precision_score`` definition as the source reports
  (via ``metrics.binary_classification_metrics``); no trapezoidal PR-AUC switch.

This module reads frozen artifacts and computes metrics; it does not modify any
prediction, threshold, label, split, or model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crispr_gnn.evaluation.metrics import binary_classification_metrics

ROOT = Path(__file__).resolve().parents[3]

#: Guide cluster key. Resampled only in Slice 3; never resample rows.
GUIDE_KEY = "grna_target_id"

#: Metrics replayed and checked against the source comparison CSVs.
REPLAY_METRICS = (
    "auprc",
    "auroc",
    "mcc",
    "specificity",
    "macro_f1",
    "f1",
    "sensitivity",
    "tn",
    "fp",
    "fn",
    "tp",
)

#: Confusion-cell counts are integers and must match exactly (tolerance 0).
INTEGER_METRICS = frozenset({"tn", "fp", "fn", "tp"})

#: Default replay tolerance for float metrics (observed diffs are ~1e-15).
DEFAULT_REPLAY_ATOL = 1e-9

_PRED_7F = ROOT / "outputs/sprint7f/diagnostics/target_context_encoder_predictions.csv"
_COMP_7F = ROOT / "outputs/sprint7f/target_context_encoder_comparison.csv"
_PRED_8A = ROOT / "outputs/sprint8a/diagnostics/target_context_interaction_predictions.csv"
_COMP_8A = ROOT / "outputs/sprint8a/target_context_interaction_comparison.csv"
_PRED_8B = ROOT / "outputs/sprint8b/diagnostics/sequence_context_predictions.csv"
_COMP_8B = ROOT / "outputs/sprint8b/sequence_context_comparison.csv"


@dataclass(frozen=True)
class RegistryEntry:
    """One predeclared Sprint 9 model and where its frozen artifacts live."""

    registry_id: str
    predeclared_run_id: str
    sprint: str
    predictions_path: Path
    comparison_path: Path


#: Predeclared Sprint 9 registry (009 plan §3.1). ``XGB_F4`` is added in Slice 2
#: (Sprint 2 saved no per-row XGBoost predictions).
GNN_REGISTRY: tuple[RegistryEntry, ...] = (
    RegistryEntry("S7F_R1", "S7F_R1_unified_deep_context_encoder", "sprint7f", _PRED_7F, _COMP_7F),
    RegistryEntry("S7F_R2", "S7F_R2_family_aware_context_encoder", "sprint7f", _PRED_7F, _COMP_7F),
    RegistryEntry("S7F_R3", "S7F_R3_family_aware_experimental_emphasis", "sprint7f", _PRED_7F, _COMP_7F),
    RegistryEntry("S8A_R0", "S8A_R0_base_reference", "sprint8a", _PRED_8A, _COMP_8A),
    RegistryEntry("S8A_R1", "S8A_R1_family_gated_v2", "sprint8a", _PRED_8A, _COMP_8A),
    RegistryEntry("S8A_R2", "S8A_R2_context_edge_film", "sprint8a", _PRED_8A, _COMP_8A),
    RegistryEntry("S8A_R3", "S8A_R3_gated_plus_film", "sprint8a", _PRED_8A, _COMP_8A),
    RegistryEntry("S8A_R4", "S8A_R4_regularized_exp_branch", "sprint8a", _PRED_8A, _COMP_8A),
    RegistryEntry("S8B_R1", "S8B_R1_sequence_only", "sprint8b", _PRED_8B, _COMP_8B),
    RegistryEntry("S8B_R2", "S8B_R2_sequence_plus_context", "sprint8b", _PRED_8B, _COMP_8B),
)


@dataclass(frozen=True)
class ModelScores:
    """Loaded per-row predictions plus the frozen validation-selected threshold."""

    registry_id: str
    predeclared_run_id: str
    run_id: str
    sprint: str
    threshold: float
    threshold_selection_split: str
    frame: pd.DataFrame  # columns: split, grna_target_id, label, score

    def split(self, split: str = "test") -> pd.DataFrame:
        sub = self.frame[self.frame["split"] == split]
        if sub.empty:
            raise ValueError(f"{self.registry_id}: no '{split}' rows in predictions")
        return sub


def _require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")


def load_model_scores(entry: RegistryEntry) -> ModelScores:
    """Load one registry model's per-row scores and its read-only frozen threshold."""
    preds = pd.read_csv(entry.predictions_path)
    _require_columns(
        preds, ["predeclared_run_id", "run_id", "split", GUIDE_KEY, "label", "score"], entry.predictions_path
    )
    sub = preds[preds["predeclared_run_id"] == entry.predeclared_run_id]
    if sub.empty:
        raise ValueError(
            f"{entry.registry_id}: '{entry.predeclared_run_id}' not in {entry.predictions_path}"
        )
    run_ids = sorted(sub["run_id"].unique())
    if len(run_ids) != 1:
        raise ValueError(
            f"{entry.registry_id}: expected one authoritative batch run_id, found {run_ids}"
        )

    comp = pd.read_csv(entry.comparison_path)
    crow = comp[comp["predeclared_run_id"] == entry.predeclared_run_id]
    if len(crow) != 1:
        raise ValueError(
            f"{entry.registry_id}: expected one comparison row for "
            f"'{entry.predeclared_run_id}', found {len(crow)}"
        )
    crow = crow.iloc[0]
    threshold = crow["threshold"]
    if pd.isna(threshold):
        raise ValueError(
            f"{entry.registry_id}: comparison threshold is NaN — not a trained run with a "
            "validation-selected threshold"
        )
    threshold_selection_split = str(crow.get("threshold_selection_split", ""))

    frame = sub[["split", GUIDE_KEY, "label", "score"]].reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    frame["score"] = frame["score"].astype(float)
    return ModelScores(
        registry_id=entry.registry_id,
        predeclared_run_id=entry.predeclared_run_id,
        run_id=str(run_ids[0]),
        sprint=entry.sprint,
        threshold=float(threshold),
        threshold_selection_split=threshold_selection_split,
        frame=frame,
    )


def load_registry(entries: tuple[RegistryEntry, ...] = GNN_REGISTRY) -> dict[str, ModelScores]:
    """Load every registry entry, keyed by ``registry_id``."""
    return {entry.registry_id: load_model_scores(entry) for entry in entries}


def replay_split_metrics(scores: ModelScores, split: str = "test") -> dict[str, float | int]:
    """Recompute full-split metrics with the frozen (read) threshold."""
    sub = scores.split(split)
    return binary_classification_metrics(
        sub["label"].to_numpy(),
        sub["score"].to_numpy(),
        scores.threshold,
        prefix=f"{split}_",
    )


def _source_metrics(entry: RegistryEntry, split: str = "test") -> dict[str, float]:
    comp = pd.read_csv(entry.comparison_path)
    crow = comp[comp["predeclared_run_id"] == entry.predeclared_run_id].iloc[0]
    out: dict[str, float] = {}
    for metric in REPLAY_METRICS:
        column = f"{split}_{metric}"
        out[metric] = float(crow[column]) if column in crow.index and not pd.isna(crow[column]) else np.nan
    return out


def replay_check_records(
    entries: tuple[RegistryEntry, ...] = GNN_REGISTRY,
    split: str = "test",
    atol: float = DEFAULT_REPLAY_ATOL,
) -> list[dict[str, object]]:
    """Replay each model's metrics and compare against the source comparison CSV.

    Returns one record per (model, metric) with the source value, the replayed
    value, the absolute difference, the applied tolerance, and a ``within_tol`` flag.
    Confusion-cell counts are checked exactly; float metrics use ``atol``.
    """
    records: list[dict[str, object]] = []
    for entry in entries:
        scores = load_model_scores(entry)
        replay = replay_split_metrics(scores, split)
        source = _source_metrics(entry, split)
        for metric in REPLAY_METRICS:
            source_value = source[metric]
            if np.isnan(source_value):
                continue
            replay_value = float(replay[f"{split}_{metric}"])
            diff = abs(replay_value - source_value)
            tolerance = 0.0 if metric in INTEGER_METRICS else atol
            records.append(
                {
                    "registry_id": entry.registry_id,
                    "predeclared_run_id": entry.predeclared_run_id,
                    "run_id": scores.run_id,
                    "sprint": entry.sprint,
                    "split": split,
                    "threshold": scores.threshold,
                    "threshold_selection_split": scores.threshold_selection_split,
                    "metric": metric,
                    "source_value": source_value,
                    "replay_value": replay_value,
                    "abs_diff": diff,
                    "atol": tolerance,
                    "within_tol": bool(diff <= tolerance),
                }
            )
    return records
