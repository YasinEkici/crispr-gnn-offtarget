"""Sprint 9 robustness layer — registry, metric replay, and guide-cluster bootstrap.

Loads the saved per-row predictions of the frozen Sprint 7F/8A/8B runs (plus the
regenerated XGBoost ``F4`` bar) and each model's validation-selected classification
threshold, then:

- replays full-split metrics with the canonical
  :func:`binary_classification_metrics` (Slice 1), anchoring Sprint 9 to the frozen
  numbers;
- computes **guide-cluster bootstrap** confidence intervals (Slice 3): resample
  guides (clusters), not rows; percentile CI primary, BCa as a sensitivity check
  with a leave-one-guide jackknife trust gate (009 plan §4/§14;
  ``docs/literature/sprint9-deep-research.pdf`` §2.1/§3.3).

Discipline (see ``docs/exec-plans/active/009-sprint9-robustness.md`` and the PDF):

- Thresholds are **read** (validation-selected) and **never recomputed from test**;
  thresholded metrics apply the frozen threshold in every bootstrap replicate.
- The resampling unit is the guide cluster :data:`GUIDE_KEY` (``grna_target_id``);
  a guide drawn k times contributes its rows k times. Rows are never resampled.
- AUPRC uses the same ``average_precision_score`` definition as the source reports;
  no trapezoidal PR-AUC switch. ``0.900705`` is the no-skill PR baseline, not a floor.
- Degenerate replicates (no positives → AUPRC; no negatives → specificity; single
  class → AUROC/MCC/macro-F1) yield NaN and are counted, never crash.

This module reads frozen artifacts and computes metrics; it does not modify any
prediction, threshold, label, split, or model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

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

#: XGBoost F4 bar — regenerated in Slice 2 (Sprint 2 saved no per-row predictions).
F4_REGISTRY_ID = "XGB_F4"
_PRED_F4 = ROOT / "outputs/sprint9/diagnostics/f4_predictions.csv"


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


def load_f4_model_scores(predictions_path: Path = _PRED_F4) -> ModelScores:
    """Load the regenerated XGBoost F4 scores and its validation-selected threshold.

    F4 is regenerated under the current pinned XGBoost (2026-06-13 Option C; the
    historical ``0.992522`` was an earlier build — version drift ~1e-4, negligible
    vs the bootstrap CI width). The F4 bar is therefore a single self-consistent
    model: the threshold is the regenerated model's own ``validation_max_f1``
    threshold (computed on regenerated validation scores; never from test), carried
    in the ``threshold`` column of the predictions file. Raises ``FileNotFoundError``
    until ``scripts/regenerate_f4_predictions.py`` has run.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"F4 predictions not found ({predictions_path}); run "
            "scripts/regenerate_f4_predictions.py (Slice 2) first"
        )
    preds = pd.read_csv(predictions_path)
    _require_columns(
        preds,
        ["predeclared_run_id", "run_id", "split", GUIDE_KEY, "label", "score", "threshold"],
        predictions_path,
    )
    sub = preds[preds["predeclared_run_id"] == F4_REGISTRY_ID]
    if sub.empty:
        raise ValueError(f"{F4_REGISTRY_ID}: not found in {predictions_path}")
    run_ids = sorted(sub["run_id"].unique())
    if len(run_ids) != 1:
        raise ValueError(f"{F4_REGISTRY_ID}: expected one run_id, found {run_ids}")
    thresholds = sub["threshold"].unique()
    if len(thresholds) != 1:
        raise ValueError(f"{F4_REGISTRY_ID}: expected one threshold, found {sorted(thresholds)}")

    frame = sub[["split", GUIDE_KEY, "label", "score"]].reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    frame["score"] = frame["score"].astype(float)
    return ModelScores(
        registry_id=F4_REGISTRY_ID,
        predeclared_run_id=F4_REGISTRY_ID,
        run_id=str(run_ids[0]),
        sprint="sprint2",
        threshold=float(thresholds[0]),
        threshold_selection_split="validation",  # regenerated validation_max_f1
        frame=frame,
    )


def load_full_registry(
    entries: tuple[RegistryEntry, ...] = GNN_REGISTRY,
    *,
    include_f4: bool = True,
) -> dict[str, ModelScores]:
    """Load the GNN registry plus (optionally) the regenerated XGBoost F4 bar."""
    registry = load_registry(entries)
    if include_f4:
        registry[F4_REGISTRY_ID] = load_f4_model_scores()
    return registry


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


# --------------------------------------------------------------------------------
# Slice 3 — guide-cluster bootstrap (percentile primary, BCa sensitivity)
# --------------------------------------------------------------------------------

#: Metrics carried through the bootstrap (ranking: auprc/auroc; operating point at
#: the frozen threshold: mcc/specificity/macro_f1).
BOOTSTRAP_METRICS = ("auprc", "auroc", "mcc", "specificity", "macro_f1")
#: Metrics bounded in [0, 1] whose upper edge (1.0) is inspected for pile-up.
_UPPER_BOUNDED_METRICS = frozenset({"auprc", "auroc", "specificity", "macro_f1"})
DEFAULT_N_BOOT = 5000
DEFAULT_BOOTSTRAP_SEED = 12345
#: Test positive prevalence — the no-skill PR baseline (NOT a floor).
NO_SKILL_BASELINE = 0.900705
#: Negative-class-dominant test guide (holds 80/169 = 47.3% of negatives).
DOMINANT_NEGATIVE_GUIDE = 9251


def _split_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Lean NaN-aware metrics for one (resampled) row set; never raises on degeneracy."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    both_classes = np.unique(labels).shape[0] == 2
    predictions = (scores >= threshold).astype(int)
    tn, fp, _fn, _tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auprc": float(average_precision_score(labels, scores)) if labels.sum() > 0 else float("nan"),
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both_classes else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))
        if both_classes
        else float("nan"),
    }


def _guide_groups(frame: pd.DataFrame) -> tuple[list, list[np.ndarray], list[np.ndarray]]:
    """Per-guide (label, score) arrays in stable guide order — the resampling units."""
    guide_ids: list = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for guide_id, group in frame.groupby(GUIDE_KEY, sort=True):
        guide_ids.append(guide_id)
        labels.append(group["label"].to_numpy().astype(int))
        scores.append(group["score"].to_numpy().astype(float))
    return guide_ids, labels, scores


def _bca_interval(
    point: float, samples: np.ndarray, jackknife: np.ndarray, ci: float
) -> tuple[float, float, bool, str]:
    """BCa interval with a leave-one-guide jackknife trust gate (009 plan §14).

    Returns (lo, hi, trusted, note). With only ~29 clusters and a few influential
    negative-bearing guides the acceleration term is fragile, so BCa is reported as
    a sensitivity check only: ``trusted`` is False whenever the bias-correction or
    jackknife acceleration is degenerate, non-finite, or dominated by one guide.
    """
    if samples.size < 2 or np.isnan(point):
        return float("nan"), float("nan"), False, "insufficient_samples"
    proportion = float(np.mean(samples < point))
    if proportion <= 0.0 or proportion >= 1.0:
        return float("nan"), float("nan"), False, "bias_correction_degenerate"
    z0 = float(norm.ppf(proportion))

    finite_jack = jackknife[~np.isnan(jackknife)]
    if finite_jack.size < 3 or finite_jack.size < jackknife.size:
        # A dropped guide left the metric undefined → jackknife unreliable.
        return float("nan"), float("nan"), False, "jackknife_undefined"
    diffs = finite_jack.mean() - finite_jack
    sum_sq = float(np.sum(diffs**2))
    denom = 6.0 * (sum_sq**1.5)
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan"), float("nan"), False, "acceleration_degenerate"
    acceleration = float(np.sum(diffs**3) / denom)
    if not np.isfinite(acceleration):
        return float("nan"), float("nan"), False, "acceleration_nonfinite"

    cubed = np.abs(diffs**3)
    dominated = bool(cubed.max() > 0.5 * cubed.sum()) if cubed.sum() > 0 else True

    z_lo = float(norm.ppf((1.0 - ci) / 2.0))
    z_hi = float(norm.ppf(1.0 - (1.0 - ci) / 2.0))

    def _adjust(z: float) -> float:
        scale = 1.0 - acceleration * (z0 + z)
        if scale == 0.0:
            return float("nan")
        return float(norm.cdf(z0 + (z0 + z) / scale))

    alpha_lo = _adjust(z_lo)
    alpha_hi = _adjust(z_hi)
    if not (np.isfinite(alpha_lo) and np.isfinite(alpha_hi)) or not (0.0 < alpha_lo < 1.0 and 0.0 < alpha_hi < 1.0):
        return float("nan"), float("nan"), False, "alpha_out_of_range"

    lo = float(np.quantile(samples, alpha_lo))
    hi = float(np.quantile(samples, alpha_hi))
    return lo, hi, (not dominated), ("dominated_by_one_guide" if dominated else "ok")


@dataclass
class GuideBootstrapResult:
    """Guide-cluster bootstrap output for one model (009 plan §4/§4.1)."""

    registry_id: str
    n_test_guides: int
    n_boot: int
    ci: float
    seed: int
    threshold: float
    point: dict[str, float]
    percentile: dict[str, tuple[float, float]]
    bca: dict[str, tuple[float, float]]
    bca_trusted: dict[str, bool]
    bca_note: dict[str, str]
    undefined_rate: dict[str, float]
    shape: dict[str, dict[str, object]]
    samples: dict[str, np.ndarray]
    jackknife: dict[str, list[float]]
    guide_ids: list
    mean_unique_guides: float
    mean_negative_bearing_guides: float
    dominant_guide: object
    dominant_guide_inclusion_rate: float
    negatives_per_guide: dict = field(default_factory=dict)


def guide_cluster_bootstrap(
    scores: ModelScores,
    *,
    split: str = "test",
    metrics: tuple[str, ...] = BOOTSTRAP_METRICS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = 0.95,
    dominant_guide: object = DOMINANT_NEGATIVE_GUIDE,
) -> GuideBootstrapResult:
    """Guide-cluster (not row) bootstrap CIs for one model.

    Resamples the ``n_guides`` test guides with replacement (a guide drawn k times
    contributes its rows k times), applies the frozen threshold every replicate, and
    returns percentile + BCa intervals, per-metric undefined rates, distribution
    shape flags, leave-one-guide jackknife values, and guide-composition diagnostics.
    """
    frame = scores.split(split)
    guide_ids, group_labels, group_scores = _guide_groups(frame)
    n_guides = len(guide_ids)
    if n_guides < 2:
        raise ValueError(f"{scores.registry_id}: need >=2 guides to bootstrap, found {n_guides}")

    full_labels = np.concatenate(group_labels)
    full_scores = np.concatenate(group_scores)
    point = _split_metrics(full_labels, full_scores, scores.threshold)

    negatives_per_guide = np.array([int((lab == 0).sum()) for lab in group_labels])
    is_negative_bearing = negatives_per_guide > 0
    dominant_index = guide_ids.index(dominant_guide) if dominant_guide in guide_ids else None

    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {metric: [] for metric in metrics}
    unique_guide_counts: list[int] = []
    negative_bearing_counts: list[int] = []
    dominant_inclusions = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_guides, size=n_guides)
        labels = np.concatenate([group_labels[i] for i in pick])
        values = np.concatenate([group_scores[i] for i in pick])
        replicate = _split_metrics(labels, values, scores.threshold)
        for metric in metrics:
            value = replicate[metric]
            if not np.isnan(value):
                boot[metric].append(value)
        drawn = np.unique(pick)
        unique_guide_counts.append(int(drawn.size))
        negative_bearing_counts.append(int(is_negative_bearing[drawn].sum()))
        if dominant_index is not None and dominant_index in drawn:
            dominant_inclusions += 1

    # Leave-one-guide jackknife (reused for BCa acceleration and LOGO influence).
    jackknife: dict[str, list[float]] = {metric: [] for metric in metrics}
    for i in range(n_guides):
        keep = [j for j in range(n_guides) if j != i]
        labels = np.concatenate([group_labels[j] for j in keep])
        values = np.concatenate([group_scores[j] for j in keep])
        replicate = _split_metrics(labels, values, scores.threshold)
        for metric in metrics:
            jackknife[metric].append(replicate[metric])

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    percentile: dict[str, tuple[float, float]] = {}
    bca: dict[str, tuple[float, float]] = {}
    bca_trusted: dict[str, bool] = {}
    bca_note: dict[str, str] = {}
    undefined_rate: dict[str, float] = {}
    shape: dict[str, dict[str, object]] = {}
    samples_out: dict[str, np.ndarray] = {}
    for metric in metrics:
        samples = np.asarray(boot[metric], dtype=float)
        samples_out[metric] = samples
        undefined_rate[metric] = float((n_boot - samples.size) / n_boot)
        if samples.size:
            percentile[metric] = (float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q)))
        else:
            percentile[metric] = (float("nan"), float("nan"))
        lo, hi, trusted, note = _bca_interval(
            point[metric], samples, np.asarray(jackknife[metric], dtype=float), ci
        )
        bca[metric] = (lo, hi)
        bca_trusted[metric] = trusted
        bca_note[metric] = note
        n_unique = int(np.unique(samples).size) if samples.size else 0
        frac_upper = (
            float(np.mean(np.isclose(samples, 1.0))) if (samples.size and metric in _UPPER_BOUNDED_METRICS) else 0.0
        )
        flags = []
        if samples.size and n_unique < 20:
            flags.append("discrete")
        if frac_upper > 0.05:
            flags.append("upper_bound_pileup")
        shape[metric] = {"n_unique": n_unique, "frac_at_upper_bound": frac_upper, "flag": ";".join(flags) or "ok"}

    return GuideBootstrapResult(
        registry_id=scores.registry_id,
        n_test_guides=n_guides,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
        threshold=scores.threshold,
        point=point,
        percentile=percentile,
        bca=bca,
        bca_trusted=bca_trusted,
        bca_note=bca_note,
        undefined_rate=undefined_rate,
        shape=shape,
        samples=samples_out,
        jackknife=jackknife,
        guide_ids=guide_ids,
        mean_unique_guides=float(np.mean(unique_guide_counts)),
        mean_negative_bearing_guides=float(np.mean(negative_bearing_counts)),
        dominant_guide=dominant_guide if dominant_index is not None else None,
        dominant_guide_inclusion_rate=float(dominant_inclusions / n_boot),
        negatives_per_guide=dict(zip(guide_ids, negatives_per_guide.tolist())),
    )


def leave_one_guide_influence(result: GuideBootstrapResult, metrics: tuple[str, ...] = BOOTSTRAP_METRICS) -> list[dict[str, object]]:
    """Per (model, metric, guide) leave-one-guide jackknife value and its delta from
    the full-sample point estimate (009 plan §4.1; flags negative-class fragility)."""
    records: list[dict[str, object]] = []
    for metric in metrics:
        point = result.point[metric]
        for guide_id, value in zip(result.guide_ids, result.jackknife[metric]):
            records.append(
                {
                    "registry_id": result.registry_id,
                    "metric": metric,
                    "guide": guide_id,
                    "negatives_in_guide": int(result.negatives_per_guide.get(guide_id, 0)),
                    "point_estimate": float(point),
                    "leave_one_guide_value": float(value),
                    "delta": float(value - point) if not np.isnan(value) else float("nan"),
                }
            )
    return records
