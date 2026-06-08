from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TEST_ROWS = 1702
EXPECTED_TEST_POSITIVES = 1533
EXPECTED_TEST_NEGATIVES = 169
EXPECTED_TEST_POSITIVE_RATE = 0.900705


@dataclass(frozen=True)
class PredictionSource:
    analysis_model_id: str
    predeclared_run_id: str
    source_label: str
    path: Path
    row_filter_column: str | None = None
    row_filter_value: str | None = None


PREDICTION_SOURCES = (
    PredictionSource(
        "graph_a_gcn_s6_weighted_bce",
        "S7C_REF_GRAPH_A_GCN",
        "Sprint 6 Graph A GCN weighted-BCE reference",
        Path("outputs/sprint6/loss_comparison/diagnostics_sprint6/sprint6_loss_comparison_predictions.csv"),
        "predeclared_run_id",
        "S6R0_wbce",
    ),
    PredictionSource(
        "graph_a_gatv2_s7",
        "S7C_REF_GRAPH_A_GATV2",
        "Sprint 7 Graph A GATv2 reference",
        Path("outputs/sprint7/diagnostics/gat_predictions.csv"),
        "predeclared_run_id",
        "S7R2_gatv2_edge_aware",
    ),
    PredictionSource(
        "graph_c_gcn_s5b",
        "S7C_REF_GRAPH_C_GCN",
        "Sprint 5B Graph C GCN S5F2 reference",
        Path("outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_predictions.csv"),
    ),
    PredictionSource(
        "graph_b_gcn_s7b",
        "S7C_REF_GRAPH_B_GCN",
        "Sprint 7B Graph B GCN S5F2 reference",
        Path("outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv"),
        "predeclared_run_id",
        "S7B_R1_graph_b_gcn_s5f2",
    ),
    PredictionSource(
        "graph_b_gatv2_s7b",
        "S7C_REF_GRAPH_B_GATV2",
        "Sprint 7B Graph B GATv2 S5F2 reference",
        Path("outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv"),
        "predeclared_run_id",
        "S7B_R2_graph_b_gatv2_s5f2",
    ),
    PredictionSource(
        "graph_c_gatv2_s7b",
        "S7C_GRAPH_C_GATV2",
        "Sprint 7B Graph C GATv2 S5F2 result",
        Path("outputs/sprint7b/diagnostics/gatv2_topology_predictions.csv"),
        "predeclared_run_id",
        "S7B_R3_graph_c_gatv2_s5f2",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the Sprint 7B Graph C GATv2 explanation evidence.")
    parser.add_argument("--output-dir", default="outputs/sprint7c")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = run_sprint7c_graphc_explanation(
        output_dir=ROOT / args.output_dir,
        write_figures=not args.skip_figures,
    )
    print(f"Sprint 7C output directory: {_relative(output_dir)}")
    print(f"Report: {_relative(output_dir / 'sprint7c_graphc_gatv2_explanation_report.md')}")
    return 0


def run_sprint7c_graphc_explanation(*, output_dir: Path, write_figures: bool = True) -> Path:
    diagnostics_dir = output_dir / "diagnostics"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_prediction_sources(ROOT)
    thresholds = _load_thresholds(ROOT)
    attention = _load_optional_csv(ROOT / "outputs/sprint7b/diagnostics/gatv2_topology_attention_summary.csv")

    alignment = _alignment_audit(predictions)
    alignment_path = diagnostics_dir / "sprint7c_prediction_alignment_audit.csv"
    alignment.to_csv(alignment_path, index=False)

    identity_report_path = output_dir / "sprint7c_identity_alignment_audit.md"
    identity_report_path.write_text(_identity_report(alignment), encoding="utf-8")

    metric_recomputation = _metric_recomputation(predictions, thresholds)
    metric_path = diagnostics_dir / "sprint7c_metric_recomputation.csv"
    metric_recomputation.to_csv(metric_path, index=False)

    threshold_transfer = _threshold_transfer(metric_recomputation)
    threshold_transfer_path = diagnostics_dir / "sprint7c_threshold_transfer.csv"
    threshold_transfer.to_csv(threshold_transfer_path, index=False)

    score_distribution = _score_distribution_by_label(predictions)
    score_distribution_path = diagnostics_dir / "sprint7c_score_distribution_by_label.csv"
    score_distribution.to_csv(score_distribution_path, index=False)

    negative_rank = _negative_rank_summary(predictions)
    negative_rank_path = diagnostics_dir / "sprint7c_negative_rank_summary.csv"
    negative_rank.to_csv(negative_rank_path, index=False)

    graphc_transition_allowed = _comparison_passed(
        alignment,
        "graph_c_gcn_s5b",
        "graph_c_gatv2_s7b",
        split="test",
    )
    if graphc_transition_allowed:
        transitions = _error_transitions(
            predictions,
            thresholds,
            baseline_model_id="graph_c_gcn_s5b",
            candidate_model_id="graph_c_gatv2_s7b",
            split="test",
        )
        per_guide_gain = _per_guide_error_gain(transitions)
    else:
        transitions = _blocked_transition_table("graph_c_gcn_s5b", "graph_c_gatv2_s7b")
        per_guide_gain = pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "reason": "Graph C GCN to Graph C GATv2 identity/alignment audit did not pass.",
                }
            ]
        )
    transition_path = diagnostics_dir / "sprint7c_error_transitions.csv"
    transitions.to_csv(transition_path, index=False)
    per_guide_path = diagnostics_dir / "sprint7c_per_guide_error_gain.csv"
    per_guide_gain.to_csv(per_guide_path, index=False)

    attention_summary = _attention_edge_kind_summary(attention)
    attention_path = diagnostics_dir / "sprint7c_attention_edge_kind_summary.csv"
    attention_summary.to_csv(attention_path, index=False)

    figure_paths: list[Path] = []
    if write_figures:
        figure_paths = _write_figures(
            predictions=predictions,
            transitions=transitions,
            per_guide_gain=per_guide_gain,
            attention_summary=attention_summary,
            figures_dir=figures_dir,
        )

    report_path = output_dir / "sprint7c_graphc_gatv2_explanation_report.md"
    report_path.write_text(
        _analysis_report(
            alignment=alignment,
            metric_recomputation=metric_recomputation,
            threshold_transfer=threshold_transfer,
            score_distribution=score_distribution,
            negative_rank=negative_rank,
            transitions=transitions,
            per_guide_gain=per_guide_gain,
            attention_summary=attention_summary,
            figure_paths=figure_paths,
        ),
        encoding="utf-8",
    )
    _write_manifest(
        output_dir / "sprint7c_analysis_manifest.json",
        output_dir=output_dir,
        diagnostics=[
            alignment_path,
            metric_path,
            threshold_transfer_path,
            score_distribution_path,
            negative_rank_path,
            transition_path,
            per_guide_path,
            attention_path,
        ],
        figures=figure_paths,
        reports=[identity_report_path, report_path],
    )
    return output_dir


def _load_prediction_sources(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in PREDICTION_SOURCES:
        path = root / source.path
        if not path.exists():
            raise FileNotFoundError(f"Required prediction artifact not found: {path}")
        frame = pd.read_csv(path)
        if source.row_filter_column is not None:
            if source.row_filter_column not in frame.columns:
                raise ValueError(f"{path} lacks required column {source.row_filter_column}")
            frame = frame.loc[frame[source.row_filter_column].astype(str) == str(source.row_filter_value)].copy()
        if frame.empty:
            raise ValueError(f"No prediction rows loaded for {source.analysis_model_id} from {path}")
        frame = frame.copy()
        frame["analysis_model_id"] = source.analysis_model_id
        frame["analysis_predeclared_run_id"] = source.predeclared_run_id
        frame["analysis_source_label"] = source.source_label
        frame["analysis_source_path"] = source.path.as_posix()
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    required = {"analysis_model_id", "split", "row_index", "grna_target_id", "label", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction artifacts are missing required columns: {sorted(missing)}")
    predictions["label"] = predictions["label"].astype(int)
    predictions["score"] = predictions["score"].astype(float)
    return predictions


def _load_thresholds(root: Path) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    _maybe_add_threshold(
        thresholds,
        root / "outputs/sprint5b/graph_c/diagnostics/gcn_graph_c_fixed_threshold_metrics.csv",
        "graph_c_gcn_s5b",
        split="test",
    )
    _maybe_add_threshold(
        thresholds,
        root / "outputs/sprint6/loss_comparison/diagnostics_sprint6/imbalance_threshold_metrics.csv",
        "graph_a_gcn_s6_weighted_bce",
        split="test",
        row_filter_column="run_id",
        row_filter_contains="S6R0_wbce",
    )
    _maybe_add_threshold_from_results(
        thresholds,
        root / "outputs/sprint7/gat_comparison.csv",
        "graph_a_gatv2_s7",
        "S7R2_gatv2_edge_aware",
    )
    _maybe_add_threshold_from_results(
        thresholds,
        root / "outputs/sprint7b/gatv2_topology_comparison.csv",
        "graph_b_gcn_s7b",
        "S7B_R1_graph_b_gcn_s5f2",
    )
    _maybe_add_threshold_from_results(
        thresholds,
        root / "outputs/sprint7b/gatv2_topology_comparison.csv",
        "graph_b_gatv2_s7b",
        "S7B_R2_graph_b_gatv2_s5f2",
    )
    _maybe_add_threshold_from_results(
        thresholds,
        root / "outputs/sprint7b/gatv2_topology_comparison.csv",
        "graph_c_gatv2_s7b",
        "S7B_R3_graph_c_gatv2_s5f2",
    )
    return thresholds


def _maybe_add_threshold(
    thresholds: dict[str, float],
    path: Path,
    model_id: str,
    *,
    split: str,
    row_filter_column: str | None = None,
    row_filter_contains: str | None = None,
) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    rows = frame.loc[frame["split"].astype(str) == split].copy() if "split" in frame.columns else frame.copy()
    if row_filter_column is not None and row_filter_column in rows.columns:
        rows = rows.loc[rows[row_filter_column].astype(str).str.contains(str(row_filter_contains), regex=False)]
    if not rows.empty and "threshold" in rows.columns and not pd.isna(rows.iloc[0]["threshold"]):
        thresholds[model_id] = float(rows.iloc[0]["threshold"])


def _maybe_add_threshold_from_results(
    thresholds: dict[str, float],
    path: Path,
    model_id: str,
    predeclared_run_id: str,
) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    rows = frame.loc[frame["predeclared_run_id"].astype(str) == predeclared_run_id]
    if not rows.empty and "threshold" in rows.columns and not pd.isna(rows.iloc[0]["threshold"]):
        thresholds[model_id] = float(rows.iloc[0]["threshold"])


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _alignment_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("analysis_model_id", sort=False):
        for split, split_group in group.groupby("split", sort=False):
            positives = int(split_group["label"].sum())
            row_count = int(len(split_group))
            genome_missing_rate = (
                float(split_group["genome"].isna().mean()) if "genome" in split_group.columns else float("nan")
            )
            rows.append(
                {
                    "audit_type": "single_model_split",
                    "analysis_model_id": model_id,
                    "comparison_model_id": None,
                    "split": split,
                    "rows": row_count,
                    "positives": positives,
                    "negatives": row_count - positives,
                    "positive_rate": positives / row_count if row_count else float("nan"),
                    "row_index_unique": bool(split_group["row_index"].is_unique),
                    "genome_missing_rate": genome_missing_rate,
                    "passed": _split_distribution_passes(split, row_count, positives),
                    "note": _single_model_note(split, row_count, positives, genome_missing_rate),
                }
            )
    model_ids = sorted(predictions["analysis_model_id"].unique())
    for left_index, left_id in enumerate(model_ids):
        for right_id in model_ids[left_index + 1 :]:
            for split in sorted(set(predictions["split"].astype(str))):
                rows.append(_pair_alignment_row(predictions, left_id, right_id, split))
    return pd.DataFrame(rows)


def _single_model_note(split: str, rows: int, positives: int, genome_missing_rate: float) -> str:
    notes = []
    if not _split_distribution_passes(split, rows, positives):
        notes.append("distribution differs from expected headline test counts" if split == "test" else "non-test split audited")
    if not math.isnan(genome_missing_rate) and genome_missing_rate > 0:
        notes.append(f"genome missing rate {genome_missing_rate:.3f}; metadata join required for per-genome claims")
    return "; ".join(notes) if notes else "passed basic split distribution audit"


def _split_distribution_passes(split: str, rows: int, positives: int) -> bool:
    if split != "test":
        return True
    return rows == EXPECTED_TEST_ROWS and positives == EXPECTED_TEST_POSITIVES


def _pair_alignment_row(predictions: pd.DataFrame, left_id: str, right_id: str, split: str) -> dict[str, object]:
    left = predictions.loc[(predictions["analysis_model_id"] == left_id) & (predictions["split"].astype(str) == split)]
    right = predictions.loc[(predictions["analysis_model_id"] == right_id) & (predictions["split"].astype(str) == split)]
    if left.empty or right.empty:
        return {
            "audit_type": "pair_alignment",
            "analysis_model_id": left_id,
            "comparison_model_id": right_id,
            "split": split,
            "rows": 0,
            "positives": 0,
            "negatives": 0,
            "positive_rate": float("nan"),
            "row_index_unique": False,
            "genome_missing_rate": float("nan"),
            "passed": False,
            "note": "one or both models lack split rows",
        }
    left_key = _identity_key(left)
    right_key = _identity_key(right)
    aligned = left_key.equals(right_key)
    label_aligned = left["label"].reset_index(drop=True).equals(right["label"].reset_index(drop=True))
    guide_aligned = left["grna_target_id"].astype(str).reset_index(drop=True).equals(
        right["grna_target_id"].astype(str).reset_index(drop=True)
    )
    rows_equal = len(left) == len(right)
    passed = bool(rows_equal and aligned and label_aligned and guide_aligned)
    return {
        "audit_type": "pair_alignment",
        "analysis_model_id": left_id,
        "comparison_model_id": right_id,
        "split": split,
        "rows": int(min(len(left), len(right))),
        "positives": int(left["label"].sum()) if rows_equal else None,
        "negatives": int((left["label"] == 0).sum()) if rows_equal else None,
        "positive_rate": float(left["label"].mean()) if rows_equal else float("nan"),
        "row_index_unique": bool(left["row_index"].is_unique and right["row_index"].is_unique),
        "genome_missing_rate": float("nan"),
        "passed": passed,
        "note": (
            "aligned on row_index/grna_target_id/label; source-row identity still requires separate metadata audit"
            if passed
            else f"failed alignment: rows_equal={rows_equal}, identity_key={aligned}, label={label_aligned}, guide={guide_aligned}"
        ),
    }


def _identity_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["row_index", "grna_target_id", "label"])[["row_index", "grna_target_id", "label"]].reset_index(
        drop=True
    )


def _comparison_passed(alignment: pd.DataFrame, left_id: str, right_id: str, *, split: str) -> bool:
    rows = alignment.loc[
        (alignment["audit_type"] == "pair_alignment")
        & (alignment["analysis_model_id"] == left_id)
        & (alignment["comparison_model_id"] == right_id)
        & (alignment["split"].astype(str) == split)
    ]
    if rows.empty:
        rows = alignment.loc[
            (alignment["audit_type"] == "pair_alignment")
            & (alignment["analysis_model_id"] == right_id)
            & (alignment["comparison_model_id"] == left_id)
            & (alignment["split"].astype(str) == split)
        ]
    return bool(not rows.empty and bool(rows.iloc[0]["passed"]))


def _metric_recomputation(predictions: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_id, split), group in predictions.groupby(["analysis_model_id", "split"], sort=False):
        y_true = group["label"].astype(int)
        scores = group["score"].astype(float)
        threshold = thresholds.get(str(model_id))
        payload = {
            "analysis_model_id": model_id,
            "analysis_predeclared_run_id": group["analysis_predeclared_run_id"].iloc[0],
            "source_label": group["analysis_source_label"].iloc[0],
            "split": split,
            "rows": int(len(group)),
            "positives": int(y_true.sum()),
            "negatives": int((y_true == 0).sum()),
            "positive_rate": float(y_true.mean()),
            "auprc": float(average_precision_score(y_true, scores)),
            "auroc": _safe_auroc(y_true, scores),
            "threshold": threshold,
            "threshold_available": threshold is not None,
        }
        payload.update(_threshold_metrics(y_true, scores, threshold) if threshold is not None else _empty_threshold_metrics())
        rows.append(payload)
    return pd.DataFrame(rows)


def _safe_auroc(y_true: pd.Series, scores: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _threshold_metrics(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, object]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _empty_threshold_metrics() -> dict[str, object]:
    return {
        "f1": float("nan"),
        "macro_f1": float("nan"),
        "mcc": float("nan"),
        "specificity": float("nan"),
        "sensitivity": float("nan"),
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
    }


def _threshold_transfer(metric_recomputation: pd.DataFrame) -> pd.DataFrame:
    rows = metric_recomputation.loc[metric_recomputation["split"].isin(["val", "test"])].copy()
    wanted = [
        "analysis_model_id",
        "analysis_predeclared_run_id",
        "source_label",
        "split",
        "threshold",
        "threshold_available",
        "auprc",
        "auroc",
        "f1",
        "macro_f1",
        "mcc",
        "specificity",
        "sensitivity",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    return rows[[column for column in wanted if column in rows.columns]].sort_values(["analysis_model_id", "split"])


def _score_distribution_by_label(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, split, label), group in predictions.groupby(["analysis_model_id", "split", "label"], sort=False):
        scores = group["score"].astype(float)
        rows.append(
            {
                "analysis_model_id": model_id,
                "split": split,
                "label": int(label),
                "rows": int(len(group)),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "std_score": float(scores.std(ddof=0)),
                "q05_score": float(scores.quantile(0.05)),
                "q25_score": float(scores.quantile(0.25)),
                "q75_score": float(scores.quantile(0.75)),
                "q95_score": float(scores.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _negative_rank_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, split), group in predictions.groupby(["analysis_model_id", "split"], sort=False):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        negatives = ranked.loc[ranked["label"] == 0].copy()
        positives = ranked.loc[ranked["label"] == 1].copy()
        rows.append(
            {
                "analysis_model_id": model_id,
                "split": split,
                "rows": int(len(ranked)),
                "negatives": int(len(negatives)),
                "negative_median_rank": float(negatives["rank"].median()) if not negatives.empty else float("nan"),
                "negative_mean_rank": float(negatives["rank"].mean()) if not negatives.empty else float("nan"),
                "negative_top_decile_count": int((negatives["rank"] <= max(1, math.ceil(len(ranked) * 0.1))).sum()),
                "positive_bottom_decile_count": int((positives["rank"] > math.floor(len(ranked) * 0.9)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _error_transitions(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    baseline_model_id: str,
    candidate_model_id: str,
    split: str,
) -> pd.DataFrame:
    baseline_threshold = thresholds[baseline_model_id]
    candidate_threshold = thresholds[candidate_model_id]
    baseline = _model_split(predictions, baseline_model_id, split)
    candidate = _model_split(predictions, candidate_model_id, split)
    base_cols = ["row_index", "grna_target_id", "genome", "label", "score"]
    baseline = baseline[base_cols].rename(columns={"score": "baseline_score", "genome": "baseline_genome"})
    candidate = candidate[base_cols].rename(columns={"score": "candidate_score", "genome": "candidate_genome"})
    merged = baseline.merge(candidate, on=["row_index", "grna_target_id", "label"], how="inner", validate="one_to_one")
    merged["baseline_pred"] = (merged["baseline_score"] >= baseline_threshold).astype(int)
    merged["candidate_pred"] = (merged["candidate_score"] >= candidate_threshold).astype(int)
    merged["baseline_confusion"] = [
        _confusion_name(label, pred) for label, pred in zip(merged["label"], merged["baseline_pred"], strict=True)
    ]
    merged["candidate_confusion"] = [
        _confusion_name(label, pred) for label, pred in zip(merged["label"], merged["candidate_pred"], strict=True)
    ]
    merged["transition"] = merged["baseline_confusion"] + "_to_" + merged["candidate_confusion"]
    merged["score_delta_candidate_minus_baseline"] = merged["candidate_score"] - merged["baseline_score"]
    merged["baseline_model_id"] = baseline_model_id
    merged["candidate_model_id"] = candidate_model_id
    merged["split"] = split
    return merged[
        [
            "baseline_model_id",
            "candidate_model_id",
            "split",
            "row_index",
            "grna_target_id",
            "baseline_genome",
            "candidate_genome",
            "label",
            "baseline_score",
            "candidate_score",
            "score_delta_candidate_minus_baseline",
            "baseline_pred",
            "candidate_pred",
            "baseline_confusion",
            "candidate_confusion",
            "transition",
        ]
    ]


def _model_split(predictions: pd.DataFrame, model_id: str, split: str) -> pd.DataFrame:
    rows = predictions.loc[
        (predictions["analysis_model_id"] == model_id) & (predictions["split"].astype(str) == split)
    ].copy()
    if rows.empty:
        raise ValueError(f"No prediction rows for {model_id} split={split}")
    return rows.sort_values(["row_index", "grna_target_id", "label"]).reset_index(drop=True)


def _confusion_name(label: int, pred: int) -> str:
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return "TP"


def _blocked_transition_table(baseline_model_id: str, candidate_model_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline_model_id": baseline_model_id,
                "candidate_model_id": candidate_model_id,
                "split": "test",
                "status": "blocked",
                "reason": "identity/alignment audit failed",
            }
        ]
    )


def _per_guide_error_gain(transitions: pd.DataFrame) -> pd.DataFrame:
    if "transition" not in transitions.columns:
        return pd.DataFrame()
    rows = []
    for guide, group in transitions.groupby("grna_target_id", dropna=False):
        negatives = group.loc[group["label"] == 0]
        rows.append(
            {
                "grna_target_id": guide,
                "rows": int(len(group)),
                "negatives": int(len(negatives)),
                "baseline_tn": int((group["baseline_confusion"] == "TN").sum()),
                "candidate_tn": int((group["candidate_confusion"] == "TN").sum()),
                "recovered_fp_to_tn": int((group["transition"] == "FP_to_TN").sum()),
                "lost_tn_to_fp": int((group["transition"] == "TN_to_FP").sum()),
                "new_fn": int((group["transition"] == "TP_to_FN").sum()),
                "mean_score_delta": float(group["score_delta_candidate_minus_baseline"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["recovered_fp_to_tn", "candidate_tn"], ascending=False)


def _attention_edge_kind_summary(attention: pd.DataFrame) -> pd.DataFrame:
    if attention.empty:
        return pd.DataFrame(
            [
                {
                    "status": "blocked",
                    "note": "Sprint 7B aggregate attention CSV was not found. No attention claims are available.",
                }
            ]
        )
    grouped = (
        attention.groupby(["predeclared_run_id", "graph_schema", "architecture", "split", "edge_kind"], dropna=False)
        .agg(
            rows=("attention_mean", "size"),
            total_edges=("edge_count", "sum"),
            mean_attention=("attention_mean", "mean"),
            mean_attention_std=("attention_std", "mean"),
            min_attention=("attention_min", "min"),
            max_attention=("attention_max", "max"),
        )
        .reset_index()
    )
    grouped["interpretation_limit"] = "aggregate edge-kind attention only; no row-level confusion-category claim"
    return grouped


def _write_figures(
    *,
    predictions: pd.DataFrame,
    transitions: pd.DataFrame,
    per_guide_gain: pd.DataFrame,
    attention_summary: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    test = predictions.loc[predictions["split"].astype(str) == "test"].copy()

    path = figures_dir / "sprint7c_score_distribution_by_label.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_id, group in test.groupby("analysis_model_id", sort=False):
        negatives = group.loc[group["label"] == 0, "score"].astype(float)
        positives = group.loc[group["label"] == 1, "score"].astype(float)
        ax.scatter([model_id], [negatives.mean()], color="tab:red", marker="x", label="negative mean" if not paths else None)
        ax.scatter([model_id], [positives.mean()], color="tab:blue", marker="o", label="positive mean" if not paths else None)
    ax.set_ylabel("Mean test score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if "transition" in transitions.columns:
        path = figures_dir / "sprint7c_graphc_score_delta.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        transitions["score_delta_candidate_minus_baseline"].astype(float).hist(bins=30, ax=ax)
        ax.set_xlabel("Graph C GATv2 score - Graph C GCN score")
        ax.set_ylabel("Rows")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "sprint7c_negative_rank_shift.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        negatives = transitions.loc[transitions["label"] == 0].copy()
        colors = ["tab:green" if item == "FP_to_TN" else "tab:gray" for item in negatives["transition"]]
        ax.scatter(negatives["baseline_score"], negatives["candidate_score"], c=colors, alpha=0.75)
        ax.set_xlabel("Graph C GCN negative score")
        ax.set_ylabel("Graph C GATv2 negative score")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        path = figures_dir / "sprint7c_error_transition_matrix.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        matrix = pd.crosstab(transitions["baseline_confusion"], transitions["candidate_confusion"])
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
        ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
        for row_index, row_name in enumerate(matrix.index):
            for col_index, col_name in enumerate(matrix.columns):
                ax.text(col_index, row_index, int(matrix.loc[row_name, col_name]), ha="center", va="center")
        ax.set_xlabel("Graph C GATv2")
        ax.set_ylabel("Graph C GCN")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    path = figures_dir / "sprint7c_threshold_transfer.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    threshold_view = _metric_recomputation(predictions, _load_thresholds(ROOT))
    threshold_view = threshold_view.loc[threshold_view["split"].astype(str) == "test"]
    threshold_view.set_index("analysis_model_id")[["mcc", "specificity"]].astype(float).plot(kind="bar", ax=ax)
    ax.set_ylabel("Test threshold metric")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if not per_guide_gain.empty and "recovered_fp_to_tn" in per_guide_gain.columns:
        path = figures_dir / "sprint7c_per_guide_negative_gain.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        top = per_guide_gain.head(20)
        ax.bar(top["grna_target_id"].astype(str), top["recovered_fp_to_tn"].astype(int))
        ax.set_ylabel("FP to TN recovered negatives")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    if "edge_kind" in attention_summary.columns:
        path = figures_dir / "sprint7c_attention_edge_kind_summary.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        pivot = (
            attention_summary.groupby(["predeclared_run_id", "edge_kind"], dropna=False)["mean_attention"]
            .mean()
            .unstack("edge_kind")
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean attention")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _identity_report(alignment: pd.DataFrame) -> str:
    pair_failures = alignment.loc[(alignment["audit_type"] == "pair_alignment") & (~alignment["passed"].astype(bool))]
    graphc_pass = _comparison_passed(alignment, "graph_c_gcn_s5b", "graph_c_gatv2_s7b", split="test")
    status = "PASSED" if graphc_pass else "BLOCKED"
    return f"""# Sprint 7C Identity / Alignment Audit

Graph C GCN to Graph C GATv2 test transition gate: `{status}`

## Scope

This audit checks prediction-row comparability. Passing this audit means the
loaded prediction rows align on `row_index`, `grna_target_id`, and `label`.
It does not prove that `row_index` is a stable raw source-row ID for metadata
profiling; that remains a separate metadata-join gate.

## Summary

- Audit rows: `{len(alignment)}`
- Pair-alignment failures: `{len(pair_failures)}`
- Graph C transition analysis allowed: `{graphc_pass}`
- Per-genome claims from Graph C prediction CSVs remain blocked if `genome` is
  missing; use a verified metadata join before making those claims.

## Artifact

- `outputs/sprint7c/diagnostics/sprint7c_prediction_alignment_audit.csv`
"""


def _analysis_report(
    *,
    alignment: pd.DataFrame,
    metric_recomputation: pd.DataFrame,
    threshold_transfer: pd.DataFrame,
    score_distribution: pd.DataFrame,
    negative_rank: pd.DataFrame,
    transitions: pd.DataFrame,
    per_guide_gain: pd.DataFrame,
    attention_summary: pd.DataFrame,
    figure_paths: Iterable[Path],
) -> str:
    graphc_metrics = metric_recomputation.loc[
        metric_recomputation["analysis_model_id"].isin(["graph_c_gcn_s5b", "graph_c_gatv2_s7b"])
        & (metric_recomputation["split"].astype(str) == "test")
    ].copy()
    transition_summary = _transition_summary_markdown(transitions)
    figure_lines = "\n".join(f"- `{_relative(path)}`" for path in figure_paths)
    return f"""# Sprint 7C Graph C GATv2 Explanation Report

Generated at UTC: `{datetime.datetime.now(datetime.UTC).isoformat()}`

## Contract

- Analysis-only sprint: no model was trained and no threshold/model/graph
  setting was changed.
- Label/split/evaluation remain Scheme A, `sprint2_main_seed42`,
  guide-disjoint, measured-only, validation-selected checkpoint/threshold.
- AUPRC remains the primary metric; MCC, macro F1, specificity and TN/FP/FN/TP
  are threshold diagnostics.
- Attention summaries are model-interpretation artifacts only, not biological
  causal evidence.
- Graph C is topology plus target-observation semantics/context, not topology
  only.

## Identity Gate

{_markdown_table(alignment.loc[alignment["audit_type"] == "single_model_split"].head(12))}

Graph C GCN -> Graph C GATv2 row-level transition allowed:
`{_comparison_passed(alignment, "graph_c_gcn_s5b", "graph_c_gatv2_s7b", split="test")}`.

## Metric Recheck

{_markdown_table(graphc_metrics)}

## Threshold Transfer

{_markdown_table(threshold_transfer.loc[threshold_transfer["analysis_model_id"].isin(["graph_c_gcn_s5b", "graph_c_gatv2_s7b"])])}

## Error Transitions

{transition_summary}

## Score Distribution And Negative Rank

Score distribution and rank diagnostics are written as tables. Use them to
describe model score movement, not to select a new threshold or model.

Key Graph C score rows:

{_markdown_table(score_distribution.loc[score_distribution["analysis_model_id"].isin(["graph_c_gcn_s5b", "graph_c_gatv2_s7b"])])}

Negative rank summary:

{_markdown_table(negative_rank.loc[negative_rank["analysis_model_id"].isin(["graph_c_gcn_s5b", "graph_c_gatv2_s7b"])])}

## Per-Guide Concentration

{_markdown_table(per_guide_gain.head(20))}

## Attention Summary

Existing Sprint 7B attention evidence is aggregate by edge kind/layer/head.
It does not support attention-by-confusion-category claims.

{_markdown_table(attention_summary.head(20))}

## Interpretation

Sprint 7C explains the observed Sprint 7B behavior as an operating-point
rare-negative separation improvement by Graph C GATv2, conditional on the
identity audit. It does not prove robustness, and it does not prove biological
causality. Graph C GATv2's MCC/specificity gain is meaningful as a secondary
diagnostic, while XGBoost F4 remains the AUPRC bar.

## Artifact Index

Diagnostics:

- `outputs/sprint7c/diagnostics/sprint7c_prediction_alignment_audit.csv`
- `outputs/sprint7c/diagnostics/sprint7c_metric_recomputation.csv`
- `outputs/sprint7c/diagnostics/sprint7c_threshold_transfer.csv`
- `outputs/sprint7c/diagnostics/sprint7c_score_distribution_by_label.csv`
- `outputs/sprint7c/diagnostics/sprint7c_negative_rank_summary.csv`
- `outputs/sprint7c/diagnostics/sprint7c_error_transitions.csv`
- `outputs/sprint7c/diagnostics/sprint7c_per_guide_error_gain.csv`
- `outputs/sprint7c/diagnostics/sprint7c_attention_edge_kind_summary.csv`

Figures:

{figure_lines}
"""


def _transition_summary_markdown(transitions: pd.DataFrame) -> str:
    if "transition" not in transitions.columns:
        return _markdown_table(transitions)
    summary = transitions.groupby(["baseline_confusion", "candidate_confusion", "transition"], dropna=False).size().reset_index(
        name="rows"
    )
    recovered = int((transitions["transition"] == "FP_to_TN").sum())
    lost = int((transitions["transition"] == "TN_to_FP").sum())
    new_fn = int((transitions["transition"] == "TP_to_FN").sum())
    return (
        f"- Recovered negatives (`FP_to_TN`): `{recovered}`\n"
        f"- Lost negatives (`TN_to_FP`): `{lost}`\n"
        f"- Net TN gain: `{recovered - lost}`\n"
        f"- New false negatives (`TP_to_FN`): `{new_fn}`\n\n"
        f"{_markdown_table(summary)}"
    )


def _write_manifest(
    path: Path,
    *,
    output_dir: Path,
    diagnostics: list[Path],
    figures: list[Path],
    reports: list[Path],
) -> None:
    payload = {
        "manifest_type": "sprint7c_graphc_gatv2_explanation_manifest",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "contract": {
            "analysis_only": True,
            "no_model_training": True,
            "no_test_tuning": True,
            "primary_metric": "AUPRC",
        },
        "output_dir": _relative(output_dir),
        "diagnostics": [_relative(path) for path in diagnostics],
        "figures": [_relative(path) for path in figures],
        "reports": [_relative(path) for path in reports],
        "prediction_sources": [
            {
                "analysis_model_id": source.analysis_model_id,
                "predeclared_run_id": source.predeclared_run_id,
                "path": source.path.as_posix(),
                "filter": (
                    {source.row_filter_column: source.row_filter_value}
                    if source.row_filter_column is not None
                    else None
                ),
            }
            for source in PREDICTION_SOURCES
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.head(max_rows).copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            table[column] = table[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _index, row in table.iterrows():
        lines.append("| " + " | ".join(_escape_markdown_cell(row[column]) for column in table.columns) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    sys.exit(main())
