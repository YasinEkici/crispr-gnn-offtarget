"""Post-hoc guide-level (cluster) bootstrap confidence intervals for Sprint 6.

Quantifies the single-seed/single-split fragility of the returned Slice 4 results
WITHOUT retraining: it reads the already-saved per-row test predictions and each
run's validation-selected threshold, then resamples **guides** (clusters) with
replacement to build percentile confidence intervals.

Why cluster (guide-level) bootstrap, not row-level: rows that share a guide are
correlated, so resampling individual rows underestimates the interval width
(false confidence). The locked test universe has only ~29 guides / 169 negatives,
so the intervals will be wide — that is the honest point of this diagnostic. It is
a reporting/estimation step only: thresholds are the frozen validation-selected
values and are never re-selected per resample (no test-driven tuning).

References: Boyd et al. 2013 (AUPRC point estimates + CIs); cluster/block bootstrap
for grouped data. This is interpretation-only and does not change any model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs/sprint6/loss_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guide-level bootstrap CIs for Sprint 6 test metrics.")
    parser.add_argument("--predictions", default=str(DEFAULT_DIR / "diagnostics_sprint6/sprint6_loss_comparison_predictions.csv"))
    parser.add_argument("--results", default=str(DEFAULT_DIR / "sprint6_loss_comparison_results.csv"))
    parser.add_argument("--output", default=str(DEFAULT_DIR / "diagnostics_sprint6/imbalance_bootstrap_cis.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for the bootstrap itself (reproducibility).")
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    results = pd.read_csv(args.results)

    split_rows = predictions[predictions["split"] == args.split].copy()
    if split_rows.empty:
        raise ValueError(f"No '{args.split}' rows found in {args.predictions}")

    thresholds = results.set_index("run_id")["threshold"].to_dict()
    meta_cols = {c: results.set_index("run_id")[c].to_dict() for c in ("predeclared_run_id", "loss")}

    lo_q = (1.0 - args.ci) / 2.0
    hi_q = 1.0 - lo_q
    rng = np.random.default_rng(args.seed)

    records: list[dict[str, object]] = []
    for run_id in results["run_id"]:
        sub = split_rows[split_rows["run_id"] == run_id]
        if sub.empty:
            continue
        threshold = float(thresholds[run_id])
        # Group rows by guide so a resampled guide contributes ALL its rows.
        guide_arrays = [
            (g["label"].to_numpy().astype(int), g["score"].to_numpy().astype(float))
            for _, g in sub.groupby("grna_target_id", sort=True)
        ]
        n_guides = len(guide_arrays)
        labels_full = sub["label"].to_numpy().astype(int)
        scores_full = sub["score"].to_numpy().astype(float)

        point = _metrics(labels_full, scores_full, threshold)
        boot: dict[str, list[float]] = {m: [] for m in point}
        for _ in range(args.n_boot):
            pick = rng.integers(0, n_guides, size=n_guides)
            labels = np.concatenate([guide_arrays[i][0] for i in pick])
            scores = np.concatenate([guide_arrays[i][1] for i in pick])
            for metric, value in _metrics(labels, scores, threshold).items():
                if not np.isnan(value):
                    boot[metric].append(value)

        for metric, point_value in point.items():
            samples = np.asarray(boot[metric], dtype=float)
            ci_low = float(np.quantile(samples, lo_q)) if samples.size else float("nan")
            ci_high = float(np.quantile(samples, hi_q)) if samples.size else float("nan")
            records.append(
                {
                    "run_id": run_id,
                    "predeclared_run_id": meta_cols["predeclared_run_id"].get(run_id),
                    "loss": meta_cols["loss"].get(run_id),
                    "metric": metric,
                    "point_estimate": float(point_value),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_level": args.ci,
                    "n_boot": args.n_boot,
                    "n_boot_effective": int(samples.size),
                    "n_test_guides": n_guides,
                    "threshold": threshold,
                    "bootstrap": "guide_level_cluster_percentile",
                }
            )

    table = pd.DataFrame.from_records(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    auprc = table[table["metric"] == "auprc"]
    print(f"Guide-level cluster bootstrap ({args.n_boot} resamples, {int(args.ci * 100)}% percentile CI)")
    print(f"Output: {Path(args.output).relative_to(ROOT)}\n")
    print("AUPRC (primary) with CI:")
    for _, r in auprc.iterrows():
        print(
            f"  {str(r['predeclared_run_id']):24s} {r['point_estimate']:.6f}  "
            f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
        )
    return 0


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    both_classes = np.unique(labels).shape[0] == 2
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auprc": float(average_precision_score(labels, scores)) if labels.sum() > 0 else float("nan"),
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both_classes else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
