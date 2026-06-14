"""Sprint 9 Slice 2 — regenerate XGBoost F4 per-row predictions under the Sprint 2
contract.

Sprint 2 saved no per-row XGBoost predictions and no model checkpoint, so the
paired GNN-vs-F4 bootstrap (009 plan §5.1 P4-P6) and the F4 confidence interval
(Slice 3) need the F4 test/validation scores regenerated. This script REPRODUCES
the frozen ``xgboost_unweighted / F4`` run — it does not tune:

- Same code path: reuses ``run_xgboost_baselines`` (the exact Sprint 2 trainer).
- Same config: ``configs/experiments/baseline_xgboost.yaml`` xgboost block, seed 42.
- Same data/split: ``data/raw/260520_putative_nucleosomal.parquet`` +
  ``outputs/splits/sprint2_guides.json`` (``sprint2_main_seed42``), measured-only,
  ``experiment_id=18`` excluded, F4 feature ladder, train-only preprocessing.

Reproduction policy (009 plan §6, refined by the 2026-06-13 Option C decision):
XGBoost is **not bit-reproducible across library versions**. The frozen F4 was
trained on an earlier XGBoost build; this environment uses a newer one, so the
test AUPRC lands a hair off the historical ``0.992522`` (~1e-4) — far below the
guide-cluster bootstrap CI width (~1e-2). The **blocking** gate therefore checks a
faithful reproduction under the current pinned environment: the test geometry must
be exact (1702 rows / 29 guides / 169 negatives) and the test AUPRC must be within
``--sanity-atol`` (default 2e-3) of the historical bar (this still catches genuine
pipeline/leakage breakage). The strict 1e-4 historical match is reported as a
**non-blocking** diagnostic so the version drift is transparent. F4's operating
point uses the **regenerated model's own validation-selected threshold**
(validation_max_f1 on the regenerated validation scores — validation-only, never
recomputed from test), keeping the F4 bar a single self-consistent model.

Outputs (under ``outputs/sprint9/`` only; no frozen Sprint 2 artifact is touched):
- ``outputs/sprint9/diagnostics/f4_predictions.csv`` (registry schema + threshold)
- ``outputs/sprint9/diagnostics/f4_reproduction_check.csv``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.splits import assign_measured_splits, load_split_manifest  # noqa: E402
from crispr_gnn.evaluation.metrics import select_threshold_by_f1  # noqa: E402
from crispr_gnn.training.baselines import XGBoostRunConfig, run_xgboost_baselines  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402

# Frozen Sprint 2 F4 reference (xgboost_unweighted / F4, sprint2_main_seed42),
# trained on an earlier XGBoost build. Used as the historical comparison target.
HISTORICAL_TEST_AUPRC = 0.992522
HISTORICAL_THRESHOLD = 0.605734
HISTORICAL_CONFUSION = {"tn": 38, "fp": 131, "fn": 21, "tp": 1512}
GUIDE_KEY = "grna_target_id"
PREDECLARED_RUN_ID = "XGB_F4"
RUN_ID = "xgboost_unweighted_F4_sprint2_main_seed42_regen"

XGBOOST_CONFIG_PATH = ROOT / "configs/experiments/baseline_xgboost.yaml"
DATA_CONFIG_PATH = ROOT / "configs/data/mak2022.yaml"
SPLIT_PATH = ROOT / "outputs/splits/sprint2_guides.json"
OUTPUT_ROOT = ROOT / "outputs/sprint9"
PRED_OUTPUT = OUTPUT_ROOT / "diagnostics/f4_predictions.csv"
CHECK_OUTPUT = OUTPUT_ROOT / "diagnostics/f4_reproduction_check.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 9 Slice 2 — regenerate XGBoost F4 predictions")
    parser.add_argument(
        "--sanity-atol",
        type=float,
        default=2e-3,
        help="Blocking AUPRC tolerance vs the historical bar (catches real breakage; absorbs version drift).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the XGBoost training seed for Sprint 9 multiseed.")
    parser.add_argument("--output-dir", default=None, help="Output root; defaults to outputs/sprint9.")
    parser.add_argument("--pred-output", default=None)
    parser.add_argument("--check-output", default=None)
    return parser.parse_args()


def _xgboost_config(seed_override: int | None = None) -> XGBoostRunConfig:
    config = load_yaml(str(XGBOOST_CONFIG_PATH))
    split = load_split_manifest(SPLIT_PATH)
    xgb = config.get("xgboost", {})
    return XGBoostRunConfig(
        sprint=str(config.get("sprint", "sprint2")),
        split_id=split.config.split_id,
        seed=int(seed_override if seed_override is not None else config.get("seed", split.config.seed)),
        n_estimators=int(xgb.get("n_estimators", 400)),
        max_depth=int(xgb.get("max_depth", 3)),
        learning_rate=float(xgb.get("learning_rate", 0.05)),
        subsample=float(xgb.get("subsample", 0.9)),
        colsample_bytree=float(xgb.get("colsample_bytree", 0.9)),
        min_child_weight=float(xgb.get("min_child_weight", 5.0)),
        reg_alpha=float(xgb.get("reg_alpha", 0.0)),
        reg_lambda=float(xgb.get("reg_lambda", 1.0)),
        early_stopping_rounds=xgb.get("early_stopping_rounds", 30),
        eval_metric=str(xgb.get("eval_metric", "aucpr")),
        tree_method=str(xgb.get("tree_method", "hist")),
        n_jobs=int(xgb.get("n_jobs", 4)),
    )


def _prediction_frame(assigned: pd.DataFrame, predictions: list[dict[str, object]], *, run_id: str) -> pd.DataFrame:
    """Extract xgboost_unweighted / F4 val+test predictions, keyed by grna_target_id."""
    has_genome = "genome" in assigned.columns
    frames: list[pd.DataFrame] = []
    for record in predictions:
        if record["model_name"] != "xgboost_unweighted" or record["feature_set"] != "F4":
            continue
        row_index = np.asarray(record["row_index"])
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "predeclared_run_id": PREDECLARED_RUN_ID,
                    "split": record["split"],
                    "row_index": row_index,
                    GUIDE_KEY: assigned.loc[row_index, GUIDE_KEY].to_numpy(),
                    "genome": assigned.loc[row_index, "genome"].to_numpy() if has_genome else np.nan,
                    "label": np.asarray(record["y_true"]).astype(int),
                    "score": np.asarray(record["y_score"]).astype(float),
                }
            )
        )
    if not frames:
        raise ValueError("No xgboost_unweighted / F4 predictions were produced")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    output_root = _resolve_output_root(args.output_dir)
    pred_output = Path(args.pred_output) if args.pred_output else output_root / "diagnostics/f4_predictions.csv"
    check_output = Path(args.check_output) if args.check_output else output_root / "diagnostics/f4_reproduction_check.csv"

    raw_path = ROOT / load_yaml(str(DATA_CONFIG_PATH))["dataset"]["raw_path"]
    if not raw_path.exists():
        print(f"Raw dataset not found ({raw_path}); cannot regenerate F4 predictions.")
        return 1

    config = _xgboost_config(seed_override=args.seed)
    run_id = RUN_ID if args.seed is None else f"{RUN_ID}_seed{config.seed}"
    # The vs-historical AUPRC bar reproduces the canonical seed-42 F4 result (Option C
    # version-drift guard). For any other Sprint 9 multi-seed run there is no historical
    # bar to reproduce — the AUPRC *is* the seed-variance signal we want to measure — so
    # that comparison must not block. Geometry checks (rows/guides/negatives) stay
    # blocking for every seed because they catch genuine pipeline breakage.
    is_canonical_seed = args.seed is None or int(config.seed) == 42
    split = load_split_manifest(SPLIT_PATH)
    assigned = assign_measured_splits(pd.read_parquet(raw_path), split)

    results, predictions, _feature_importance, feature_audit = run_xgboost_baselines(
        assigned=assigned,
        feature_sets=["F4"],
        config=config,
        include_balanced=False,
    )
    if feature_audit["is_forbidden"].any():
        forbidden = feature_audit.loc[feature_audit["is_forbidden"], "feature"].unique().tolist()
        raise ValueError(f"Forbidden (leakage) columns in F4 feature audit: {forbidden}")

    preds = _prediction_frame(assigned, predictions, run_id=run_id)
    test = preds[preds["split"] == "test"]
    val = preds[preds["split"] == "val"]

    # F4 operating point = the regenerated model's own validation-selected threshold
    # (validation_max_f1 on regenerated validation scores; never from test).
    f4_threshold = float(select_threshold_by_f1(val["label"].to_numpy(), val["score"].to_numpy()).threshold)
    preds = preds.assign(threshold=f4_threshold, threshold_selection_split="validation")

    test_auprc = float(average_precision_score(test["label"], test["score"]))
    test_pred = (test["score"].to_numpy() >= f4_threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(test["label"], test_pred, labels=[0, 1]).ravel())
    n_test, n_guides, n_neg = len(test), int(test[GUIDE_KEY].nunique()), int((test["label"] == 0).sum())

    auprc_diff = abs(test_auprc - HISTORICAL_TEST_AUPRC)
    checks = [
        # Blocking: faithful reproduction under the current pinned environment.
        ("test_rows", 1702, n_test, abs(n_test - 1702), 0, True),
        ("test_guides", 29, n_guides, abs(n_guides - 29), 0, True),
        ("test_negatives", 169, n_neg, abs(n_neg - 169), 0, True),
        ("test_auprc_sanity_vs_historical", HISTORICAL_TEST_AUPRC, test_auprc, auprc_diff, args.sanity_atol, is_canonical_seed),
        # Non-blocking diagnostics: document the XGBoost version drift transparently.
        ("test_auprc_strict_vs_historical", HISTORICAL_TEST_AUPRC, test_auprc, auprc_diff, 1e-4, False),
        ("regen_threshold_vs_historical", HISTORICAL_THRESHOLD, f4_threshold, abs(f4_threshold - HISTORICAL_THRESHOLD), 1e-6, False),
        ("test_tn_vs_historical", HISTORICAL_CONFUSION["tn"], tn, abs(tn - HISTORICAL_CONFUSION["tn"]), 0, False),
        ("test_fp_vs_historical", HISTORICAL_CONFUSION["fp"], fp, abs(fp - HISTORICAL_CONFUSION["fp"]), 0, False),
        ("test_fn_vs_historical", HISTORICAL_CONFUSION["fn"], fn, abs(fn - HISTORICAL_CONFUSION["fn"]), 0, False),
        ("test_tp_vs_historical", HISTORICAL_CONFUSION["tp"], tp, abs(tp - HISTORICAL_CONFUSION["tp"]), 0, False),
    ]
    check_rows = [
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "abs_diff": diff,
            "atol": tol,
            "pass": bool(diff <= tol),
            "blocking": blocking,
            "xgboost_version": xgboost.__version__,
            "regenerated_threshold": f4_threshold,
            "note": "option_c_version_drift_accepted",
        }
        for name, expected, observed, diff, tol, blocking in checks
    ]
    check_table = pd.DataFrame(check_rows)
    reproduced = bool(check_table.loc[check_table["blocking"], "pass"].all())

    pred_output.parent.mkdir(parents=True, exist_ok=True)
    if reproduced:
        preds.to_csv(pred_output, index=False)
    check_table.to_csv(check_output, index=False)

    mode = "canonical reproduction" if is_canonical_seed else "multi-seed variance (vs-historical non-blocking)"
    print(f"XGBoost {xgboost.__version__} - F4 regeneration (seed {config.seed}; {mode})")
    print(f"Regenerated validation-selected threshold: {f4_threshold:.6f}")
    print(check_table[["check", "expected", "observed", "abs_diff", "atol", "pass", "blocking"]].to_string(index=False))
    if reproduced:
        strict = check_table.loc[check_table["check"] == "test_auprc_strict_vs_historical"].iloc[0]
        if is_canonical_seed:
            print(
                f"\nACCEPTED (Option C). Test AUPRC {test_auprc:.6f} vs historical {HISTORICAL_TEST_AUPRC:.6f} "
                f"(diff {strict['abs_diff']:.2e}; XGBoost version drift, non-blocking)."
            )
        else:
            print(
                f"\nACCEPTED (seed variance). Test AUPRC {test_auprc:.6f}; the vs-historical bar is "
                f"non-blocking for non-canonical seeds (geometry checks passed)."
            )
        print(f"Predictions: {pred_output.relative_to(ROOT)}")
        print(f"Check table: {check_output.relative_to(ROOT)}")
        return 0
    print(f"\nFAILED blocking gate. Check table: {check_output.relative_to(ROOT)}")
    print("F4 predictions NOT written; geometry/reproduction check failed (009 plan section 6).")
    return 1


def _resolve_output_root(output_dir: str | None) -> Path:
    path = Path(output_dir) if output_dir else OUTPUT_ROOT
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
