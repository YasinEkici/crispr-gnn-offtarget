"""Sprint 9 robustness runner.

Stages (009 plan §12), all CPU, reading frozen predictions — no retraining:

- ``replay``    Slice 1 — prediction registry + metric replay vs the source reports.
- ``bootstrap`` Slice 3 — guide-cluster bootstrap CIs (percentile primary, BCa
                sensitivity) + fragility diagnostics + figures, over the full
                registry (GNN + regenerated XGBoost F4).

Later slices add paired-difference (Slice 4), multi-seed consolidation (Slice 5),
and the final report (Slice 6). Resampling is by guide (``grna_target_id``), never
rows; thresholds are read, never recomputed from test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.evaluation.plots import write_sprint9_bootstrap_plots  # noqa: E402
from crispr_gnn.evaluation.robustness import (  # noqa: E402
    BOOTSTRAP_METRICS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    DEFAULT_REPLAY_ATOL,
    GNN_REGISTRY,
    NO_SKILL_BASELINE,
    guide_cluster_bootstrap,
    leave_one_guide_influence,
    load_full_registry,
    replay_check_records,
)

OUTPUT_ROOT = ROOT / "outputs/sprint9"


def _rel(path: Path) -> Path:
    """Repo-relative display path; falls back to the absolute path (e.g. tmp dirs)."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 9 robustness runner")
    parser.add_argument("--stage", choices=["replay", "bootstrap", "all"], default="all")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--atol", type=float, default=DEFAULT_REPLAY_ATOL, help="Slice 1 replay tolerance")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Slice 3 bootstrap replicates")
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="Slice 3 bootstrap RNG seed")
    parser.add_argument("--ci", type=float, default=0.95)
    return parser.parse_args()


def run_replay_stage(output_root: Path, atol: float) -> int:
    records = replay_check_records(GNN_REGISTRY, split="test", atol=atol)
    table = pd.DataFrame.from_records(records)
    output = output_root / "diagnostics/metric_replay_check.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    n_fail = int((~table["within_tol"]).sum())
    print(f"[replay] {table['registry_id'].nunique()} models, {len(table)} checks, atol={atol}")
    print(f"[replay] output: {_rel(output)}")
    if n_fail:
        print(f"[replay] FAIL: {n_fail} metric(s) outside tolerance")
        print(table[~table["within_tol"]].to_string(index=False))
        return 1
    print("[replay] all replayed metrics match source within tolerance.")
    return 0


def run_bootstrap_stage(output_root: Path, *, n_boot: int, seed: int, ci: float) -> int:
    registry = load_full_registry()
    ci_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    influence_rows: list[dict[str, object]] = []
    distributions: dict[str, dict[str, object]] = {}

    for registry_id, scores in registry.items():
        result = guide_cluster_bootstrap(scores, split="test", n_boot=n_boot, seed=seed, ci=ci)
        distributions[registry_id] = result.samples
        influence_rows.extend(leave_one_guide_influence(result))
        diag = {
            "registry_id": registry_id,
            "sprint": scores.sprint,
            "n_test_guides": result.n_test_guides,
            "n_boot": result.n_boot,
            "ci": result.ci,
            "seed": result.seed,
            "mean_unique_guides_per_replicate": result.mean_unique_guides,
            "mean_negative_bearing_guides_per_replicate": result.mean_negative_bearing_guides,
            "dominant_guide": result.dominant_guide,
            "dominant_guide_inclusion_rate": result.dominant_guide_inclusion_rate,
        }
        for metric in BOOTSTRAP_METRICS:
            pct_lo, pct_hi = result.percentile[metric]
            bca_lo, bca_hi = result.bca[metric]
            shape = result.shape[metric]
            ci_rows.append(
                {
                    "registry_id": registry_id,
                    "sprint": scores.sprint,
                    "metric": metric,
                    "point": result.point[metric],
                    "percentile_lo": pct_lo,
                    "percentile_hi": pct_hi,
                    "bca_lo": bca_lo,
                    "bca_hi": bca_hi,
                    "bca_trusted": result.bca_trusted[metric],
                    "bca_note": result.bca_note[metric],
                    "undefined_rate": result.undefined_rate[metric],
                    "n_unique_samples": shape["n_unique"],
                    "frac_at_upper_bound": shape["frac_at_upper_bound"],
                    "shape_flag": shape["flag"],
                    "interval_type": "guide_cluster_finite_sample_compatibility",
                    "threshold": result.threshold,
                    "threshold_metric": metric in ("mcc", "specificity", "macro_f1"),
                    "n_test_guides": result.n_test_guides,
                    "n_boot": result.n_boot,
                    "ci": result.ci,
                    "seed": result.seed,
                    "no_skill_baseline": NO_SKILL_BASELINE,
                }
            )
            diag[f"{metric}_undefined_rate"] = result.undefined_rate[metric]
        diag_rows.append(diag)

    ci_table = pd.DataFrame(ci_rows)
    diag_table = pd.DataFrame(diag_rows)
    influence_table = pd.DataFrame(influence_rows)

    (output_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    ci_path = output_root / "robustness_bootstrap_cis.csv"
    diag_path = output_root / "diagnostics/bootstrap_replicate_diagnostics.csv"
    influence_path = output_root / "diagnostics/leave_one_guide_influence.csv"
    ci_table.to_csv(ci_path, index=False)
    diag_table.to_csv(diag_path, index=False)
    influence_table.to_csv(influence_path, index=False)
    figure_paths = write_sprint9_bootstrap_plots(
        ci_table, distributions, output_root / "figures", baseline=NO_SKILL_BASELINE, metric="auprc"
    )

    print(f"[bootstrap] {len(registry)} models, B={n_boot}, seed={seed}, guide-cluster percentile (BCa sensitivity)")
    print(f"[bootstrap] tables: {_rel(ci_path)}, {_rel(diag_path)}, {_rel(influence_path)}")
    for path in figure_paths:
        print(f"[bootstrap] figure: {_rel(path)}")
    print(f"\nAUPRC (primary) — point [percentile 95% CI]  (no-skill baseline {NO_SKILL_BASELINE:.4f}):")
    auprc = ci_table[ci_table["metric"] == "auprc"].sort_values("point", ascending=False)
    for _, r in auprc.iterrows():
        print(
            f"  {r['registry_id']:8s} {r['point']:.6f}  [{r['percentile_lo']:.6f}, {r['percentile_hi']:.6f}]"
            f"  bca_trusted={r['bca_trusted']}"
        )
    return 0


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    status = 0
    if args.stage in ("replay", "all"):
        status |= run_replay_stage(output_root, args.atol)
    if args.stage in ("bootstrap", "all"):
        status |= run_bootstrap_stage(output_root, n_boot=args.n_boot, seed=args.seed, ci=args.ci)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
