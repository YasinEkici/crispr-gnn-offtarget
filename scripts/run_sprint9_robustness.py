"""Sprint 9 robustness runner.

Stages (009 plan §12), all CPU, reading frozen predictions — no retraining:

- ``replay``    Slice 1 — prediction registry + metric replay vs the source reports.
- ``bootstrap`` Slice 3 — guide-cluster bootstrap CIs (percentile primary, BCa
                sensitivity) + fragility diagnostics + figures, over the full
                registry (GNN + regenerated XGBoost F4).

Later slices add the final report (Slice 6). Resampling is by guide
(``grna_target_id``), never rows; thresholds are read, never recomputed from test.
Slice 5 multi-seed consolidation reads per-seed validation-selected metrics from
isolated Sprint 9 output directories and reports every seed without best-seed
selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.evaluation.plots import (  # noqa: E402
    write_sprint9_bootstrap_plots,
    write_sprint9_multiseed_plot,
    write_sprint9_paired_plots,
)
from crispr_gnn.evaluation.robustness import (  # noqa: E402
    BOOTSTRAP_METRICS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOT,
    DEFAULT_REPLAY_ATOL,
    GNN_REGISTRY,
    NO_SKILL_BASELINE,
    PAIRED_COMPARISONS,
    collect_multiseed_results,
    guide_cluster_bootstrap,
    leave_one_guide_influence,
    load_full_registry,
    paired_guide_bootstrap,
    replay_check_records,
)

OUTPUT_ROOT = ROOT / "outputs/sprint9"
MULTISEED_CONFIG = ROOT / "configs/sweeps/sprint9_multiseed.yaml"


def _rel(path: Path) -> Path:
    """Repo-relative display path; falls back to the absolute path (e.g. tmp dirs)."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 9 robustness runner")
    parser.add_argument("--stage", choices=["replay", "bootstrap", "paired", "multiseed", "all"], default="all")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--multiseed-config", default=str(MULTISEED_CONFIG))
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
    print(f"\nAUPRC (primary) - point [percentile 95% CI]  (no-skill baseline {NO_SKILL_BASELINE:.4f}):")
    auprc = ci_table[ci_table["metric"] == "auprc"].sort_values("point", ascending=False)
    for _, r in auprc.iterrows():
        print(
            f"  {r['registry_id']:8s} {r['point']:.6f}  [{r['percentile_lo']:.6f}, {r['percentile_hi']:.6f}]"
            f"  bca_trusted={r['bca_trusted']}"
        )
    return 0


def run_paired_stage(output_root: Path, *, n_boot: int, seed: int, ci: float) -> int:
    registry = load_full_registry()
    rows: list[dict[str, object]] = []
    distributions: dict[str, dict[str, object]] = {}

    for comparison_id, a_id, b_id in PAIRED_COMPARISONS:
        if a_id not in registry or b_id not in registry:
            raise KeyError(f"{comparison_id}: registry missing {a_id!r} or {b_id!r}")
        result = paired_guide_bootstrap(
            registry[a_id], registry[b_id], comparison_id=comparison_id, n_boot=n_boot, seed=seed, ci=ci
        )
        distributions[comparison_id] = result.samples
        for metric in BOOTSTRAP_METRICS:
            pct_lo, pct_hi = result.percentile[metric]
            bca_lo, bca_hi = result.bca[metric]
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "a_id": a_id,
                    "b_id": b_id,
                    "metric": metric,
                    "point_a": result.point_a[metric],
                    "point_b": result.point_b[metric],
                    "point_delta": result.point_delta[metric],
                    "percentile_lo": pct_lo,
                    "percentile_hi": pct_hi,
                    "interval_excludes_zero": result.interval_excludes_zero[metric],
                    "prob_positive": result.prob_positive[metric],
                    "bca_lo": bca_lo,
                    "bca_hi": bca_hi,
                    "bca_trusted": result.bca_trusted[metric],
                    "bca_note": result.bca_note[metric],
                    "undefined_delta_rate": result.undefined_delta_rate[metric],
                    "interval_type": "paired_guide_cluster_finite_sample_compatibility",
                    "threshold_metric": metric in ("mcc", "specificity", "macro_f1"),
                    "n_test_guides": result.n_test_guides,
                    "n_boot": result.n_boot,
                    "ci": result.ci,
                    "seed": result.seed,
                }
            )

    table = pd.DataFrame(rows)
    output = output_root / "robustness_paired_differences.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    figure_paths = write_sprint9_paired_plots(table, distributions, output_root / "figures", metric="auprc")

    print(f"[paired] {len(PAIRED_COMPARISONS)} comparisons, B={n_boot}, seed={seed}, common-guide resamples")
    print(f"[paired] table: {_rel(output)}")
    for path in figure_paths:
        print(f"[paired] figure: {_rel(path)}")
    print("\nAUPRC paired delta = A - B  [percentile 95% CI]  (overlap of marginal CIs is NOT used):")
    auprc = table[table["metric"] == "auprc"]
    for _, r in auprc.iterrows():
        verdict = "excludes 0" if bool(r["interval_excludes_zero"]) else "includes 0"
        print(
            f"  {r['comparison_id']} {r['a_id']:7s}-{r['b_id']:7s} delta={r['point_delta']:+.6f}"
            f"  [{r['percentile_lo']:+.6f}, {r['percentile_hi']:+.6f}]  ({verdict})"
        )
    return 0


def run_multiseed_stage(output_root: Path, *, manifest_path: Path) -> int:
    table = collect_multiseed_results(manifest_path)
    output = output_root / "robustness_model_seed_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    figure_paths = write_sprint9_multiseed_plot(table, output_root / "figures")

    seed_rows = table.loc[table["record_type"] == "seed"]
    missing_rows = table.loc[table["record_type"] == "missing_seed"]
    print(f"[multiseed] manifest: {_rel(manifest_path)}")
    print(f"[multiseed] observed seeds: {len(seed_rows)}, missing seeds: {len(missing_rows)}")
    print(f"[multiseed] table: {_rel(output)}")
    for path in figure_paths:
        print(f"[multiseed] figure: {_rel(path)}")
    if not missing_rows.empty:
        print("[multiseed] partial outputs are expected until Colab/CPU seed jobs finish; no best-seed selection.")
    return 0


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    status = 0
    if args.stage in ("replay", "all"):
        status |= run_replay_stage(output_root, args.atol)
    if args.stage in ("bootstrap", "all"):
        status |= run_bootstrap_stage(output_root, n_boot=args.n_boot, seed=args.seed, ci=args.ci)
    if args.stage in ("paired", "all"):
        status |= run_paired_stage(output_root, n_boot=args.n_boot, seed=args.seed, ci=args.ci)
    if args.stage in ("multiseed", "all"):
        status |= run_multiseed_stage(output_root, manifest_path=Path(args.multiseed_config))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
