"""Sprint 9 robustness runner.

Slice 1: prediction registry + metric replay. Loads the frozen Sprint 7F/8A/8B
per-row predictions and their validation-selected thresholds (read, never
recomputed from test) and replays full-split metrics, writing a replay-check table
that must match the source comparison CSVs within tolerance. Later slices extend
this runner with guide-cluster bootstrap (Slice 3), paired-difference (Slice 4),
multi-seed consolidation (Slice 5), and the final report (Slice 6).

See ``docs/exec-plans/active/009-sprint9-robustness.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.evaluation.robustness import (  # noqa: E402
    DEFAULT_REPLAY_ATOL,
    GNN_REGISTRY,
    replay_check_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 9 robustness — Slice 1 metric replay")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/sprint9/diagnostics/metric_replay_check.csv"),
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_REPLAY_ATOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = replay_check_records(GNN_REGISTRY, split="test", atol=args.atol)
    table = pd.DataFrame.from_records(records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    n_models = int(table["registry_id"].nunique())
    n_fail = int((~table["within_tol"]).sum())
    print(
        f"Sprint 9 Slice 1 metric replay: {n_models} models, {len(table)} metric checks, "
        f"atol={args.atol}"
    )
    print(f"Output: {output.relative_to(ROOT)}\n")
    print("AUPRC (primary) replay vs source:")
    for _, row in table[table["metric"] == "auprc"].iterrows():
        print(
            f"  {row['registry_id']:8s} src={row['source_value']:.6f} "
            f"replay={row['replay_value']:.6f} diff={row['abs_diff']:.2e}"
        )

    if n_fail:
        print(f"\nFAIL: {n_fail} metric(s) outside tolerance:")
        print(table[~table["within_tol"]].to_string(index=False))
        return 1
    print("\nAll replayed metrics match the source comparison CSVs within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
