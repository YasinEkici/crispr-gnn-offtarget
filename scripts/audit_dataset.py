from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crispr_gnn.data.audit_console import (  # noqa: E402
    load_columns,
    load_dataset,
    print_check,
    print_info,
    report_step_1,
    report_step_2,
    report_step_4,
    report_step_5,
    report_step_6,
    report_step_7,
    report_step_8,
    report_step_9,
    report_step_10,
)
from crispr_gnn.data.audit_reports import generate_reports  # noqa: E402
from crispr_gnn.utils.config import load_yaml  # noqa: E402


REPORT_DIR = ROOT / "outputs" / "sprint1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the configured dataset.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--sample", action="store_true", help="Run a lightweight sample audit.")
    return parser.parse_args()


def resolve_audit_path(dataset: dict[str, object], sample: bool) -> Path:
    key = "sample_path" if sample else "raw_path"
    return ROOT / Path(str(dataset.get(key, "")))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    dataset = config.get("dataset", {})
    raw_path = resolve_audit_path(dataset, args.sample)

    can_load = report_step_1(config, config_path, ROOT / Path(str(dataset.get("raw_path", ""))), args.sample, ROOT)
    if args.sample:
        sample_path = ROOT / Path(str(dataset.get("sample_path", "")))
        print(f"Resolved sample path: {sample_path}")
        if sample_path.exists():
            raw_path = sample_path
            can_load = True
        elif can_load:
            print_info("Sample file is unavailable; falling back to raw parquet metadata for Step 1/2 smoke checks.")
            raw_path = ROOT / Path(str(dataset.get("raw_path", "")))
        else:
            print_info("Sample file is unavailable.")
            print("Step 2 skipped because neither full raw data nor sample data is available.")
            return 0

    if not can_load:
        print("Step 2 skipped because the configured raw dataset is unavailable.")
        return 1

    columns, row_count = load_columns(raw_path, args.sample)
    report_step_2(columns, row_count)

    if args.sample:
        print()
        print("Sample audit completed after Step 2. Full dataset load and Steps 3-10 were intentionally skipped.")
        return 0

    df = load_dataset(raw_path)
    report_step_4(df)
    cleavage = report_step_5(df, columns)
    report_step_6(df, cleavage, columns)
    report_step_7(df, cleavage)
    feature_parse_counts = report_step_8(df)
    report_step_9(df, feature_parse_counts)
    report_step_10(df, cleavage)

    print()
    print("Step 11: Generate report artifacts")
    for path in generate_reports(df, columns, cleavage, feature_parse_counts, REPORT_DIR):
        print_check(f"Report artifact written: {path.relative_to(ROOT).as_posix()}", path.exists(), True, path.exists())
    print()
    print("Step 11 audit reports completed. Canonical docs updates are intentionally left for Step 12.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
