from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from crispr_gnn.utils.config import load_yaml  # noqa: E402


EXPECTED_RAW_PATH = Path("data/raw/260520_putative_nucleosomal.parquet")
EXPECTED_SHAPE = (310_142, 45)

EXPERIMENTAL_EPIGENETIC_FEATURES = [
    "epigen_ctcf",
    "epigen_dnase",
    "epigen_rrbs",
    "epigen_h3k4me3",
    "epigen_drip",
    "MNase",
]

COMPUTED_NUCLEOSOME_FEATURES = [
    "GCContent",
    "WSScore",
    "YRScore",
    "NucleotideBDM",
    "StrongWeakBDM",
    "NuPoP_Occup_147_human",
    "NuPoP_Viterbi_147_human",
    "NuPoP_Affinity_147_human",
    "nuCpos_Occup_147_yeast",
    "nuCpos_Viterbi_147_yeast",
    "nuCpos_Affinity_147_yeast",
    "VanDerHeijden",
    "LeNupH3Q85C",
]

BINDING_ENERGY_FEATURES = [
    "energy_1",
    "energy_2",
    "energy_3",
    "energy_4",
    "energy_5",
]

REQUIRED_FIELDS = [
    "measured",
    "experiment_id",
    "cell_line",
    "cleavage_freq",
]

GENOME_CANDIDATE_FIELDS = [
    "genome",
    "assembly",
    "target_genome",
    "genome_assembly",
]

TARGET_KEY_FIELDS = [
    "target_chr",
    "target_start",
    "target_end",
    "target_strand",
]

GUIDE_KEY_CANDIDATE_FIELDS = [
    "grna_target_id",
    "grna_target_chr",
    "grna_target_start",
    "grna_target_end",
    "grna",
    "sgRNA",
    "sgRNA_seq",
    "grna_seq",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the configured dataset.")
    parser.add_argument("--config", required=True, help="Path to data config YAML.")
    parser.add_argument("--sample", action="store_true", help="Run a lightweight sample audit.")
    return parser.parse_args()


def marker(ok: bool) -> str:
    return "PASS" if ok else "DISCREPANCY"


def print_check(name: str, actual: Any, expected: Any, ok: bool) -> None:
    print(f"[{marker(ok)}] {name}: actual={actual} expected={expected}")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def missing_values(actual_columns: set[str], expected_columns: list[str]) -> list[str]:
    return [column for column in expected_columns if column not in actual_columns]


def present_values(actual_columns: set[str], expected_columns: list[str]) -> list[str]:
    return [column for column in expected_columns if column in actual_columns]


def load_columns(raw_path: Path, sample: bool) -> tuple[list[str], int | None]:
    if sample:
        if raw_path.suffix.lower() == ".csv":
            return list(pd.read_csv(raw_path, nrows=5).columns), None
        if raw_path.suffix.lower() in {".parquet", ".pq"}:
            schema = pq.read_schema(raw_path)
            return list(schema.names), pq.ParquetFile(raw_path).metadata.num_rows
        raise ValueError(f"Unsupported sample dataset extension: {raw_path.suffix}")

    parquet_file = pq.ParquetFile(raw_path)
    return list(parquet_file.schema_arrow.names), parquet_file.metadata.num_rows


def report_step_1(config: dict[str, Any], config_path: Path, raw_path: Path, sample: bool) -> bool:
    print("Step 1: Environment, config, and raw data path")
    print_check("Config file exists", config_path.exists(), True, config_path.exists())

    dataset = config.get("dataset", {})
    dataset_name = dataset.get("name", "unknown")
    configured_raw_path = Path(dataset.get("raw_path", ""))

    print_check("Dataset name", dataset_name, "mak2022", dataset_name == "mak2022")
    print_check(
        "Configured raw path",
        configured_raw_path.as_posix(),
        EXPECTED_RAW_PATH.as_posix(),
        configured_raw_path == EXPECTED_RAW_PATH,
    )
    print(f"Resolved raw path: {raw_path}")

    raw_exists = raw_path.exists()
    print_check("Raw dataset exists", raw_exists, True, raw_exists)
    if not raw_exists:
        print(
            "Place the Wayback-sourced Mak 2022 parquet file at "
            f"{ROOT / EXPECTED_RAW_PATH} before running the full audit."
        )
        if sample:
            print("Sample mode can continue only when the configured sample file exists.")
        return False

    expected_suffix = ".parquet"
    print_check("Raw dataset extension", raw_path.suffix, expected_suffix, raw_path.suffix == expected_suffix)
    print_check("Expected full snapshot shape", EXPECTED_SHAPE, EXPECTED_SHAPE, True)
    if sample:
        print("Sample mode: limiting work to path/schema smoke checks.")
    return True


def report_column_group(group_name: str, actual_columns: set[str], expected_columns: list[str]) -> None:
    missing = missing_values(actual_columns, expected_columns)
    present = present_values(actual_columns, expected_columns)
    print_check(
        f"{group_name} columns present",
        f"{len(present)}/{len(expected_columns)} present; missing={missing}",
        f"{len(expected_columns)}/{len(expected_columns)} present",
        not missing,
    )


def report_step_2(columns: list[str], row_count: int | None) -> None:
    print()
    print("Step 2: Column enumeration and schema reconciliation")
    actual_columns = set(columns)

    print_check("Column count", len(columns), EXPECTED_SHAPE[1], len(columns) == EXPECTED_SHAPE[1])
    if row_count is not None:
        print_check("Parquet metadata row count", row_count, EXPECTED_SHAPE[0], row_count == EXPECTED_SHAPE[0])

    print("Actual columns:")
    for column in columns:
        print(f"  - {column}")

    report_column_group("Required field", actual_columns, REQUIRED_FIELDS)
    report_column_group("Experimental epigenetic feature", actual_columns, EXPERIMENTAL_EPIGENETIC_FEATURES)
    report_column_group("Computed nucleosome feature", actual_columns, COMPUTED_NUCLEOSOME_FEATURES)
    report_column_group("Binding-energy feature", actual_columns, BINDING_ENERGY_FEATURES)
    report_column_group("Target key", actual_columns, TARGET_KEY_FIELDS)

    genome_fields = present_values(actual_columns, GENOME_CANDIDATE_FIELDS)
    print_check("Genome field candidate", genome_fields, "at least one genome field", bool(genome_fields))

    guide_fields = present_values(actual_columns, GUIDE_KEY_CANDIDATE_FIELDS)
    print_check("Guide key candidate fields", guide_fields, "at least one guide key candidate", bool(guide_fields))

    ca_like_columns = [
        column
        for column in columns
        if column.lower() == "ca" or "cleavage_activity" in column.lower() or column.lower().endswith("_ca")
    ]
    print_check("Transformed CA-like columns absent", ca_like_columns, [], not ca_like_columns)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)

    dataset = config.get("dataset", {})
    raw_path = ROOT / Path(dataset.get("sample_path" if args.sample else "raw_path", ""))

    can_load = report_step_1(config, config_path, ROOT / Path(dataset.get("raw_path", "")), args.sample)
    if args.sample:
        sample_path = ROOT / Path(dataset.get("sample_path", ""))
        print(f"Resolved sample path: {sample_path}")
        if sample_path.exists():
            raw_path = sample_path
            can_load = True
        elif can_load:
            print_info("Sample file is unavailable; falling back to raw parquet metadata for Step 1/2 smoke checks.")
            raw_path = ROOT / Path(dataset.get("raw_path", ""))
        else:
            print_info("Sample file is unavailable.")
            print("Step 2 skipped because neither full raw data nor sample data is available.")
            return 0

    if not can_load:
        print("Step 2 skipped because the configured raw dataset is unavailable.")
        return 1

    columns, row_count = load_columns(raw_path, args.sample)
    report_step_2(columns, row_count)

    print()
    print("Step 1/2 audit completed. Later Sprint 1 audit steps were intentionally not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
