from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from crispr_gnn.data.parsers import parse_numeric_array_result
from crispr_gnn.data.schemas import (
    BINDING_ENERGY_FEATURES,
    COMPUTED_NUCLEOSOME_FEATURES,
    EXPERIMENTAL_EPIGENETIC_FEATURES,
    EXPECTED_CELL_LINES,
    EXPECTED_CLEAVAGE_FREQ,
    EXPECTED_COMPUTED_FEATURE_MISSING_ROWS,
    EXPECTED_GENOME_NAMES,
    EXPECTED_GENOMES,
    EXPECTED_MEASURED_COUNTS,
    EXPECTED_MISSING_CELL_LINE,
    EXPECTED_RAW_PATH,
    EXPECTED_SHAPE,
    EXPECTED_THRESHOLDS,
    EXPECTED_UNIQUE_GUIDES,
    EXPECTED_UNIQUE_TARGETS,
    GENOME_CANDIDATE_FIELDS,
    GUIDE_KEY,
    GUIDE_KEY_CANDIDATE_FIELDS,
    REQUIRED_FIELDS,
    TARGET_KEY_FIELDS,
)


def marker(ok: bool) -> str:
    return "PASS" if ok else "DISCREPANCY"


def print_check(name: str, actual: Any, expected: Any, ok: bool) -> None:
    print(f"[{marker(ok)}] {name}: actual={actual} expected={expected}")


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_table(title: str, data: pd.Series | pd.DataFrame) -> None:
    print(title)
    if data.empty:
        print("  <empty>")
        return
    print(data.to_string())


def approx_equal(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def ratio_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "undefined"
    return f"{numerator / denominator:.2f}:1"


def missing_values(actual_columns: set[str], expected_columns: list[str]) -> list[str]:
    return [column for column in expected_columns if column not in actual_columns]


def present_values(actual_columns: set[str], expected_columns: list[str]) -> list[str]:
    return [column for column in expected_columns if column in actual_columns]


def get_ca_like_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.lower() == "ca" or "cleavage_activity" in column.lower() or column.lower().endswith("_ca")
    ]


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


def load_dataset(raw_path: Path) -> pd.DataFrame:
    print()
    print("Step 3: Dataset audit loading path")
    print(f"Loading full parquet with pandas: {raw_path}")
    df = pd.read_parquet(raw_path)
    print_check("Full dataset loaded", True, True, True)
    print_check("Loaded DataFrame shape", df.shape, EXPECTED_SHAPE, df.shape == EXPECTED_SHAPE)
    return df


def report_step_1(config: dict[str, Any], config_path: Path, raw_path: Path, sample: bool, project_root: Path) -> bool:
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
            f"{project_root / EXPECTED_RAW_PATH} before running the full audit."
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

    ca_like_columns = get_ca_like_columns(columns)
    print_check("Transformed CA-like columns absent", ca_like_columns, [], not ca_like_columns)


def report_step_4(df: pd.DataFrame) -> None:
    print()
    print("Step 4: High-level dataset audit metrics")

    print_check("Row count", len(df), EXPECTED_SHAPE[0], len(df) == EXPECTED_SHAPE[0])
    print_check("Column count", len(df.columns), EXPECTED_SHAPE[1], len(df.columns) == EXPECTED_SHAPE[1])

    measured_counts = df["measured"].value_counts(dropna=False).to_dict()
    measured_actual = {int(key): int(measured_counts.get(key, 0)) for key in EXPECTED_MEASURED_COUNTS}
    print_check(
        "Measured flag distribution",
        measured_actual,
        EXPECTED_MEASURED_COUNTS,
        measured_actual == EXPECTED_MEASURED_COUNTS,
    )

    guide_count = df[GUIDE_KEY].nunique(dropna=True)
    print_check(
        f"Unique sgRNAs using key '{GUIDE_KEY}'",
        guide_count,
        EXPECTED_UNIQUE_GUIDES,
        guide_count == EXPECTED_UNIQUE_GUIDES,
    )

    target_count = df[TARGET_KEY_FIELDS].drop_duplicates().shape[0]
    print_check(
        f"Unique target locations using key {TARGET_KEY_FIELDS}",
        target_count,
        EXPECTED_UNIQUE_TARGETS,
        target_count == EXPECTED_UNIQUE_TARGETS,
    )

    genome_counts = df["genome"].value_counts(dropna=False)
    print_table("Rows by genome:", genome_counts)
    genome_names = set(df["genome"].dropna().unique())
    print_check("Genome names", sorted(genome_names), sorted(EXPECTED_GENOME_NAMES), genome_names == EXPECTED_GENOME_NAMES)
    for genome, expected_count in EXPECTED_GENOMES.items():
        actual_count = int(genome_counts.get(genome, 0))
        print_check(
            f"Approximate rows for genome {genome}",
            actual_count,
            f"approximately {expected_count}",
            approx_equal(actual_count, expected_count, tolerance=2_500),
        )

    non_null_cell_lines = df["cell_line"].dropna().nunique()
    print_check(
        "Cell line count excluding missing",
        non_null_cell_lines,
        EXPECTED_CELL_LINES,
        non_null_cell_lines == EXPECTED_CELL_LINES,
    )
    missing_cell_line = int(df["cell_line"].isna().sum())
    print_check(
        "Missing cell_line rows",
        missing_cell_line,
        f"approximately {EXPECTED_MISSING_CELL_LINE}",
        approx_equal(missing_cell_line, EXPECTED_MISSING_CELL_LINE, tolerance=500),
    )
    print_table("Rows by cell_line, including missing:", df["cell_line"].fillna("<missing>").value_counts())

    experiment_counts = df["experiment_id"].value_counts(dropna=False).sort_index()
    print_table("Rows by experiment_id:", experiment_counts)
    if 18 in set(df["experiment_id"].dropna().astype(int)):
        experiment_18 = df[df["experiment_id"] == 18]
        missing_in_experiment_18 = int(experiment_18["cell_line"].isna().sum())
        print_check(
            "Missing cell_line rows concentrated in experiment_id=18",
            f"{missing_in_experiment_18}/{missing_cell_line} missing rows",
            "most missing cell_line rows in experiment_id=18",
            missing_cell_line > 0 and missing_in_experiment_18 / missing_cell_line >= 0.95,
        )

    per_guide_targets = df[[GUIDE_KEY, *TARGET_KEY_FIELDS]].drop_duplicates().groupby(GUIDE_KEY).size()
    distribution = per_guide_targets.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print_table(f"Unique target locations per sgRNA using guide key '{GUIDE_KEY}':", distribution)
    print_table("Top 10 sgRNAs by unique target locations:", per_guide_targets.sort_values(ascending=False).head(10))


def report_step_5(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    print()
    print("Step 5: cleavage_freq quality audit")

    print_check("cleavage_freq column exists", "cleavage_freq" in df.columns, True, "cleavage_freq" in df.columns)
    ca_like_columns = get_ca_like_columns(columns)
    print_check("Transformed CA-like columns absent", ca_like_columns, [], not ca_like_columns)

    cleavage = pd.to_numeric(df["cleavage_freq"], errors="coerce")
    minimum = float(cleavage.min(skipna=True))
    maximum = float(cleavage.max(skipna=True))
    nan_count = int(cleavage.isna().sum())
    negative_count = int((cleavage < 0).sum())
    zero_count = int((cleavage == 0).sum())
    above_one_count = int((cleavage > 1).sum())

    print_check(
        "cleavage_freq minimum",
        f"{minimum:.6g}",
        f"approximately {EXPECTED_CLEAVAGE_FREQ['min']}",
        approx_equal(minimum, EXPECTED_CLEAVAGE_FREQ["min"], tolerance=0.0002),
    )
    print_check(
        "cleavage_freq maximum",
        f"{maximum:.6g}",
        f"approximately {EXPECTED_CLEAVAGE_FREQ['max']}",
        approx_equal(maximum, EXPECTED_CLEAVAGE_FREQ["max"], tolerance=0.05),
    )
    print_check("cleavage_freq NaN count", nan_count, EXPECTED_CLEAVAGE_FREQ["nan"], nan_count == EXPECTED_CLEAVAGE_FREQ["nan"])
    print_check(
        "cleavage_freq negative count",
        negative_count,
        EXPECTED_CLEAVAGE_FREQ["negative"],
        negative_count == EXPECTED_CLEAVAGE_FREQ["negative"],
    )
    print_check("cleavage_freq zero count", zero_count, "reported for audit; no reference", True)
    print_check("cleavage_freq > 1 count", above_one_count, "reported for audit; no reference", True)

    buckets = {
        "(0, 1e-5]": int(((cleavage > 0) & (cleavage <= 1e-5)).sum()),
        "(1e-5, 1e-3]": int(((cleavage > 1e-5) & (cleavage <= 1e-3)).sum()),
        "(1e-3, 1]": int(((cleavage > 1e-3) & (cleavage <= 1)).sum()),
        ">1": above_one_count,
    }
    print_check("cleavage_freq threshold buckets", buckets, "reported for audit; no reference", True)
    print_info("Outlier handling policy recommendations are reported later in Step 7.")
    return cleavage


def report_threshold_distribution(df: pd.DataFrame, cleavage: pd.Series, label: str, threshold: float) -> None:
    label_eligible = cleavage.notna()
    eligible_count = int(label_eligible.sum())
    labels = cleavage[label_eligible] > threshold
    positives = int(labels.sum())
    negatives = eligible_count - positives
    positive_rate = positives / eligible_count
    expected = EXPECTED_THRESHOLDS[label]

    print_check(
        f"{label} positives at cleavage_freq > {threshold:g}",
        positives,
        f"approximately {expected['positives']}",
        approx_equal(positives, expected["positives"], tolerance=max(25, expected["positives"] * 0.01)),
    )
    imbalance = negatives / positives if positives else np.inf
    print_check(
        f"{label} imbalance ratio",
        ratio_text(negatives, positives),
        f"approximately {expected['imbalance']}:1",
        np.isfinite(imbalance) and approx_equal(
            imbalance,
            expected["imbalance"],
            tolerance=max(0.75, expected["imbalance"] * 0.05),
        ),
    )
    print(
        f"  label_eligible_total={eligible_count} excluded_nan={int(cleavage.isna().sum())} "
        f"positives={positives} negatives={negatives} positive_rate={positive_rate:.6f}"
    )

    measured = df["measured"]
    for measured_value in [1, 0]:
        mask = label_eligible & (measured == measured_value)
        subgroup_labels = cleavage[mask] > threshold
        subgroup_positive = int(subgroup_labels.sum())
        subgroup_negative = int(mask.sum()) - subgroup_positive
        print(
            f"  measured={measured_value}: positives={subgroup_positive} "
            f"negatives={subgroup_negative} label_eligible_total={int(mask.sum())}"
        )


def report_step_6(df: pd.DataFrame, cleavage: pd.Series, columns: list[str]) -> None:
    print()
    print("Step 6: Label distributions for Schemes A, B, and C")

    report_threshold_distribution(df, cleavage, "scheme_a", EXPECTED_THRESHOLDS["scheme_a"]["threshold"])
    report_threshold_distribution(df, cleavage, "scheme_c", EXPECTED_THRESHOLDS["scheme_c"]["threshold"])
    report_threshold_distribution(df, cleavage, "high_0.1", EXPECTED_THRESHOLDS["high_0.1"]["threshold"])

    ca_like_columns = get_ca_like_columns(columns)
    if ca_like_columns:
        print_check("Scheme B stored CA availability", ca_like_columns, "CA-like column found", True)
    else:
        print_check("Scheme B stored CA availability", ca_like_columns, "no stored CA-like column", True)
        print_info(
            "Scheme B is not directly computable from a stored column in this file. "
            "It requires reproducing the paper's per-study Box-Cox CA transform in a later step."
        )


def report_step_7(df: pd.DataFrame, cleavage: pd.Series) -> None:
    print()
    print("Step 7: Proposed audit policies for labels and noisy subsets")
    print_info("These policies are emitted for review. Docs/DECISIONS.md is intentionally not updated in this step.")

    nan_count = int(cleavage.isna().sum())
    negative_count = int((cleavage < 0).sum())
    above_one_count = int((cleavage > 1).sum())
    experiment_18_count = int((df["experiment_id"] == 18).sum())
    measured_zero_count = int((df["measured"] == 0).sum())
    non_human_count = int((df["genome"] != "hg19").sum())

    policies = {
        "NaN cleavage_freq": (
            nan_count,
            "Exclude from supervised binary train/validation/test label generation; do not silently impute as negative.",
        ),
        "negative cleavage_freq": (
            negative_count,
            "Treat as below-threshold for binary sensitivity counts, but flag as raw-label quality issue; "
            "do not use directly for regression/CA without a separate policy.",
        ),
        "cleavage_freq > 1": (
            above_one_count,
            "Treat as positive for binary threshold labels; do not clip for binary classification. "
            "Handle separately for regression/CA reproduction.",
        ),
        "measured=0 rows": (
            measured_zero_count,
            "Use only as optional training negatives with label-noise caveat; never use as test ground truth.",
        ),
        "experiment_id=18": (
            experiment_18_count,
            "Keep out of main evaluation or report as a separate sensitivity/no-cell-line subset.",
        ),
        "non-hg19 genomes": (
            non_human_count,
            "Do not drop by default; report per-genome breakdown and avoid human-only overclaims.",
        ),
        "Mak CA reproduction": (
            "not implemented",
            "Keep secondary/optional. Main track remains binary AUPRC; do not center the project on Mak reproduction.",
        ),
    }

    for name, (count, policy) in policies.items():
        print(f"- {name}: count={count}; policy={policy}")


def classify_computed_feature_arrays(series: pd.Series) -> dict[str, int]:
    counts = {"valid": 0, "missing": 0, "malformed_length": 0, "non_numeric": 0}
    for value in series:
        result = parse_numeric_array_result(value)
        counts[result.status] += 1
    return counts


def report_step_8(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    print()
    print("Step 8: Computed nucleosome feature parser validation")
    print_info("Parser requires exactly 23 numeric values; missing is tracked separately from malformed values.")

    feature_parse_counts: dict[str, dict[str, int]] = {}
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        counts = classify_computed_feature_arrays(df[feature])
        feature_parse_counts[feature] = counts
        print_check(
            f"{feature} parsed valid or missing only",
            counts,
            "nonzero valid count, zero malformed_length, zero non_numeric",
            counts["valid"] > 0 and counts["malformed_length"] == 0 and counts["non_numeric"] == 0,
        )

    rows_with_any_missing = int(df[COMPUTED_NUCLEOSOME_FEATURES].isna().any(axis=1).sum())
    print_check(
        "Rows missing at least one computed nucleosome feature",
        rows_with_any_missing,
        f"approximately {EXPECTED_COMPUTED_FEATURE_MISSING_ROWS}",
        approx_equal(rows_with_any_missing, EXPECTED_COMPUTED_FEATURE_MISSING_ROWS, tolerance=500),
    )
    print_check(
        "Computed nucleosome feature missingness rate",
        f"{rows_with_any_missing / len(df):.4%}",
        "approximately 4.9%",
        approx_equal(rows_with_any_missing / len(df), 0.049, tolerance=0.003),
    )
    return feature_parse_counts


def missingness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        missing = int(df[column].isna().sum())
        rows.append(
            {
                "feature": column,
                "missing": missing,
                "missing_pct": missing / len(df),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def print_missingness_breakdown(df: pd.DataFrame, columns: list[str], title: str) -> None:
    missing_any = df[columns].isna().any(axis=1)
    print_table(f"{title} missing-any rows by measured:", df.loc[missing_any, "measured"].value_counts(dropna=False))
    print_table(f"{title} missing-any rows by genome:", df.loc[missing_any, "genome"].value_counts(dropna=False))
    print_table(
        f"{title} missing-any rows by experiment_id:",
        df.loc[missing_any, "experiment_id"].value_counts(dropna=False).sort_index(),
    )


def report_step_9(df: pd.DataFrame, feature_parse_counts: dict[str, dict[str, int]]) -> None:
    print()
    print("Step 9: Epigenetic and binding-energy feature missingness")

    experimental_missingness = missingness_table(df, EXPERIMENTAL_EPIGENETIC_FEATURES)
    computed_missingness = missingness_table(df, COMPUTED_NUCLEOSOME_FEATURES)
    energy_missingness = missingness_table(df, BINDING_ENERGY_FEATURES)

    print_table("Experimental epigenetic feature missingness:", experimental_missingness)
    print_missingness_breakdown(df, EXPERIMENTAL_EPIGENETIC_FEATURES, "Experimental epigenetic features")

    print_table("Computed nucleosome raw-field missingness:", computed_missingness)
    malformed_summary = pd.DataFrame(feature_parse_counts).T[["missing", "malformed_length", "non_numeric", "valid"]]
    print_table("Computed nucleosome parser status counts:", malformed_summary)
    print_missingness_breakdown(df, COMPUTED_NUCLEOSOME_FEATURES, "Computed nucleosome features")

    print_table("Binding-energy feature missingness:", energy_missingness)
    print_missingness_breakdown(df, BINDING_ENERGY_FEATURES, "Binding-energy features")

    all_computed_malformed = int(malformed_summary["malformed_length"].sum() + malformed_summary["non_numeric"].sum())
    print_check("Computed feature malformed array count", all_computed_malformed, 0, all_computed_malformed == 0)
    print_info("No imputation or exclusion is applied in this step; missingness policy remains a later modeling decision.")


def label_counts_for_measured_rows(df: pd.DataFrame, cleavage: pd.Series) -> pd.DataFrame:
    measured_mask = df["measured"] == 1
    label_eligible_mask = measured_mask & cleavage.notna()
    eligible_count = int(label_eligible_mask.sum())
    rows = []
    for name, config in EXPECTED_THRESHOLDS.items():
        threshold = config["threshold"]
        labels = cleavage[label_eligible_mask] > threshold
        positives = int(labels.sum())
        negatives = eligible_count - positives
        rows.append(
            {
                "scheme": name,
                "threshold": threshold,
                "measured_rows": int(measured_mask.sum()),
                "label_eligible_rows": eligible_count,
                "positives": positives,
                "negatives": negatives,
                "positive_rate": positives / eligible_count if eligible_count else np.nan,
                "imbalance": ratio_text(negatives, positives),
            }
        )
    return pd.DataFrame(rows).set_index("scheme")


def report_step_10(df: pd.DataFrame, cleavage: pd.Series) -> None:
    print()
    print("Step 10: Split implications and measured=1 rule")
    print_info("Final splits are intentionally not built in this step.")

    print("- Final test set MUST contain measured=1 rows only.")
    print("- Validation SHOULD prefer measured=1 rows.")
    print("- measured=0 rows MAY be used as training negatives only with a label-noise caveat.")
    print("- Random edge split is debug-only; final evaluation should use guide-level split.")
    print("- Feature normalization/statistics must be fit on train data only in later modeling steps.")
    print("- Similarity/context edges must not use labels when graph construction begins.")

    measured_label_counts = label_counts_for_measured_rows(df, cleavage)
    print_table("Measured-only label distributions for future split design:", measured_label_counts)

    per_guide_rows = df.groupby(GUIDE_KEY).size().sort_values(ascending=False)
    measured_per_guide = df[df["measured"] == 1].groupby(GUIDE_KEY).size().reindex(per_guide_rows.index, fill_value=0)
    guide_split_risk = pd.DataFrame(
        {
            "total_rows": per_guide_rows,
            "measured_rows": measured_per_guide,
        }
    ).head(10)
    print_table("Top 10 guide-level split concentration risks:", guide_split_risk)
    print_info(
        "Guide-level split should stratify or otherwise account for very large guides so one guide does not dominate "
        "validation/test composition."
    )
