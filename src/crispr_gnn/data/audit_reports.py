from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crispr_gnn.data.schemas import (
    BINDING_ENERGY_FEATURES,
    COMPUTED_NUCLEOSOME_FEATURES,
    EXPERIMENTAL_EPIGENETIC_FEATURES,
    EXPECTED_CELL_LINES,
    EXPECTED_CLEAVAGE_FREQ,
    EXPECTED_COMPUTED_FEATURE_MISSING_ROWS,
    EXPECTED_GENOME_NAMES,
    EXPECTED_MEASURED_COUNTS,
    EXPECTED_MISSING_CELL_LINE,
    EXPECTED_SHAPE,
    EXPECTED_THRESHOLDS,
    EXPECTED_UNIQUE_GUIDES,
    EXPECTED_UNIQUE_TARGETS,
    GUIDE_KEY,
    TARGET_KEY_FIELDS,
)


def marker(ok: bool) -> str:
    return "PASS" if ok else "DISCREPANCY"


def check_row(name: str, actual: Any, expected: Any, ok: bool) -> dict[str, str]:
    return {
        "check": name,
        "actual": str(actual),
        "expected": str(expected),
        "status": marker(ok),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def approx_equal(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def ratio_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "undefined"
    return f"{numerator / denominator:.2f}:1"


def get_ca_like_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.lower() == "ca" or "cleavage_activity" in column.lower() or column.lower().endswith("_ca")
    ]


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


def threshold_report_rows(df: pd.DataFrame, cleavage: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_eligible = cleavage.notna()
    eligible_count = int(label_eligible.sum())
    for name, config in EXPECTED_THRESHOLDS.items():
        threshold = config["threshold"]
        labels = cleavage[label_eligible] > threshold
        positives = int(labels.sum())
        negatives = eligible_count - positives
        measured_eligible = label_eligible & (df["measured"] == 1)
        putative_eligible = label_eligible & (df["measured"] == 0)
        measured_labels = cleavage[measured_eligible] > threshold
        putative_labels = cleavage[putative_eligible] > threshold
        measured_positive = int(measured_labels.sum())
        measured_negative = int(measured_eligible.sum()) - measured_positive
        putative_positive = int(putative_labels.sum())
        putative_negative = int(putative_eligible.sum()) - putative_positive
        imbalance = negatives / positives if positives else np.inf
        expected_imbalance = config["imbalance"]
        positives_ok = approx_equal(positives, config["positives"], tolerance=max(25, config["positives"] * 0.01))
        imbalance_ok = np.isfinite(imbalance) and approx_equal(
            imbalance,
            expected_imbalance,
            tolerance=max(0.75, expected_imbalance * 0.05),
        )
        rows.append(
            {
                "scheme": name,
                "threshold": f"{threshold:g}",
                "label_eligible_rows": eligible_count,
                "positives": positives,
                "negatives": negatives,
                "positive_rate": f"{positives / eligible_count:.6f}",
                "imbalance": ratio_text(negatives, positives),
                "measured_1_pos_neg": f"{measured_positive}/{measured_negative}",
                "measured_0_pos_neg": f"{putative_positive}/{putative_negative}",
                "expected": f"{config['positives']} positives; ~{expected_imbalance}:1",
                "status": marker(positives_ok and imbalance_ok),
            }
        )
    return rows


def make_dataset_report(df: pd.DataFrame, columns: list[str], cleavage: pd.Series) -> list[str]:
    measured_counts = df["measured"].value_counts(dropna=False).to_dict()
    measured_actual = {int(key): int(measured_counts.get(key, 0)) for key in EXPECTED_MEASURED_COUNTS}
    guide_count = df[GUIDE_KEY].nunique(dropna=True)
    target_count = df[TARGET_KEY_FIELDS].drop_duplicates().shape[0]
    genome_counts = df["genome"].value_counts(dropna=False)
    genome_names = set(df["genome"].dropna().unique())
    missing_cell_line = int(df["cell_line"].isna().sum())
    experiment_18 = df[df["experiment_id"] == 18]
    missing_in_experiment_18 = int(experiment_18["cell_line"].isna().sum())
    computed_missing_rows = int(df[COMPUTED_NUCLEOSOME_FEATURES].isna().any(axis=1).sum())
    ca_like_columns = get_ca_like_columns(columns)

    summary_rows = [
        check_row("Shape", df.shape, EXPECTED_SHAPE, df.shape == EXPECTED_SHAPE),
        check_row("Measured distribution", measured_actual, EXPECTED_MEASURED_COUNTS, measured_actual == EXPECTED_MEASURED_COUNTS),
        check_row(f"Unique sgRNAs using {GUIDE_KEY}", guide_count, EXPECTED_UNIQUE_GUIDES, guide_count == EXPECTED_UNIQUE_GUIDES),
        check_row(
            f"Unique target locations using {TARGET_KEY_FIELDS}",
            target_count,
            EXPECTED_UNIQUE_TARGETS,
            target_count == EXPECTED_UNIQUE_TARGETS,
        ),
        check_row("Genome names", sorted(genome_names), sorted(EXPECTED_GENOME_NAMES), genome_names == EXPECTED_GENOME_NAMES),
        check_row(
            "Cell line count excluding missing",
            df["cell_line"].dropna().nunique(),
            EXPECTED_CELL_LINES,
            df["cell_line"].dropna().nunique() == EXPECTED_CELL_LINES,
        ),
        check_row(
            "Missing cell_line rows",
            missing_cell_line,
            f"approximately {EXPECTED_MISSING_CELL_LINE}",
            approx_equal(missing_cell_line, EXPECTED_MISSING_CELL_LINE, tolerance=500),
        ),
        check_row(
            "Missing cell_line rows in experiment_id=18",
            f"{missing_in_experiment_18}/{missing_cell_line}",
            ">=95% of missing cell_line rows",
            missing_cell_line > 0 and missing_in_experiment_18 / missing_cell_line >= 0.95,
        ),
        check_row(
            "Rows missing at least one computed nucleosome feature",
            computed_missing_rows,
            f"approximately {EXPECTED_COMPUTED_FEATURE_MISSING_ROWS}",
            approx_equal(computed_missing_rows, EXPECTED_COMPUTED_FEATURE_MISSING_ROWS, tolerance=500),
        ),
        check_row("Transformed CA-like columns absent", ca_like_columns, [], not ca_like_columns),
    ]

    cleavage_rows = [
        check_row(
            "cleavage_freq minimum",
            f"{float(cleavage.min(skipna=True)):.6g}",
            f"approximately {EXPECTED_CLEAVAGE_FREQ['min']}",
            approx_equal(float(cleavage.min(skipna=True)), EXPECTED_CLEAVAGE_FREQ["min"], tolerance=0.0002),
        ),
        check_row(
            "cleavage_freq maximum",
            f"{float(cleavage.max(skipna=True)):.6g}",
            f"approximately {EXPECTED_CLEAVAGE_FREQ['max']}",
            approx_equal(float(cleavage.max(skipna=True)), EXPECTED_CLEAVAGE_FREQ["max"], tolerance=0.05),
        ),
        check_row("cleavage_freq NaN count", int(cleavage.isna().sum()), EXPECTED_CLEAVAGE_FREQ["nan"], int(cleavage.isna().sum()) == EXPECTED_CLEAVAGE_FREQ["nan"]),
        check_row(
            "cleavage_freq negative count",
            int((cleavage < 0).sum()),
            EXPECTED_CLEAVAGE_FREQ["negative"],
            int((cleavage < 0).sum()) == EXPECTED_CLEAVAGE_FREQ["negative"],
        ),
        check_row("cleavage_freq zero count", int((cleavage == 0).sum()), "reported for audit; no reference", True),
        check_row("cleavage_freq > 1 count", int((cleavage > 1).sum()), "reported for audit; no reference", True),
    ]

    per_guide_targets = df[[GUIDE_KEY, *TARGET_KEY_FIELDS]].drop_duplicates().groupby(GUIDE_KEY).size()
    guide_stats = per_guide_targets.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    top_guides = per_guide_targets.sort_values(ascending=False).head(10)

    return [
        "# Dataset Audit Report",
        "",
        "Generated by `scripts/audit_dataset.py` from the local Mak 2022 parquet snapshot.",
        "",
        "## Source",
        "",
        "- Local raw path: `data/raw/260520_putative_nucleosomal.parquet`.",
        "- Original dataset URL: `https://crisprsql.com/downloads/260520_putative_nucleosomal.parquet.gz`.",
        "- Project source method: Internet Archive Wayback snapshot of the original Mak 2022 URL.",
        "- Main modeling track remains binary off-target classification; Mak CA reproduction is later-only paper comparison.",
        "",
        "## Reference Checks",
        "",
        *markdown_table(summary_rows, ["check", "actual", "expected", "status"]),
        "",
        "## Genome Counts",
        "",
        *markdown_table(
            [{"genome": str(index), "rows": int(value)} for index, value in genome_counts.items()],
            ["genome", "rows"],
        ),
        "",
        "## Cell Line Counts",
        "",
        *markdown_table(
            [
                {"cell_line": str(index), "rows": int(value)}
                for index, value in df["cell_line"].fillna("<missing>").value_counts().items()
            ],
            ["cell_line", "rows"],
        ),
        "",
        "## Cleavage Frequency Quality",
        "",
        *markdown_table(cleavage_rows, ["check", "actual", "expected", "status"]),
        "",
        "## Guide-Level Split Risk",
        "",
        "Unique target locations per sgRNA are highly uneven; later guide-level split code should account for large guides.",
        "",
        *markdown_table(
            [{"stat": str(index), "value": f"{value:.3f}"} for index, value in guide_stats.items()],
            ["stat", "value"],
        ),
        "",
        "### Top Guides By Unique Target Locations",
        "",
        *markdown_table(
            [{"guide": str(index), "unique_target_locations": int(value)} for index, value in top_guides.items()],
            ["guide", "unique_target_locations"],
        ),
        "",
        "## Split Implications",
        "",
        "- Final test set must contain `measured=1` rows only.",
        "- Validation should prefer `measured=1` rows.",
        "- `measured=0` rows may be used only as optional training negatives with label-noise caveat.",
        "- Random edge split remains debug-only; final evaluation should use guide-level split and AUPRC.",
    ]


def make_label_report(df: pd.DataFrame, cleavage: pd.Series, columns: list[str]) -> list[str]:
    ca_like_columns = get_ca_like_columns(columns)
    measured_rows = label_counts_for_measured_rows(df, cleavage).reset_index()
    measured_rows["threshold"] = measured_rows["threshold"].map(lambda value: f"{value:g}")
    measured_rows["positive_rate"] = measured_rows["positive_rate"].map(lambda value: f"{value:.6f}")

    policy_rows = [
        {
            "case": "NaN cleavage_freq",
            "count": int(cleavage.isna().sum()),
            "policy": "Exclude from supervised binary train/validation/test label generation; do not silently impute as negative.",
        },
        {
            "case": "negative cleavage_freq",
            "count": int((cleavage < 0).sum()),
            "policy": "Below-threshold for binary sensitivity counts; flag as raw-label quality issue.",
        },
        {
            "case": "cleavage_freq > 1",
            "count": int((cleavage > 1).sum()),
            "policy": "Positive for binary thresholds; do not clip for binary classification.",
        },
        {
            "case": "measured=0 rows",
            "count": int((df["measured"] == 0).sum()),
            "policy": "Training-only optional noisy negatives; never test ground truth.",
        },
        {
            "case": "experiment_id=18",
            "count": int((df["experiment_id"] == 18).sum()),
            "policy": "Keep out of main evaluation or report as separate sensitivity/no-cell-line subset.",
        },
        {
            "case": "non-hg19 genomes",
            "count": int((df["genome"] != "hg19").sum()),
            "policy": "Do not drop by default; report per-genome breakdown and avoid human-only overclaims.",
        },
    ]

    return [
        "# Label Threshold Sensitivity",
        "",
        "Generated by `scripts/audit_dataset.py` from raw `cleavage_freq`.",
        "",
        "Rows with NaN `cleavage_freq` are label-ineligible and are excluded from threshold label counts.",
        "",
        "## Label Schemes",
        "",
        "- Scheme A: `cleavage_freq > 1e-5`; primary binary track.",
        "- Scheme B: Mak paper-comparison track only; CA/Box-Cox reproduction is deferred and not central.",
        "- Scheme C: `cleavage_freq > 1e-3`; later robustness sensitivity.",
        "- High threshold: `cleavage_freq > 0.1`; audit-only sensitivity.",
        "",
        "## Whole-Dataset Threshold Table",
        "",
        *markdown_table(
            threshold_report_rows(df, cleavage),
            [
                "scheme",
                "threshold",
                "label_eligible_rows",
                "positives",
                "negatives",
                "positive_rate",
                "imbalance",
                "measured_1_pos_neg",
                "measured_0_pos_neg",
                "expected",
                "status",
            ],
        ),
        "",
        "## Measured-Only Threshold Table",
        "",
        *markdown_table(
            measured_rows.to_dict(orient="records"),
            [
                "scheme",
                "threshold",
                "measured_rows",
                "label_eligible_rows",
                "positives",
                "negatives",
                "positive_rate",
                "imbalance",
            ],
        ),
        "",
        "## Scheme B Status",
        "",
        f"- CA-like columns found: `{ca_like_columns}`.",
        "- Stored CA is absent if the list above is empty.",
        "- Reproducing Mak Scheme B requires per-study Box-Cox transformation, standardization, and clipping.",
        "- This project does not center on Mak CA reproduction; binary guide-level AUPRC remains the main track.",
        "",
        "## Outlier And Subset Policies",
        "",
        *markdown_table(policy_rows, ["case", "count", "policy"]),
    ]


def make_feature_missingness_report(
    df: pd.DataFrame,
    feature_parse_counts: dict[str, dict[str, int]],
) -> list[str]:
    feature_group_rows = []
    for group_name, features in [
        ("experimental_epigenetic", EXPERIMENTAL_EPIGENETIC_FEATURES),
        ("computed_nucleosome", COMPUTED_NUCLEOSOME_FEATURES),
        ("binding_energy", BINDING_ENERGY_FEATURES),
    ]:
        missing_any = int(df[features].isna().any(axis=1).sum())
        if group_name == "computed_nucleosome":
            expected = f"approximately {EXPECTED_COMPUTED_FEATURE_MISSING_ROWS}"
            status = marker(approx_equal(missing_any, EXPECTED_COMPUTED_FEATURE_MISSING_ROWS, tolerance=500))
        else:
            expected = "0"
            status = marker(missing_any == 0)
        feature_group_rows.append(
            {
                "group": group_name,
                "features": len(features),
                "rows_missing_any": missing_any,
                "missing_any_pct": f"{missing_any / len(df):.4%}",
                "expected_rows_missing_any": expected,
                "status": status,
            }
        )

    feature_rows = []
    for group_name, features in [
        ("experimental_epigenetic", EXPERIMENTAL_EPIGENETIC_FEATURES),
        ("computed_nucleosome", COMPUTED_NUCLEOSOME_FEATURES),
        ("binding_energy", BINDING_ENERGY_FEATURES),
    ]:
        for feature in features:
            missing = int(df[feature].isna().sum())
            row = {
                "group": group_name,
                "feature": feature,
                "missing": missing,
                "missing_pct": f"{missing / len(df):.4%}",
                "valid": "",
                "malformed_length": "",
                "non_numeric": "",
                "status": "PASS",
            }
            if feature in feature_parse_counts:
                counts = feature_parse_counts[feature]
                row.update(
                    {
                        "valid": counts["valid"],
                        "malformed_length": counts["malformed_length"],
                        "non_numeric": counts["non_numeric"],
                        "status": marker(counts["malformed_length"] == 0 and counts["non_numeric"] == 0),
                    }
                )
            feature_rows.append(row)

    computed_missing_by_experiment = (
        df.loc[df[COMPUTED_NUCLEOSOME_FEATURES].isna().any(axis=1), "experiment_id"].value_counts().sort_index()
    )

    return [
        "# Feature Missingness Report",
        "",
        "Generated by `scripts/audit_dataset.py`.",
        "",
        "## Parser Policy",
        "",
        "- Computed nucleosome arrays must contain exactly 23 numeric values.",
        "- Missing values are tracked separately from malformed arrays.",
        "- No silent padding, clipping, coercion, or imputation is applied in Sprint 1.",
        "",
        "## Feature Group Summary",
        "",
        *markdown_table(
            feature_group_rows,
            ["group", "features", "rows_missing_any", "missing_any_pct", "expected_rows_missing_any", "status"],
        ),
        "",
        "## Per-Feature Missingness And Parser Status",
        "",
        *markdown_table(
            feature_rows,
            ["group", "feature", "missing", "missing_pct", "valid", "malformed_length", "non_numeric", "status"],
        ),
        "",
        "## Computed Feature Missingness By Experiment",
        "",
        *markdown_table(
            [
                {"experiment_id": str(index), "rows_missing_any_computed_feature": int(value)}
                for index, value in computed_missing_by_experiment.items()
            ],
            ["experiment_id", "rows_missing_any_computed_feature"],
        ),
        "",
        "## Modeling Implications",
        "",
        "- Experimental epigenetic and binding-energy features are complete in this snapshot.",
        "- Computed nucleosome features have a shared missingness pattern affecting about 4.9% of rows.",
        "- Later feature builders must choose an explicit missingness policy before using computed features.",
    ]


def generate_reports(
    df: pd.DataFrame,
    columns: list[str],
    cleavage: pd.Series,
    feature_parse_counts: dict[str, dict[str, int]],
    report_dir: Path,
) -> list[Path]:
    reports = {
        "dataset_audit.md": make_dataset_report(df, columns, cleavage),
        "label_threshold_sensitivity.md": make_label_report(df, cleavage, columns),
        "feature_missingness.md": make_feature_missingness_report(df, feature_parse_counts),
    }
    written_paths = []
    for filename, lines in reports.items():
        path = report_dir / filename
        write_markdown(path, lines)
        written_paths.append(path)
    return written_paths
