import numpy as np
import pandas as pd
import pytest

from crispr_gnn.features.tabular import (
    FORBIDDEN_PREDICTIVE_COLUMNS,
    RAW_ID_COLUMNS,
    TrainOnlyPreprocessor,
    audit_feature_columns,
    build_computed_nucleosome_features,
    build_feature_set,
)
from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES


def make_feature_df() -> pd.DataFrame:
    valid_array = "[" + " ".join(str(value) for value in range(23)) + "]"
    row = {
        "id": 1,
        "experiment_id": 1,
        "grna_target_id": "guide-a",
        "genome": "hg19",
        "cell_line": "K562",
        "grna_target_sequence": "ACGTACGTACGTACGTACGTAGG",
        "target_sequence": "ACGTACGTTCGTACGTACGTGGG",
        "energy_1": -1.0,
        "energy_2": -2.0,
        "energy_3": -3.0,
        "energy_4": -4.0,
        "energy_5": -5.0,
        "epigen_ctcf": 0.1,
        "epigen_dnase": 0.2,
        "epigen_rrbs": 0.3,
        "epigen_h3k4me3": 0.4,
        "epigen_drip": 0.5,
        "MNase": 0.6,
    }
    for feature in COMPUTED_NUCLEOSOME_FEATURES:
        row[feature] = valid_array
    missing_row = row.copy()
    missing_row["id"] = 2
    missing_row["target_sequence"] = "ACGTACGTACGTACGTACGTAGG"
    missing_row[COMPUTED_NUCLEOSOME_FEATURES[0]] = None
    return pd.DataFrame([row, missing_row])


def test_feature_ladder_increases_columns_without_raw_id_leakage() -> None:
    df = make_feature_df()
    f1 = build_feature_set(df, "F1")
    f2 = build_feature_set(df, "F2")
    f3 = build_feature_set(df, "F3")
    f4 = build_feature_set(df, "F4")

    assert f1.shape[1] < f2.shape[1] < f3.shape[1] < f4.shape[1]
    assert set(f4.columns).isdisjoint(RAW_ID_COLUMNS)
    assert "mismatch_count" in f1.columns
    assert f1.loc[0, "mismatch_count"] == 2
    assert set(f4.columns).isdisjoint(FORBIDDEN_PREDICTIVE_COLUMNS)


def test_feature_column_audit_flags_forbidden_features() -> None:
    audit = audit_feature_columns("F1", ["mismatch_count", "genome", "cleavage_freq"])

    flagged = set(audit.loc[audit["is_forbidden"], "feature"])
    assert flagged == {"genome", "cleavage_freq"}


def test_computed_features_add_missingness_indicators() -> None:
    features = build_computed_nucleosome_features(make_feature_df())

    missing_col = f"{COMPUTED_NUCLEOSOME_FEATURES[0]}_missing"
    mean_col = f"{COMPUTED_NUCLEOSOME_FEATURES[0]}_mean"
    assert features.loc[0, missing_col] == 0.0
    assert features.loc[1, missing_col] == 1.0
    assert np.isnan(features.loc[1, mean_col])


def test_train_only_preprocessor_preserves_fit_columns_and_rejects_schema_drift() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [0.0, 1.0, 2.0]})
    val = pd.DataFrame({"a": [np.nan], "b": [3.0]})
    preprocessor = TrainOnlyPreprocessor(scale=True).fit(train)

    transformed = preprocessor.transform(val)
    assert list(transformed.columns) == ["a", "b"]
    assert not transformed.isna().any().any()

    with pytest.raises(ValueError, match="Feature columns differ"):
        preprocessor.transform(pd.DataFrame({"a": [1.0], "c": [2.0]}))
