import pandas as pd

from crispr_gnn.data.splits import LABEL_COLUMN, SPLIT_COLUMN
from crispr_gnn.data.schemas import COMPUTED_NUCLEOSOME_FEATURES
from crispr_gnn.training.baselines import (
    BaselineRunConfig,
    MLPRunConfig,
    XGBoostRunConfig,
    run_dummy_and_logistic_baselines,
    run_tabular_mlp_baselines,
    run_xgboost_baselines,
)


def make_assigned_feature_rows() -> pd.DataFrame:
    valid_array = "[" + " ".join(str(value) for value in range(23)) + "]"
    rows = []
    for split_name, count in [("train", 12), ("val", 6), ("test", 6)]:
        for index in range(count):
            label = int(index % 2 == 0)
            row = {
                SPLIT_COLUMN: split_name,
                LABEL_COLUMN: label,
                "grna_target_id": f"{split_name}_g{index // 2}",
                "grna_target_sequence": "ACGTACGTACGTACGTACGTAGG",
                "target_sequence": "ACGTACGTACGTACGTACGTAGG" if label else "TCGTACGTACGTACGTACGTAGG",
                "energy_1": float(label),
                "energy_2": float(index),
                "energy_3": 0.0,
                "energy_4": 0.0,
                "energy_5": 0.0,
                "epigen_ctcf": float(label),
                "epigen_dnase": 0.1,
                "epigen_rrbs": 0.2,
                "epigen_h3k4me3": 0.3,
                "epigen_drip": 0.4,
                "MNase": 0.5,
            }
            for feature in COMPUTED_NUCLEOSOME_FEATURES:
                row[feature] = valid_array
            rows.append(row)
    return pd.DataFrame(rows)


def test_dummy_and_logistic_baselines_write_result_rows() -> None:
    results, predictions = run_dummy_and_logistic_baselines(
        assigned=make_assigned_feature_rows(),
        feature_sets=["F1", "F2"],
        config=BaselineRunConfig(sprint="sprint2", split_id="test_split", seed=7),
    )

    assert set(results["model_name"]) == {"dummy_prior", "logistic_regression"}
    assert set(results["feature_set"]) == {"F1", "F2"}
    assert results.shape[0] == 4
    assert len(predictions) == 8
    assert {prediction["split"] for prediction in predictions} == {"val", "test"}
    assert results["threshold_policy"].eq("validation_max_f1").all()
    assert results["test_rows"].eq(6).all()


def test_xgboost_baseline_writes_result_rows() -> None:
    results, predictions, feature_importance, feature_audit = run_xgboost_baselines(
        assigned=make_assigned_feature_rows(),
        feature_sets=["F1"],
        config=XGBoostRunConfig(
            sprint="sprint2",
            split_id="test_split",
            seed=7,
            n_estimators=10,
            max_depth=2,
            early_stopping_rounds=2,
            n_jobs=1,
        ),
        include_balanced=True,
    )

    assert set(results["model_name"]) == {"xgboost_unweighted", "xgboost_balanced_train_weights"}
    assert results["feature_set"].eq("F1").all()
    assert len(predictions) == 4
    assert not feature_importance.empty
    assert not feature_audit["is_forbidden"].any()


def test_tabular_mlp_baseline_writes_result_rows() -> None:
    results, predictions, training_summary, feature_audit = run_tabular_mlp_baselines(
        assigned=make_assigned_feature_rows(),
        feature_sets=["F1"],
        config=MLPRunConfig(
            sprint="sprint2",
            split_id="test_split",
            seed=7,
            hidden_dims=(8,),
            max_epochs=5,
            min_epochs=1,
            patience=2,
            batch_size=4,
        ),
        include_balanced=True,
        balanced_feature_sets=["F1"],
    )

    assert set(results["model_name"]) == {"tabular_mlp_unweighted", "tabular_mlp_balanced_train_weights"}
    assert results["feature_set"].eq("F1").all()
    assert len(predictions) == 4
    assert set(training_summary["model_name"]) == {"tabular_mlp_unweighted", "tabular_mlp_balanced_train_weights"}
    assert not feature_audit["is_forbidden"].any()
