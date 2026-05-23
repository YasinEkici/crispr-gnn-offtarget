"""Measured-only Sprint 2 baseline training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from crispr_gnn.data.splits import LABEL_COLUMN, SPLIT_COLUMN
from crispr_gnn.evaluation.metrics import binary_classification_metrics, select_threshold_by_f1
from crispr_gnn.features.tabular import FeatureSetName, TrainOnlyPreprocessor, audit_feature_columns, build_feature_set, feature_family


@dataclass(frozen=True)
class BaselineRunConfig:
    sprint: str
    split_id: str
    seed: int
    label_scheme: str = "scheme_a"
    training_regime: str = "measured_only"
    logistic_max_iter: int = 2_000
    logistic_class_weight: str | None = "balanced"


@dataclass(frozen=True)
class XGBoostRunConfig:
    sprint: str
    split_id: str
    seed: int
    label_scheme: str = "scheme_a"
    training_regime: str = "measured_only"
    n_estimators: int = 400
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 5.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 30
    eval_metric: str = "aucpr"
    tree_method: str = "hist"
    n_jobs: int = 4


def run_dummy_and_logistic_baselines(
    assigned: pd.DataFrame,
    feature_sets: list[FeatureSetName],
    config: BaselineRunConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for feature_set in feature_sets:
        split_data = _prepared_feature_split(assigned, feature_set)
        for model_name in ["dummy_prior", "logistic_regression"]:
            model = _fit_model(model_name, split_data, config)
            val_scores = _positive_class_scores(model, split_data["X_val"])
            test_scores = _positive_class_scores(model, split_data["X_test"])
            selection = select_threshold_by_f1(split_data["y_val"], val_scores)
            row = _result_row(
                model_name=model_name,
                feature_set=feature_set,
                config=config,
                split_data=split_data,
                val_scores=val_scores,
                test_scores=test_scores,
                threshold=selection.threshold,
                threshold_policy=selection.policy,
            )
            rows.append(row)
            predictions.extend(_prediction_records(model_name, feature_set, split_data, val_scores, test_scores))
    return pd.DataFrame(rows), predictions


def run_xgboost_baselines(
    assigned: pd.DataFrame,
    feature_sets: list[FeatureSetName],
    config: XGBoostRunConfig,
    *,
    include_balanced: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    feature_importance_rows: list[dict[str, object]] = []
    feature_audit_frames: list[pd.DataFrame] = []
    variants = ["xgboost_unweighted"]
    if include_balanced:
        variants.append("xgboost_balanced_train_weights")

    for feature_set in feature_sets:
        split_data = _prepared_feature_split(assigned, feature_set, scale=False)
        feature_audit_frames.append(audit_feature_columns(feature_set, split_data["feature_columns"]))
        for model_name in variants:
            model = _fit_xgboost_model(model_name, split_data, config)
            feature_importance_rows.extend(_xgboost_feature_importance_rows(model, model_name, feature_set, split_data["feature_columns"]))
            val_scores = _positive_class_scores(model, split_data["X_val"])
            test_scores = _positive_class_scores(model, split_data["X_test"])
            selection = select_threshold_by_f1(split_data["y_val"], val_scores)
            row = _result_row(
                model_name=model_name,
                feature_set=feature_set,
                config=config,
                split_data=split_data,
                val_scores=val_scores,
                test_scores=test_scores,
                threshold=selection.threshold,
                threshold_policy=selection.policy,
                notes=f"{model_name}; measured-only main split; experiment_id=18 excluded",
            )
            rows.append(row)
            predictions.extend(_prediction_records(model_name, feature_set, split_data, val_scores, test_scores))
    feature_audit = pd.concat(feature_audit_frames, axis=0, ignore_index=True)
    return pd.DataFrame(rows), predictions, pd.DataFrame(feature_importance_rows), feature_audit


def _prepared_feature_split(assigned: pd.DataFrame, feature_set: FeatureSetName, *, scale: bool = True) -> dict[str, Any]:
    features = build_feature_set(assigned, feature_set)
    data = pd.concat([assigned[[SPLIT_COLUMN, LABEL_COLUMN]], features], axis=1)
    train = data.loc[data[SPLIT_COLUMN] == "train"]
    val = data.loc[data[SPLIT_COLUMN] == "val"]
    test = data.loc[data[SPLIT_COLUMN] == "test"]
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all be non-empty")

    feature_columns = list(features.columns)
    preprocessor = TrainOnlyPreprocessor(scale=scale).fit(train[feature_columns])
    return {
        "X_train": preprocessor.transform(train[feature_columns]).to_numpy(),
        "X_val": preprocessor.transform(val[feature_columns]).to_numpy(),
        "X_test": preprocessor.transform(test[feature_columns]).to_numpy(),
        "y_train": train[LABEL_COLUMN].to_numpy(dtype=int),
        "y_val": val[LABEL_COLUMN].to_numpy(dtype=int),
        "y_test": test[LABEL_COLUMN].to_numpy(dtype=int),
        "train_index": train.index.to_numpy(),
        "val_index": val.index.to_numpy(),
        "test_index": test.index.to_numpy(),
        "feature_columns": feature_columns,
    }


def _xgboost_feature_importance_rows(
    model: XGBClassifier,
    model_name: str,
    feature_set: FeatureSetName,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    booster = model.get_booster()
    scores_by_type = {
        importance_type: booster.get_score(importance_type=importance_type)
        for importance_type in ["weight", "gain", "cover", "total_gain", "total_cover"]
    }
    rows = []
    for index, feature in enumerate(feature_columns):
        xgb_name = f"f{index}"
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "feature": feature,
                "family": feature_family(feature),
                "weight": float(scores_by_type["weight"].get(xgb_name, 0.0)),
                "gain": float(scores_by_type["gain"].get(xgb_name, 0.0)),
                "cover": float(scores_by_type["cover"].get(xgb_name, 0.0)),
                "total_gain": float(scores_by_type["total_gain"].get(xgb_name, 0.0)),
                "total_cover": float(scores_by_type["total_cover"].get(xgb_name, 0.0)),
            }
        )
    return rows


def _fit_model(model_name: str, split_data: dict[str, Any], config: BaselineRunConfig) -> object:
    if model_name == "dummy_prior":
        model = DummyClassifier(strategy="prior", random_state=config.seed)
    elif model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=config.logistic_max_iter,
            class_weight=config.logistic_class_weight,
            solver="lbfgs",
            random_state=config.seed,
        )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")
    model.fit(split_data["X_train"], split_data["y_train"])
    return model


def _fit_xgboost_model(model_name: str, split_data: dict[str, Any], config: XGBoostRunConfig) -> XGBClassifier:
    if model_name == "xgboost_unweighted":
        sample_weight = None
    elif model_name == "xgboost_balanced_train_weights":
        sample_weight = compute_sample_weight(class_weight="balanced", y=split_data["y_train"])
    else:
        raise ValueError(f"Unsupported XGBoost model variant: {model_name}")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        early_stopping_rounds=config.early_stopping_rounds,
        eval_metric=config.eval_metric,
        tree_method=config.tree_method,
        n_jobs=config.n_jobs,
        random_state=config.seed,
    )
    model.fit(
        split_data["X_train"],
        split_data["y_train"],
        sample_weight=sample_weight,
        eval_set=[(split_data["X_val"], split_data["y_val"])],
        verbose=False,
    )
    return model


def _positive_class_scores(model: object, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    matches = np.where(classes == 1)[0]
    if matches.shape[0] != 1:
        raise ValueError("Model does not expose exactly one positive class probability")
    positive_index = int(matches[0])
    return model.predict_proba(features)[:, positive_index]


def _result_row(
    *,
    model_name: str,
    feature_set: FeatureSetName,
    config: BaselineRunConfig | XGBoostRunConfig,
    split_data: dict[str, Any],
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    threshold_policy: str,
    notes: str = "measured-only main split; experiment_id=18 excluded",
) -> dict[str, object]:
    val_metrics = binary_classification_metrics(split_data["y_val"], val_scores, threshold, prefix="val_")
    test_metrics = binary_classification_metrics(split_data["y_test"], test_scores, threshold, prefix="test_")
    return {
        "sprint": config.sprint,
        "label_scheme": config.label_scheme,
        "split_id": config.split_id,
        "seed": config.seed,
        "training_regime": config.training_regime,
        "model_name": model_name,
        "feature_set": feature_set,
        "train_rows": int(split_data["y_train"].shape[0]),
        "val_rows": int(split_data["y_val"].shape[0]),
        "test_rows": int(split_data["y_test"].shape[0]),
        "feature_columns": len(split_data["feature_columns"]),
        "threshold_policy": threshold_policy,
        "threshold": float(threshold),
        "notes": notes,
        **val_metrics,
        **test_metrics,
    }


def _prediction_records(
    model_name: str,
    feature_set: FeatureSetName,
    split_data: dict[str, Any],
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "split": "val",
            "row_index": split_data["val_index"],
            "y_true": split_data["y_val"],
            "y_score": val_scores,
        },
        {
            "model_name": model_name,
            "feature_set": feature_set,
            "split": "test",
            "row_index": split_data["test_index"],
            "y_true": split_data["y_test"],
            "y_score": test_scores,
        },
    ]
