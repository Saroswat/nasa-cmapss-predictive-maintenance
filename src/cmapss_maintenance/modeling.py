from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GroupShuffleSplit

from .config import ExperimentConfig
from .features import add_health_features, select_feature_columns
from .metrics import maintenance_value, nasa_score, regression_metrics


@dataclass
class ExperimentResult:
    regression_model: object
    maintenance_model: RandomForestClassifier
    feature_columns: list[str]
    threshold: float
    predictions: pd.DataFrame
    metrics: dict[str, object]


def _split_by_engine(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.validation_size,
        random_state=config.random_state,
    )
    train_indices, validation_indices = next(
        splitter.split(frame, groups=frame["unit_number"])
    )
    return train_indices, validation_indices


def _regressors(config: ExperimentConfig) -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_features=0.7,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=config.random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=config.n_estimators,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=config.random_state,
        ),
    }


def _validation_checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Simulate test-like censoring at several distances from failure per engine."""
    offsets = {0, 10, 20, 30, 45, 60, 90, 120}
    checkpoints = frame[frame["rul_raw"].isin(offsets)]
    return checkpoints.sort_values(["unit_number", "time_in_cycles"]).copy()


def _best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: ExperimentConfig,
) -> tuple[float, dict[str, int]]:
    candidates = np.linspace(0.05, 0.95, 91)
    scored: list[tuple[int, float, dict[str, int]]] = []
    for threshold in candidates:
        value = maintenance_value(
            y_true,
            probabilities >= threshold,
            true_positive_value=config.true_positive_value,
            false_positive_cost=config.false_positive_cost,
            false_negative_cost=config.false_negative_cost,
        )
        scored.append((value.expected_value, float(threshold), value.to_dict()))
    _, threshold, details = max(scored, key=lambda item: (item[0], item[1]))
    return threshold, details


def run_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.Series,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Train, select, refit, and evaluate RUL and maintenance models."""
    config = config or ExperimentConfig()
    base_features = select_feature_columns(train)
    train_engineered = add_health_features(train, base_features)
    test_engineered = add_health_features(test, base_features)
    feature_columns = select_feature_columns(train_engineered)

    train_indices, validation_indices = _split_by_engine(train_engineered, config)
    development = train_engineered.iloc[train_indices]
    validation = train_engineered.iloc[validation_indices]

    validation_points = _validation_checkpoints(validation)
    validation_scores: dict[str, dict[str, float]] = {}
    fitted_regressors: dict[str, object] = {}
    for name, estimator in _regressors(config).items():
        estimator.fit(development[feature_columns], development["rul"])
        prediction = np.clip(estimator.predict(validation_points[feature_columns]), 0, None)
        validation_scores[name] = regression_metrics(
            validation_points["rul_raw"].to_numpy(), prediction
        ).to_dict()
        fitted_regressors[name] = estimator

    selected_name = min(validation_scores, key=lambda name: validation_scores[name]["nasa_score"])
    regression_model = _regressors(config)[selected_name]
    regression_model.fit(train_engineered[feature_columns], train_engineered["rul"])

    horizon = config.maintenance_horizon
    train_engineered["maintenance_due"] = (train_engineered["rul_raw"] <= horizon).astype(int)
    maintenance_model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.random_state,
    )
    maintenance_model.fit(
        development[feature_columns],
        (development["rul_raw"] <= horizon).astype(int),
    )
    validation_probabilities = maintenance_model.predict_proba(
        validation_points[feature_columns]
    )[:, 1]
    threshold, validation_value = _best_threshold(
        (validation_points["rul_raw"] <= horizon).to_numpy(),
        validation_probabilities,
        config,
    )
    maintenance_model.fit(
        train_engineered[feature_columns], train_engineered["maintenance_due"]
    )

    endpoints = test_engineered.groupby("unit_number", sort=True).tail(1).copy()
    rul_prediction = np.clip(regression_model.predict(endpoints[feature_columns]), 0, None)
    maintenance_probability = maintenance_model.predict_proba(endpoints[feature_columns])[:, 1]
    maintenance_prediction = (maintenance_probability >= threshold).astype(int)
    maintenance_truth = (test_rul.to_numpy() <= horizon).astype(int)

    predictions = pd.DataFrame(
        {
            "unit_number": endpoints["unit_number"].to_numpy(dtype=int),
            "rul_actual": test_rul.to_numpy(dtype=float),
            "rul_predicted": rul_prediction,
            "maintenance_actual": maintenance_truth,
            "maintenance_probability": maintenance_probability,
            "maintenance_predicted": maintenance_prediction,
        }
    )
    metrics = {
        "selected_regressor": selected_name,
        "validation_regression": validation_scores,
        "test_regression": regression_metrics(test_rul.to_numpy(), rul_prediction).to_dict(),
        "maintenance_threshold": threshold,
        "validation_maintenance_value": validation_value,
        "test_maintenance_value": maintenance_value(
            maintenance_truth,
            maintenance_prediction,
            true_positive_value=config.true_positive_value,
            false_positive_cost=config.false_positive_cost,
            false_negative_cost=config.false_negative_cost,
        ).to_dict(),
        "test_nasa_score_check": nasa_score(test_rul.to_numpy(), rul_prediction),
        "train_engines": int(train["unit_number"].nunique()),
        "test_engines": int(test["unit_number"].nunique()),
        "feature_count": len(feature_columns),
    }
    return ExperimentResult(
        regression_model=regression_model,
        maintenance_model=maintenance_model,
        feature_columns=feature_columns,
        threshold=threshold,
        predictions=predictions,
        metrics=metrics,
    )
