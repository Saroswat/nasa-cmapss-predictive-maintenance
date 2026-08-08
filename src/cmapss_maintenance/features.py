from __future__ import annotations

import pandas as pd


def select_feature_columns(frame: pd.DataFrame, *, variance_floor: float = 1e-12) -> list[str]:
    """Select numeric signals while removing constants and targets."""
    excluded = {"unit_number", "rul", "rul_raw", "maintenance_due"}
    candidates = [column for column in frame.select_dtypes("number") if column not in excluded]
    return [column for column in candidates if float(frame[column].var()) > variance_floor]


def add_health_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Add causal rolling features computed independently for each engine."""
    result = frame.copy()
    grouped = result.groupby("unit_number", sort=False)
    sensor_columns = [column for column in feature_columns if column.startswith("sensor_")]
    for column in sensor_columns:
        rolling = grouped[column].rolling(window=5, min_periods=1)
        result[f"{column}_mean_5"] = rolling.mean().reset_index(level=0, drop=True)
        result[f"{column}_std_5"] = rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
    return result
