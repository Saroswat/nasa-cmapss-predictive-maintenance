from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA's asymmetric RUL score; late predictions are penalized more heavily."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(errors < 0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
    return float(penalties.sum())


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    nasa_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(root_mean_squared_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        nasa_score=nasa_score(y_true, y_pred),
    )


@dataclass(frozen=True)
class MaintenanceValue:
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    expected_value: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def maintenance_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_positive_value: int = 300_000,
    false_positive_cost: int = 100_000,
    false_negative_cost: int = 200_000,
) -> MaintenanceValue:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    tp = int(((truth == 1) & (prediction == 1)).sum())
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    value = tp * true_positive_value - fp * false_positive_cost - fn * false_negative_cost
    return MaintenanceValue(tp, tn, fp, fn, int(value))
