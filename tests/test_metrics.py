import numpy as np

from cmapss_maintenance.metrics import maintenance_value, nasa_score, regression_metrics


def test_nasa_score_is_zero_for_perfect_predictions() -> None:
    values = np.array([5.0, 10.0, 20.0])
    assert nasa_score(values, values) == 0.0


def test_nasa_score_penalizes_late_predictions_more() -> None:
    truth = np.array([50.0])
    assert nasa_score(truth, np.array([60.0])) > nasa_score(truth, np.array([40.0]))


def test_regression_metrics_have_expected_keys() -> None:
    result = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0])).to_dict()
    assert result == {"mae": 0.0, "rmse": 0.0, "r2": 1.0, "nasa_score": 0.0}


def test_maintenance_value_uses_cost_matrix() -> None:
    value = maintenance_value(
        np.array([1, 1, 0, 0]),
        np.array([1, 0, 1, 0]),
    )
    assert value.to_dict() == {
        "true_positives": 1,
        "true_negatives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "expected_value": 0,
    }
