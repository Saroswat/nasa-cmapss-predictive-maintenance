import numpy as np
import pandas as pd

from cmapss_maintenance.config import COLUMNS, ExperimentConfig
from cmapss_maintenance.data import add_train_rul
from cmapss_maintenance.modeling import run_experiment


def _trajectory(units: int, cycles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            row = [0.0] * len(COLUMNS)
            row[0], row[1] = unit, cycle
            degradation = cycle / cycles
            for index in range(2, len(COLUMNS)):
                row[index] = degradation * (index + 1) + rng.normal(0, 0.01)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_experiment_runs_end_to_end_on_synthetic_data() -> None:
    train = add_train_rul(_trajectory(10, 16, seed=1), cap=12)
    test = _trajectory(4, 10, seed=2)
    truth = pd.Series([6, 6, 6, 6], name="rul")
    config = ExperimentConfig(
        rul_cap=12,
        maintenance_horizon=5,
        n_estimators=10,
        validation_size=0.2,
    )

    result = run_experiment(train, test, truth, config)

    assert len(result.predictions) == 4
    assert result.metrics["test_engines"] == 4
    assert 0.05 <= result.threshold <= 0.95
    assert set(result.predictions["maintenance_predicted"]) <= {0, 1}
