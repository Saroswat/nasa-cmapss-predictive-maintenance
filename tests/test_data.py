from pathlib import Path

import pandas as pd

from cmapss_maintenance.config import COLUMNS
from cmapss_maintenance.data import add_train_rul, load_fd001
from cmapss_maintenance.data import test_endpoints as endpoints


def _write_frame(path: Path, rows: list[list[float]]) -> None:
    pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)


def test_load_rul_and_endpoints(tmp_path: Path) -> None:
    base = [0.0] * len(COLUMNS)
    rows = []
    for unit, cycles in ((1, 3), (2, 2)):
        for cycle in range(1, cycles + 1):
            row = base.copy()
            row[0], row[1], row[5] = unit, cycle, cycle * 0.1
            rows.append(row)
    _write_frame(tmp_path / "train_FD001.txt", rows)
    _write_frame(tmp_path / "test_FD001.txt", rows)
    pd.DataFrame([4, 7]).to_csv(
        tmp_path / "RUL_FD001.txt", sep=" ", header=False, index=False
    )

    train, test, truth = load_fd001(tmp_path)
    prepared = add_train_rul(train, cap=2)

    assert prepared.groupby("unit_number")["rul"].first().tolist() == [2, 1]
    assert endpoints(test)["time_in_cycles"].tolist() == [3, 2]
    assert truth.tolist() == [4, 7]
