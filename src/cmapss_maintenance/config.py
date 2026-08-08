from __future__ import annotations

from dataclasses import dataclass

DATASET_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
DATASET_NAME = "FD001"
MIRROR_COMMIT = "97cf10d200d07c6e9e20e75c52639ce6a08736ce"
MIRROR_URL = (
    "https://huggingface.co/datasets/DeveloperMindset123/"
    f"CMAPSS_Jet_Engine_Simulated_Data/resolve/{MIRROR_COMMIT}"
)
FD001_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}

COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]

ID_COLUMNS = ["unit_number", "time_in_cycles"]
TARGET_COLUMN = "rul"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = DATASET_NAME
    rul_cap: int = 125
    maintenance_horizon: int = 30
    validation_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    false_positive_cost: int = 100_000
    false_negative_cost: int = 200_000
    true_positive_value: int = 300_000
