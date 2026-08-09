from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def markdown(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # NASA C-MAPSS: Portable End-to-End Predictive Maintenance

        This notebook is a complete, standalone implementation of remaining-useful-life (RUL)
        prediction and cost-aware maintenance decisions for NASA's C-MAPSS FD001 turbofan dataset.

        **Supported environments**

        | Environment | Recommended runtime | Notes |
        |---|---|---|
        | MacBook Pro M4 | arm64 Python 3.11+ | CPU models; no CUDA needed |
        | Windows RTX 4060 | Python 3.11+ | GPU detected; pipeline uses CPU |
        | Google Colab free tier | Standard CPU runtime | Preserves the free GPU quota |

        The dataset is small enough for 8 GB system RAM. The notebook bounds parallel work to eight
        CPU threads and does not place data in GPU VRAM. Run cells in order, or select **Run All**.

        > This is a research and educational workflow, not operational aviation guidance.
        """
    ),
    markdown(
        """
        ## 0. How to run

        **VS Code on macOS or Windows**

        1. Install Python 3.11 or newer and the VS Code Python + Jupyter extensions.
        2. Open this notebook and select a Python 3.11+ kernel.
        3. Run all cells. Internet access is required only for missing packages and the dataset.

        **Google Colab**

        Upload this `.ipynb` file, choose the standard CPU runtime, and select
        **Runtime > Run all**.
        Generated files are collected into a ZIP near the end of the notebook.

        You may change `CMAPSS_WORKDIR`, `N_ESTIMATORS`, and the maintenance economics in the
        configuration cell. Defaults favor reproducibility and free-tier compatibility.
        """
    ),
    code(
        '''
        # Install only missing dependencies. Run this cell before importing scientific libraries.
        from __future__ import annotations

        import importlib.metadata
        import importlib.util
        import os
        import platform
        import re
        import subprocess
        import sys
        from pathlib import Path

        if sys.version_info < (3, 11):  # noqa: UP036 - user-facing kernel validation
            raise RuntimeError(
                "Python 3.11 or newer is required. Select a newer VS Code kernel or Colab runtime."
            )

        requirements = {
            "joblib": ("joblib", (1, 4), "joblib>=1.4,<2"),
            "matplotlib": ("matplotlib", (3, 9), "matplotlib>=3.9,<4"),
            "numpy": ("numpy", (2, 0), "numpy>=2.0,<3"),
            "pandas": ("pandas", (2, 2), "pandas>=2.2,<4"),
            "sklearn": ("scikit-learn", (1, 6), "scikit-learn>=1.6,<2"),
            "seaborn": ("seaborn", (0, 13), "seaborn>=0.13,<1"),
        }

        def numeric_version(value: str) -> tuple[int, ...]:
            return tuple(int(part) for part in re.findall(r"\\d+", value)[:3])

        install_specs = []
        for module, (distribution, minimum, spec) in requirements.items():
            available = importlib.util.find_spec(module) is not None
            installed = (
                numeric_version(importlib.metadata.version(distribution))
                if available
                else ()
            )
            if not available or installed < minimum:
                install_specs.append(spec)

        if install_specs:
            print("Installing or upgrading packages:", ", ".join(install_specs))
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", *install_specs]
            )
        else:
            print("All required Python packages meet the minimum versions.")

        os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

        def optional_command(command: list[str]) -> str | None:
            try:
                return subprocess.check_output(
                    command, stderr=subprocess.DEVNULL, text=True, timeout=8
                ).strip()
            except (FileNotFoundError, subprocess.SubprocessError):
                return None

        in_colab = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
        gpu_name = optional_command(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        )
        print(f"Python:      {platform.python_version()}")
        print(f"System:      {platform.system()} {platform.machine()}")
        print(f"CPU threads: {os.cpu_count() or 'unknown'}")
        print(f"Environment: {'Google Colab' if in_colab else 'Local Jupyter / VS Code'}")
        print(f"NVIDIA GPU:  {gpu_name or 'Not detected (not required)'}")
        '''
    ),
    code(
        '''
        # Imports and experiment configuration
        import hashlib
        import json
        import math
        import shutil
        import time
        import urllib.error
        import urllib.request
        import warnings
        import zipfile
        from dataclasses import asdict, dataclass

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import sklearn
        from IPython.display import display
        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            mean_absolute_error,
            r2_score,
            root_mean_squared_error,
        )
        from sklearn.model_selection import GroupShuffleSplit

        RANDOM_STATE = 42
        N_ESTIMATORS = 200  # Raise to 300 for the repository's full reference configuration.
        MAX_CPU_JOBS = min(8, max(1, os.cpu_count() or 2))
        WORK_DIR = Path(
            os.environ.get("CMAPSS_WORKDIR", Path.cwd() / "cmapss_portable_run")
        ).expanduser().resolve()
        DATA_DIR = WORK_DIR / "data" / "raw"
        OUTPUT_DIR = WORK_DIR / "artifacts"

        WORK_DIR.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")
        np.random.seed(RANDOM_STATE)

        print(f"NumPy:       {np.__version__}")
        print(f"pandas:      {pd.__version__}")
        print(f"scikit-learn:{sklearn.__version__}")
        print(f"Working dir: {WORK_DIR}")
        '''
    ),
    markdown(
        """
        ## 1. Acquire and verify NASA C-MAPSS FD001

        The downloader first tries NASA's legacy archive. If that endpoint is unavailable, it uses a
        commit-pinned mirror. In either case, each FD001 file must match its pinned SHA-256 digest
        before modelling begins.
        """
    ),
    code(
        '''
        DATASET_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
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

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def fd001_is_valid(data_dir: Path) -> bool:
            return all(
                (data_dir / filename).exists()
                and sha256(data_dir / filename) == expected
                for filename, expected in FD001_SHA256.items()
            )

        def download_mirror(data_dir: Path) -> None:
            for filename, expected in FD001_SHA256.items():
                destination = data_dir / filename
                temporary = destination.with_suffix(destination.suffix + ".part")
                with (
                    urllib.request.urlopen(f"{MIRROR_URL}/{filename}", timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
                if sha256(temporary) != expected:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"Checksum mismatch for mirrored file: {filename}")
                temporary.replace(destination)

        def download_dataset(data_dir: Path, force: bool = False) -> Path:
            if fd001_is_valid(data_dir) and not force:
                print("Using existing checksum-verified FD001 files.")
                return data_dir

            data_dir.mkdir(parents=True, exist_ok=True)
            archive = data_dir / "CMAPSSData.zip"
            try:
                print("Downloading NASA C-MAPSS archive...")
                with urllib.request.urlopen(DATASET_URL, timeout=30) as response, archive.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(response, output)
                with zipfile.ZipFile(archive) as bundle:
                    root = data_dir.resolve()
                    for member in bundle.infolist():
                        destination = (data_dir / member.filename).resolve()
                        if root not in destination.parents and destination != root:
                            raise ValueError(f"Unsafe archive member: {member.filename}")
                    bundle.extractall(data_dir)
            except (OSError, TimeoutError, urllib.error.URLError, zipfile.BadZipFile) as error:
                archive.unlink(missing_ok=True)
                warnings.warn(
                    f"NASA archive unavailable ({error}); using the pinned FD001 mirror.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                download_mirror(data_dir)

            if not fd001_is_valid(data_dir):
                raise ValueError("FD001 files failed SHA-256 verification")
            return data_dir

        def read_trajectory(path: Path) -> pd.DataFrame:
            frame = pd.read_csv(path, sep=r"\\s+", header=None, names=COLUMNS)
            if frame.shape[1] != len(COLUMNS):
                raise ValueError(f"Unexpected column count in {path}: {frame.shape[1]}")
            return frame

        def load_fd001(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
            train = read_trajectory(data_dir / "train_FD001.txt")
            test = read_trajectory(data_dir / "test_FD001.txt")
            truth = pd.read_csv(
                data_dir / "RUL_FD001.txt", sep=r"\\s+", header=None
            ).iloc[:, 0]
            truth.name = "rul"
            if test["unit_number"].nunique() != len(truth):
                raise ValueError("Test engine count does not match the RUL truth file")
            return train, test, truth

        download_dataset(DATA_DIR)
        train_raw, test_raw, test_rul = load_fd001(DATA_DIR)
        print(f"Train: {train_raw.shape[0]:,} rows, {train_raw.unit_number.nunique()} engines")
        print(f"Test:  {test_raw.shape[0]:,} rows, {test_raw.unit_number.nunique()} engines")
        print(f"Truth: {len(test_rul)} endpoint RUL values")
        '''
    ),
    markdown("## 2. Data quality and exploratory analysis"),
    code(
        '''
        summary = pd.DataFrame(
            {
                "rows": [len(train_raw), len(test_raw)],
                "engines": [train_raw.unit_number.nunique(), test_raw.unit_number.nunique()],
                "minimum_cycle": [train_raw.time_in_cycles.min(), test_raw.time_in_cycles.min()],
                "maximum_cycle": [train_raw.time_in_cycles.max(), test_raw.time_in_cycles.max()],
                "missing_values": [
                    int(train_raw.isna().sum().sum()),
                    int(test_raw.isna().sum().sum()),
                ],
                "duplicate_rows": [
                    int(train_raw.duplicated().sum()),
                    int(test_raw.duplicated().sum()),
                ],
            },
            index=["train", "test"],
        )
        display(summary)

        sensor_variance = train_raw.filter(like="sensor_").var().sort_values()
        constant_sensors = sensor_variance[sensor_variance <= 1e-12].index.tolist()
        print("Near-constant sensors removed later:", constant_sensors)
        display(train_raw.head())
        '''
    ),
    code(
        '''
        sensors_to_plot = ["sensor_2", "sensor_7", "sensor_11", "sensor_15"]
        engines_to_plot = [1, 25, 50, 75, 100]
        sample = train_raw[train_raw.unit_number.isin(engines_to_plot)]

        figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
        for sensor, axis in zip(sensors_to_plot, axes.ravel(), strict=True):
            for unit, trajectory in sample.groupby("unit_number"):
                axis.plot(
                    trajectory.time_in_cycles,
                    trajectory[sensor],
                    linewidth=1.1,
                    alpha=0.85,
                    label=f"Engine {unit}",
                )
            axis.set(title=sensor.replace("_", " ").title(), xlabel="Cycle", ylabel="Reading")
        axes[0, 0].legend(ncol=2, fontsize=8)
        figure.suptitle("Selected FD001 degradation signals", fontsize=15)
        figure.tight_layout()
        plt.show()
        '''
    ),
    markdown(
        """
        ## 3. Leakage-safe targets and causal health features

        Training RUL is calculated within each engine and capped at 125 cycles to represent the
        early-life healthy plateau. Rolling features use only the current and four previous
        readings. Engines, not individual rows, are separated during validation.
        """
    ),
    code(
        '''
        def add_train_rul(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
            result = frame.copy()
            max_cycles = result.groupby("unit_number")["time_in_cycles"].transform("max")
            result["rul_raw"] = max_cycles - result["time_in_cycles"]
            result["rul"] = (
                result["rul_raw"].clip(upper=cap)
                if cap is not None
                else result["rul_raw"]
            )
            return result

        def select_feature_columns(
            frame: pd.DataFrame, variance_floor: float = 1e-12
        ) -> list[str]:
            excluded = {"unit_number", "rul", "rul_raw", "maintenance_due"}
            candidates = [
                column
                for column in frame.select_dtypes("number")
                if column not in excluded
            ]
            return [
                column for column in candidates if float(frame[column].var()) > variance_floor
            ]

        def add_health_features(
            frame: pd.DataFrame, feature_columns: list[str]
        ) -> pd.DataFrame:
            result = frame.copy()
            grouped = result.groupby("unit_number", sort=False)
            additions: dict[str, pd.Series] = {}
            sensor_columns = [
                column for column in feature_columns if column.startswith("sensor_")
            ]
            for column in sensor_columns:
                rolling = grouped[column].rolling(window=5, min_periods=1)
                additions[f"{column}_mean_5"] = rolling.mean().reset_index(level=0, drop=True)
                additions[f"{column}_std_5"] = (
                    rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
                )
            return pd.concat([result, pd.DataFrame(additions, index=result.index)], axis=1)

        train = add_train_rul(train_raw, cap=125)
        base_features = select_feature_columns(train)
        train_engineered = add_health_features(train, base_features)
        test_engineered = add_health_features(test_raw, base_features)
        feature_columns = select_feature_columns(train_engineered)

        print(f"Base features:       {len(base_features)}")
        print(f"Engineered features: {len(feature_columns)}")
        print(f"Training matrix:     {train_engineered[feature_columns].shape}")

        example = train[train.unit_number == 1]
        plt.figure(figsize=(10, 4))
        plt.plot(example.time_in_cycles, example.rul_raw, label="Raw RUL", linewidth=2)
        plt.plot(example.time_in_cycles, example.rul, label="Capped training target", linewidth=2)
        plt.title("Piecewise-linear RUL target for engine 1")
        plt.xlabel("Cycle")
        plt.ylabel("Remaining useful life")
        plt.legend()
        plt.tight_layout()
        plt.show()
        '''
    ),
    markdown("## 4. Model selection, official test evaluation, and maintenance economics"),
    code(
        '''
        @dataclass(frozen=True)
        class ExperimentConfig:
            rul_cap: int = 125
            maintenance_horizon: int = 30
            validation_size: float = 0.2
            random_state: int = RANDOM_STATE
            n_estimators: int = N_ESTIMATORS
            n_jobs: int = MAX_CPU_JOBS
            false_positive_cost: int = 100_000
            false_negative_cost: int = 200_000
            true_positive_value: int = 300_000

        @dataclass(frozen=True)
        class RegressionMetrics:
            mae: float
            rmse: float
            r2: float
            nasa_score: float

        @dataclass(frozen=True)
        class MaintenanceValue:
            true_positives: int
            true_negatives: int
            false_positives: int
            false_negatives: int
            expected_value: int

        @dataclass
        class ExperimentResult:
            regression_model: object
            maintenance_model: RandomForestClassifier
            feature_columns: list[str]
            threshold: float
            predictions: pd.DataFrame
            metrics: dict[str, object]

        def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
            penalties = np.where(
                errors < 0,
                np.exp(-errors / 13.0) - 1.0,
                np.exp(errors / 10.0) - 1.0,
            )
            return float(penalties.sum())

        def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
            return RegressionMetrics(
                mae=float(mean_absolute_error(y_true, y_pred)),
                rmse=float(root_mean_squared_error(y_true, y_pred)),
                r2=float(r2_score(y_true, y_pred)),
                nasa_score=nasa_score(y_true, y_pred),
            )

        def maintenance_value(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            config: ExperimentConfig,
        ) -> MaintenanceValue:
            truth = np.asarray(y_true, dtype=int)
            prediction = np.asarray(y_pred, dtype=int)
            tp = int(((truth == 1) & (prediction == 1)).sum())
            tn = int(((truth == 0) & (prediction == 0)).sum())
            fp = int(((truth == 0) & (prediction == 1)).sum())
            fn = int(((truth == 1) & (prediction == 0)).sum())
            expected = (
                tp * config.true_positive_value
                - fp * config.false_positive_cost
                - fn * config.false_negative_cost
            )
            return MaintenanceValue(tp, tn, fp, fn, int(expected))

        def validation_checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
            offsets = {0, 10, 20, 30, 45, 60, 90, 120}
            return frame[frame.rul_raw.isin(offsets)].sort_values(
                ["unit_number", "time_in_cycles"]
            )

        def candidate_regressors(config: ExperimentConfig) -> dict[str, object]:
            return {
                "random_forest": RandomForestRegressor(
                    n_estimators=config.n_estimators,
                    max_features=0.7,
                    min_samples_leaf=3,
                    n_jobs=config.n_jobs,
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

        def best_threshold(
            y_true: np.ndarray,
            probabilities: np.ndarray,
            config: ExperimentConfig,
        ) -> tuple[float, dict[str, int]]:
            scored = []
            for threshold in np.linspace(0.05, 0.95, 91):
                value = maintenance_value(y_true, probabilities >= threshold, config)
                scored.append((value.expected_value, float(threshold), asdict(value)))
            _, threshold, details = max(scored, key=lambda item: (item[0], item[1]))
            return threshold, details

        def run_experiment(
            train_frame: pd.DataFrame,
            test_frame: pd.DataFrame,
            truth: pd.Series,
            config: ExperimentConfig,
        ) -> ExperimentResult:
            base = select_feature_columns(train_frame)
            engineered_train = add_health_features(train_frame, base)
            engineered_test = add_health_features(test_frame, base)
            features = select_feature_columns(engineered_train)

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.validation_size,
                random_state=config.random_state,
            )
            development_indices, validation_indices = next(
                splitter.split(engineered_train, groups=engineered_train.unit_number)
            )
            development = engineered_train.iloc[development_indices]
            validation = engineered_train.iloc[validation_indices]
            validation_points = validation_checkpoints(validation)

            validation_scores = {}
            for name, estimator in candidate_regressors(config).items():
                estimator.fit(development[features], development.rul)
                prediction = np.clip(
                    estimator.predict(validation_points[features]), 0, None
                )
                validation_scores[name] = asdict(
                    regression_metrics(validation_points.rul_raw.to_numpy(), prediction)
                )

            selected_name = min(
                validation_scores,
                key=lambda name: validation_scores[name]["nasa_score"],
            )
            regression_model = candidate_regressors(config)[selected_name]
            regression_model.fit(engineered_train[features], engineered_train.rul)

            horizon = config.maintenance_horizon
            maintenance_model = RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=config.n_jobs,
                random_state=config.random_state,
            )
            development_due = (development.rul_raw <= horizon).astype(int)
            maintenance_model.fit(development[features], development_due)
            validation_probability = maintenance_model.predict_proba(
                validation_points[features]
            )[:, 1]
            threshold, validation_value = best_threshold(
                (validation_points.rul_raw <= horizon).to_numpy(),
                validation_probability,
                config,
            )
            all_due = (engineered_train.rul_raw <= horizon).astype(int)
            maintenance_model.fit(engineered_train[features], all_due)

            endpoints = engineered_test.groupby("unit_number", sort=True).tail(1)
            rul_prediction = np.clip(
                regression_model.predict(endpoints[features]), 0, None
            )
            probability = maintenance_model.predict_proba(endpoints[features])[:, 1]
            maintenance_prediction = (probability >= threshold).astype(int)
            maintenance_truth = (truth.to_numpy() <= horizon).astype(int)
            predictions = pd.DataFrame(
                {
                    "unit_number": endpoints.unit_number.to_numpy(dtype=int),
                    "rul_actual": truth.to_numpy(dtype=float),
                    "rul_predicted": rul_prediction,
                    "maintenance_actual": maintenance_truth,
                    "maintenance_probability": probability,
                    "maintenance_predicted": maintenance_prediction,
                }
            )
            test_metrics = regression_metrics(truth.to_numpy(), rul_prediction)
            metrics = {
                "selected_regressor": selected_name,
                "validation_regression": validation_scores,
                "test_regression": asdict(test_metrics),
                "maintenance_threshold": threshold,
                "validation_maintenance_value": validation_value,
                "test_maintenance_value": asdict(
                    maintenance_value(maintenance_truth, maintenance_prediction, config)
                ),
                "train_engines": int(train_frame.unit_number.nunique()),
                "test_engines": int(test_frame.unit_number.nunique()),
                "feature_count": len(features),
                "n_estimators": config.n_estimators,
            }
            return ExperimentResult(
                regression_model,
                maintenance_model,
                features,
                threshold,
                predictions,
                metrics,
            )
        '''
    ),
    code(
        '''
        CONFIG = ExperimentConfig()
        print(CONFIG)
        print(f"Training with at most {CONFIG.n_jobs} CPU threads...")
        started = time.perf_counter()
        result = run_experiment(train, test_raw, test_rul, CONFIG)
        elapsed = time.perf_counter() - started
        print(f"Completed in {elapsed:.1f} seconds")
        print(f"Selected regressor: {result.metrics['selected_regressor']}")
        '''
    ),
    markdown("## 5. Regression results"),
    code(
        '''
        test_metrics_frame = pd.Series(
            result.metrics["test_regression"], name="Official FD001 test"
        ).to_frame()
        validation_frame = pd.DataFrame(result.metrics["validation_regression"]).T
        validation_frame.index = validation_frame.index.str.replace("_", " ").str.title()
        display(test_metrics_frame.style.format("{:.4f}"))
        display(validation_frame.style.format("{:.4f}"))

        ordered = result.predictions.sort_values(
            "rul_actual", ascending=False
        ).reset_index(drop=True)
        figure, axis = plt.subplots(figsize=(13, 5))
        axis.plot(ordered.rul_actual, label="Actual RUL", linewidth=2.2, color="#1f4e5f")
        axis.plot(
            ordered.rul_predicted,
            label="Predicted RUL",
            linewidth=1.8,
            color="#c27c2c",
        )
        axis.set(
            title="FD001 test endpoints: actual vs predicted RUL",
            xlabel="Test engines sorted by actual RUL",
            ylabel="Remaining cycles",
        )
        axis.legend()
        figure.tight_layout()
        plt.show()

        figure, axis = plt.subplots(figsize=(6, 6))
        maximum = math.ceil(
            max(result.predictions.rul_actual.max(), result.predictions.rul_predicted.max()) / 10
        ) * 10
        axis.scatter(
            result.predictions.rul_actual,
            result.predictions.rul_predicted,
            alpha=0.75,
            color="#287271",
        )
        axis.plot([0, maximum], [0, maximum], "--", color="#9c4038", label="Perfect prediction")
        axis.set(
            xlim=(0, maximum),
            ylim=(0, maximum),
            title="RUL prediction parity",
            xlabel="Actual RUL",
            ylabel="Predicted RUL",
        )
        axis.legend()
        figure.tight_layout()
        plt.show()
        '''
    ),
    markdown("## 6. Cost-aware maintenance policy"),
    code(
        '''
        maintenance_metrics = result.metrics["test_maintenance_value"]
        display(
            pd.Series(
                {
                    "Decision threshold": result.threshold,
                    "Maintenance horizon (cycles)": CONFIG.maintenance_horizon,
                    **maintenance_metrics,
                },
                name="Test policy",
            ).to_frame()
        )

        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        ConfusionMatrixDisplay.from_predictions(
            result.predictions.maintenance_actual,
            result.predictions.maintenance_predicted,
            display_labels=["No action", "Maintenance"],
            cmap="Blues",
            colorbar=False,
            ax=axes[0],
        )
        axes[0].set_title("Cost-aware maintenance decisions")

        risk_ordered = result.predictions.sort_values(
            "maintenance_probability", ascending=False
        ).reset_index(drop=True)
        axes[1].plot(
            risk_ordered.maintenance_probability,
            color="#a8453d",
            linewidth=2,
            label="Predicted risk",
        )
        axes[1].axhline(
            result.threshold,
            color="#b98420",
            linestyle="--",
            label=f"Threshold ({result.threshold:.2f})",
        )
        axes[1].set(
            ylim=(-0.02, 1.02),
            title="Fleet maintenance-risk ranking",
            xlabel="Test engines sorted by risk",
            ylabel="Maintenance probability",
        )
        axes[1].legend()
        figure.tight_layout()
        plt.show()

        review_queue = result.predictions.query("maintenance_predicted == 1").sort_values(
            "maintenance_probability", ascending=False
        )
        display(
            review_queue.head(15).style.format(
                {"maintenance_probability": "{:.1%}", "rul_predicted": "{:.1f}"}
            )
        )
        '''
    ),
    markdown("## 7. Model interpretation and one-engine trajectory"),
    code(
        '''
        if hasattr(result.regression_model, "feature_importances_"):
            importance = pd.Series(
                result.regression_model.feature_importances_,
                index=result.feature_columns,
                name="importance",
            ).sort_values(ascending=False)
            display(importance.head(15).to_frame().style.format("{:.4f}"))
            figure, axis = plt.subplots(figsize=(9, 6))
            importance.head(15).sort_values().plot.barh(ax=axis, color="#356a99")
            axis.set(title="Top random-forest features", xlabel="Impurity-based importance")
            figure.tight_layout()
            plt.show()
        else:
            importance = pd.Series(dtype=float, name="importance")
            print("The selected model does not expose impurity-based feature importance.")

        ENGINE_ID = int(
            result.predictions.sort_values(
                "maintenance_probability", ascending=False
            ).iloc[0].unit_number
        )
        engine_history = test_engineered[test_engineered.unit_number == ENGINE_ID].copy()
        engine_history["predicted_rul"] = np.clip(
            result.regression_model.predict(engine_history[result.feature_columns]), 0, None
        )
        engine_history["maintenance_risk"] = result.maintenance_model.predict_proba(
            engine_history[result.feature_columns]
        )[:, 1]

        figure, left_axis = plt.subplots(figsize=(12, 5))
        right_axis = left_axis.twinx()
        left_axis.plot(
            engine_history.time_in_cycles,
            engine_history.predicted_rul,
            color="#287271",
            linewidth=2.2,
            label="Predicted RUL",
        )
        right_axis.plot(
            engine_history.time_in_cycles,
            engine_history.maintenance_risk,
            color="#a8453d",
            linewidth=1.8,
            label="Maintenance risk",
        )
        right_axis.axhline(result.threshold, color="#b98420", linestyle="--", alpha=0.8)
        left_axis.set(
            title=f"Engine {ENGINE_ID}: model trajectory",
            xlabel="Observed cycle",
            ylabel="Predicted RUL",
        )
        right_axis.set(ylabel="Maintenance probability", ylim=(-0.02, 1.02))
        lines = left_axis.lines + right_axis.lines[:1]
        left_axis.legend(lines, [line.get_label() for line in lines], loc="center left")
        figure.tight_layout()
        plt.show()
        '''
    ),
    markdown("## 8. Save models, metrics, plots, predictions, and dashboard data"),
    code(
        '''
        def save_artifacts(result: ExperimentResult, output_dir: Path) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(output_dir / "fd001_predictions.csv", index=False)
            (output_dir / "metrics.json").write_text(
                json.dumps(result.metrics, indent=2, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
            joblib.dump(
                {
                    "regression_model": result.regression_model,
                    "maintenance_model": result.maintenance_model,
                    "feature_columns": result.feature_columns,
                    "maintenance_threshold": result.threshold,
                    "config": asdict(CONFIG),
                },
                output_dir / "fd001_models.joblib",
            )

            ordered = result.predictions.sort_values("rul_actual", ascending=False)
            figure, axis = plt.subplots(figsize=(12, 5))
            axis.plot(ordered.rul_actual.to_numpy(), label="Actual RUL", linewidth=2)
            axis.plot(ordered.rul_predicted.to_numpy(), label="Predicted RUL", linewidth=1.7)
            axis.set(
                title="FD001 Remaining Useful Life",
                xlabel="Test engines sorted by actual RUL",
                ylabel="Cycles",
            )
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_dir / "rul_predictions.png", dpi=160)
            plt.close(figure)

            figure, axis = plt.subplots(figsize=(5.5, 5))
            ConfusionMatrixDisplay.from_predictions(
                result.predictions.maintenance_actual,
                result.predictions.maintenance_predicted,
                display_labels=["No action", "Maintenance"],
                cmap="Blues",
                colorbar=False,
                ax=axis,
            )
            axis.set_title("Cost-aware maintenance decisions")
            figure.tight_layout()
            figure.savefig(output_dir / "maintenance_confusion_matrix.png", dpi=160)
            plt.close(figure)

            if hasattr(result.regression_model, "feature_importances_"):
                values = np.asarray(result.regression_model.feature_importances_)
                order = np.argsort(values)[-15:]
                figure, axis = plt.subplots(figsize=(9, 6))
                axis.barh(np.asarray(result.feature_columns)[order], values[order], color="#287271")
                axis.set(title="Top regression features", xlabel="Impurity-based importance")
                figure.tight_layout()
                figure.savefig(output_dir / "feature_importance.png", dpi=160)
                plt.close(figure)

        save_artifacts(result, OUTPUT_DIR)

        importance_records = []
        if hasattr(result.regression_model, "feature_importances_"):
            ranking = sorted(
                zip(
                    result.feature_columns,
                    result.regression_model.feature_importances_,
                    strict=True,
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
            importance_records = [
                {"feature": name, "importance": round(float(value), 6)}
                for name, value in ranking
            ]

        prediction_lookup = result.predictions.set_index("unit_number")
        dashboard_engines = []
        for unit, history in test_engineered.groupby("unit_number", sort=True):
            recent = history.tail(20).copy()
            recent_rul = np.clip(
                result.regression_model.predict(recent[result.feature_columns]), 0, None
            )
            recent_risk = result.maintenance_model.predict_proba(
                recent[result.feature_columns]
            )[:, 1]
            endpoint = prediction_lookup.loc[int(unit)]
            dashboard_engines.append(
                {
                    "id": int(unit),
                    "cycle": int(recent.time_in_cycles.iloc[-1]),
                    "actualRul": float(endpoint.rul_actual),
                    "predictedRul": float(endpoint.rul_predicted),
                    "risk": float(endpoint.maintenance_probability),
                    "recommended": bool(endpoint.maintenance_predicted),
                    "actualMaintenance": bool(endpoint.maintenance_actual),
                    "history": [
                        {
                            "cycle": int(cycle),
                            "predictedRul": round(float(rul), 2),
                            "risk": round(float(risk), 4),
                        }
                        for cycle, rul, risk in zip(
                            recent.time_in_cycles, recent_rul, recent_risk, strict=True
                        )
                    ],
                }
            )

        dashboard_payload = {
            "meta": {
                "dataset": "NASA C-MAPSS FD001",
                "model": result.metrics["selected_regressor"].replace("_", " ").title(),
                "threshold": result.threshold,
                "maintenanceHorizon": CONFIG.maintenance_horizon,
            },
            "metrics": result.metrics,
            "featureImportance": importance_records,
            "engines": dashboard_engines,
        }
        (OUTPUT_DIR / "dashboard.json").write_text(
            json.dumps(dashboard_payload, indent=2) + "\\n", encoding="utf-8"
        )

        archive_path = shutil.make_archive(
            str(WORK_DIR / "cmapss_results"), "zip", root_dir=OUTPUT_DIR
        )
        print("Saved artifacts:")
        for path in sorted(OUTPUT_DIR.iterdir()):
            print(f"  {path.name:<38} {path.stat().st_size / 1024:>9.1f} KiB")
        print(f"ZIP archive: {archive_path}")

        if in_colab:
            print("Colab download command (run when wanted):")
            print("from google.colab import files; files.download(r'" + archive_path + "')")
        '''
    ),
    markdown(
        """
        ## Interpretation and deployment limitations

        - **No row leakage:** validation separates complete engines with `GroupShuffleSplit`.
        - **No future leakage:** rolling features are causal and computed per engine.
        - **Official evaluation:** endpoint predictions use NASA's untouched FD001 truth.
        - **Asymmetric safety metric:** NASA score penalizes late predictions more heavily.
        - **Cost-aware policy:** the probability threshold is selected only on held-out engines.
        - **Portable compute:** the CPU implementation behaves consistently across Apple Silicon,
          Windows/NVIDIA hardware, and Colab. The RTX GPU is intentionally not required.

        For operational use, add calibrated uncertainty, engine-specific operating regimes, drift
        monitoring, maintenance history, safety constraints, explainability review, and aviation
        certification. Replace the illustrative dollar values with organization-specific costs.
        """
    ),
]

notebook = new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
)

output = Path("notebooks/02_portable_end_to_end_cmapss.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, output)
print(f"Wrote {output} with {len(cells)} cells")
