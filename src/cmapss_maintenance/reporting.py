from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from .modeling import ExperimentResult


def save_artifacts(result: ExperimentResult, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output_dir / "fd001_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(
        {
            "regression_model": result.regression_model,
            "maintenance_model": result.maintenance_model,
            "feature_columns": result.feature_columns,
            "maintenance_threshold": result.threshold,
        },
        output_dir / "fd001_models.joblib",
    )

    sns.set_theme(style="whitegrid", context="notebook")
    ordered = result.predictions.sort_values("rul_actual", ascending=False)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(ordered["rul_actual"].to_numpy(), label="Actual RUL", linewidth=2)
    axis.plot(ordered["rul_predicted"].to_numpy(), label="Predicted RUL", linewidth=1.7)
    axis.set(title="FD001 Remaining Useful Life", xlabel="Test engines (sorted)", ylabel="Cycles")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "rul_predictions.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        result.predictions["maintenance_actual"],
        result.predictions["maintenance_predicted"],
        display_labels=["No action", "Maintenance"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Cost-aware maintenance decisions")
    figure.tight_layout()
    figure.savefig(output_dir / "maintenance_confusion_matrix.png", dpi=160)
    plt.close(figure)

    model = result.regression_model
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        order = np.argsort(importance)[-15:]
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.barh(np.asarray(result.feature_columns)[order], importance[order], color="#287271")
        axis.set(title="Top regression features", xlabel="Impurity-based importance")
        figure.tight_layout()
        figure.savefig(output_dir / "feature_importance.png", dpi=160)
        plt.close(figure)
