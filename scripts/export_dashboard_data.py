from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from cmapss_maintenance.data import add_train_rul, load_fd001
from cmapss_maintenance.features import add_health_features, select_feature_columns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export verified FD001 results for the web dashboard"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("web/public/data/dashboard.json"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    metrics_path = args.artifact_dir / "metrics.json"
    models_path = args.artifact_dir / "fd001_models.joblib"
    if not metrics_path.exists() or not models_path.exists():
        raise FileNotFoundError(
            "Run `uv run cmapss-maintenance run` before exporting dashboard data"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bundle = joblib.load(models_path)
    train, test, truth = load_fd001(args.data_dir)
    base_features = select_feature_columns(add_train_rul(train))
    engineered = add_health_features(test, base_features)
    features = bundle["feature_columns"]

    rul_prediction = np.clip(bundle["regression_model"].predict(engineered[features]), 0, None)
    risk = bundle["maintenance_model"].predict_proba(engineered[features])[:, 1]
    engineered = engineered.assign(rul_prediction=rul_prediction, risk=risk)

    endpoint_indices = engineered.groupby("unit_number")["time_in_cycles"].idxmax()
    endpoints = engineered.loc[endpoint_indices].sort_values("unit_number")
    threshold = float(bundle["maintenance_threshold"])
    engines = []
    for position, endpoint in endpoints.reset_index(drop=True).iterrows():
        unit = int(endpoint["unit_number"])
        history = engineered[engineered["unit_number"] == unit].tail(20)
        probability = float(endpoint["risk"])
        engines.append(
            {
                "id": unit,
                "cycle": int(endpoint["time_in_cycles"]),
                "actualRul": float(truth.iloc[position]),
                "predictedRul": float(endpoint["rul_prediction"]),
                "risk": probability,
                "recommended": probability >= threshold,
                "actualMaintenance": float(truth.iloc[position]) <= 30,
                "history": [
                    {
                        "cycle": int(row["time_in_cycles"]),
                        "predictedRul": round(float(row["rul_prediction"]), 2),
                        "risk": round(float(row["risk"]), 4),
                    }
                    for _, row in history.iterrows()
                ],
            }
        )

    regression_model = bundle["regression_model"]
    feature_importance = []
    if hasattr(regression_model, "feature_importances_"):
        ranked = sorted(
            zip(features, regression_model.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        feature_importance = [
            {"feature": name, "importance": round(float(value), 6)} for name, value in ranked
        ]

    payload = {
        "meta": {
            "dataset": "NASA C-MAPSS FD001",
            "generatedAt": datetime.now(UTC).isoformat(),
            "model": metrics["selected_regressor"].replace("_", " ").title(),
            "threshold": threshold,
            "maintenanceHorizon": 30,
        },
        "metrics": metrics,
        "featureImportance": feature_importance,
        "engines": engines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(engines)} engines to {args.output}")


if __name__ == "__main__":
    main()
