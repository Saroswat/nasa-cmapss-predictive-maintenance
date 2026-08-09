from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


def markdown(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip())


source_path = Path("notebooks/02_portable_end_to_end_cmapss.ipynb")
notebook = nbformat.read(source_path, as_version=4)
base_cells = deepcopy(notebook.cells[:-1])
base_cells[0].source = dedent(
    """
    # NASA C-MAPSS: Operational Predictive Maintenance and Fleet Decision Support

    This standalone notebook retains the complete portable FD001 workflow and adds an operational
    assurance layer: data-quality gates, calibrated uncertainty, calibrated failure risk,
    out-of-distribution detection, population drift, persistent-alert logic, human-review
    overrides, capacity-aware work queues, immutable decision records, and a model card.

    It runs on Apple Silicon, Windows laptops, and the Google Colab free CPU tier. It is a realistic
    **decision-support prototype**, not an approved airworthiness system.
    """
).strip()

operational_cells = [
    markdown(
        """
        ## 9. Operational assurance basis

        The added controls translate primary guidance into executable prototype behavior:

        - [FAA AC 43-218](https://www.faa.gov/regulations_policies/advisory_circulars/index.cfm/go/document.information/documentID/1035729)
          treats integrated aircraft health management as an end-to-end maintenance program, not
          a model operating in isolation.
        - [EASA AI Concept Paper Issue 2](https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-issue-2)
          emphasizes learning assurance, explainability, and human oversight.
        - [EASA MLEAP](https://www.easa.europa.eu/en/newsroom-and-events/news/artificial-intelligence-easa-publishes-final-report-machine-learning)
          focuses on data representativeness, generalisation, and model robustness.
        - [NASA prognostics research](https://ntrs.nasa.gov/api/citations/20140010623/downloads/20140010623.pdf)
          frames RUL as a time-varying uncertainty propagation problem.
        - [NIST AI RMF](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
          calls for continuous monitoring, override, incident response, recovery, and change
          control.

        This notebook therefore keeps the human maintenance controller as the decision authority.
        Any uncertain, anomalous, or drift-affected prediction is routed to review rather than
        silently converted into an automatic maintenance instruction.
        """
    ),
    code(
        '''
        # Operational policy configuration. Replace these demonstration values through a formal
        # safety and maintenance-program approval process before any real deployment.
        from datetime import UTC, datetime
        from textwrap import dedent

        from sklearn.calibration import calibration_curve
        from sklearn.ensemble import IsolationForest
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss


        @dataclass(frozen=True)
        class OperationalConfig:
            interval_alpha: float = 0.10
            minimum_interval_coverage: float = 0.88
            persistent_cycles: int = 3
            persistence_window: int = 5
            critical_rul_cycles: int = 7
            expedited_rul_cycles: int = 15
            planned_rul_cycles: int = 30
            critical_risk: float = 0.80
            expedited_risk: float = 0.60
            scheduled_shop_capacity: int = 10
            ood_reference_quantile: float = 0.01
            psi_warning: float = 0.10
            psi_alert: float = 0.25
            maximum_ood_rate: float = 0.05

        OP_CONFIG = OperationalConfig()
        OP_OUTPUT_DIR = WORK_DIR / "operational_artifacts"
        OP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(OP_CONFIG)
        '''
    ),
    markdown("## 10. Telemetry contract and pre-inference data-quality gate"),
    code(
        '''
        def telemetry_quality_report(
            reference: pd.DataFrame,
            current: pd.DataFrame,
        ) -> pd.DataFrame:
            required = set(COLUMNS)
            missing_columns = sorted(required - set(current.columns))
            duplicate_keys = int(
                current.duplicated(["unit_number", "time_in_cycles"]).sum()
            )
            non_monotonic_engines = 0
            for _, trajectory in current.groupby("unit_number", sort=False):
                if (trajectory.time_in_cycles.diff().dropna() <= 0).any():
                    non_monotonic_engines += 1

            signals = select_feature_columns(reference)
            lower = reference[signals].quantile(0.001)
            upper = reference[signals].quantile(0.999)
            outside = ((current[signals] < lower) | (current[signals] > upper)).sum().sum()
            cells = max(1, current[signals].size)
            checks = [
                {
                    "check": "required_schema",
                    "value": len(missing_columns),
                    "status": "FAIL" if missing_columns else "PASS",
                    "detail": ", ".join(missing_columns) or "All required columns present",
                },
                {
                    "check": "missing_values",
                    "value": int(current.isna().sum().sum()),
                    "status": "FAIL" if current.isna().any().any() else "PASS",
                    "detail": "Null telemetry cells",
                },
                {
                    "check": "duplicate_engine_cycles",
                    "value": duplicate_keys,
                    "status": "FAIL" if duplicate_keys else "PASS",
                    "detail": "Duplicate engine/cycle keys",
                },
                {
                    "check": "cycle_monotonicity",
                    "value": non_monotonic_engines,
                    "status": "FAIL" if non_monotonic_engines else "PASS",
                    "detail": "Engines with non-increasing cycle order",
                },
                {
                    "check": "reference_range_rate",
                    "value": float(outside / cells),
                    "status": "WARN" if outside / cells > 0.01 else "PASS",
                    "detail": "Share outside training 0.1%-99.9% ranges",
                },
            ]
            return pd.DataFrame(checks)

        quality_report = telemetry_quality_report(train_raw, test_raw)
        display(quality_report)
        hard_failures = quality_report.query("status == 'FAIL'")
        if not hard_failures.empty:
            raise RuntimeError(
                "Telemetry quality gate failed; inference is blocked until data is corrected."
            )
        '''
    ),
    markdown(
        """
        ## 11. Split-conformal RUL uncertainty

        A deployment regressor is trained only on development engines. Absolute residuals from
        untouched calibration engines define a finite-sample 90% split-conformal interval. This is
        more defensible than treating variation among random-forest trees as calibrated uncertainty.
        Official FD001 truth is used below only to evaluate coverage, never to construct intervals.
        """
    ),
    code(
        '''
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=CONFIG.validation_size,
            random_state=CONFIG.random_state,
        )
        development_indices, assurance_indices = next(
            splitter.split(train_engineered, groups=train_engineered.unit_number)
        )
        development = train_engineered.iloc[development_indices].copy()
        assurance = train_engineered.iloc[assurance_indices].copy()
        assurance_points = validation_checkpoints(assurance)

        deployment_regressor = candidate_regressors(CONFIG)[
            result.metrics["selected_regressor"]
        ]
        deployment_regressor.fit(
            development[result.feature_columns], development.rul
        )
        assurance_prediction = np.clip(
            deployment_regressor.predict(assurance_points[result.feature_columns]),
            0,
            None,
        )
        residuals = np.abs(
            assurance_points.rul_raw.to_numpy() - assurance_prediction
        )
        sample_size = len(residuals)
        quantile_level = min(
            1.0,
            math.ceil((sample_size + 1) * (1 - OP_CONFIG.interval_alpha))
            / sample_size,
        )
        interval_radius = float(
            np.quantile(residuals, quantile_level, method="higher")
        )

        operational_endpoints = test_engineered.groupby(
            "unit_number", sort=True
        ).tail(1)
        operational_predictions = pd.DataFrame(
            {
                "unit_number": operational_endpoints.unit_number.to_numpy(dtype=int),
                "cycle": operational_endpoints.time_in_cycles.to_numpy(dtype=int),
                "actual_rul_evaluation_only": test_rul.to_numpy(dtype=float),
            }
        )
        operational_predictions["predicted_rul"] = np.clip(
            deployment_regressor.predict(
                operational_endpoints[result.feature_columns]
            ),
            0,
            None,
        )
        operational_predictions["rul_lower_90"] = np.clip(
            operational_predictions.predicted_rul - interval_radius,
            0,
            None,
        )
        operational_predictions["rul_upper_90"] = (
            operational_predictions.predicted_rul + interval_radius
        )
        interval_coverage = float(
            (
                (
                    operational_predictions.actual_rul_evaluation_only
                    >= operational_predictions.rul_lower_90
                )
                & (
                    operational_predictions.actual_rul_evaluation_only
                    <= operational_predictions.rul_upper_90
                )
            ).mean()
        )
        interval_metrics = {
            "target_coverage": 1 - OP_CONFIG.interval_alpha,
            "test_coverage_evaluation_only": interval_coverage,
            "interval_radius_cycles": interval_radius,
            "mean_interval_width_cycles": float(2 * interval_radius),
            "calibration_points": sample_size,
        }
        display(pd.Series(interval_metrics, name="RUL uncertainty").to_frame())
        '''
    ),
    markdown(
        """
        ## 12. Calibrated maintenance probability and independent policy selection

        Assurance engines are split again by engine identity. One subset calibrates raw classifier
        scores with isotonic regression; the other selects the economic decision threshold. This
        avoids selecting and reporting the policy on the same engine checkpoints.
        """
    ),
    code(
        '''
        assurance_units = np.array(sorted(assurance.unit_number.unique()))
        random_generator = np.random.default_rng(CONFIG.random_state)
        random_generator.shuffle(assurance_units)
        split_point = max(1, len(assurance_units) // 2)
        calibration_units = set(assurance_units[:split_point])
        policy_units = set(assurance_units[split_point:])
        calibration_points = validation_checkpoints(
            assurance[assurance.unit_number.isin(calibration_units)]
        )
        policy_points = validation_checkpoints(
            assurance[assurance.unit_number.isin(policy_units)]
        )

        deployment_classifier = RandomForestClassifier(
            n_estimators=CONFIG.n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=CONFIG.n_jobs,
            random_state=CONFIG.random_state,
        )
        development_due = (
            development.rul_raw <= CONFIG.maintenance_horizon
        ).astype(int)
        deployment_classifier.fit(
            development[result.feature_columns], development_due
        )

        calibration_raw = deployment_classifier.predict_proba(
            calibration_points[result.feature_columns]
        )[:, 1]
        calibration_truth = (
            calibration_points.rul_raw <= CONFIG.maintenance_horizon
        ).astype(int)
        probability_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        probability_calibrator.fit(calibration_raw, calibration_truth)

        policy_raw = deployment_classifier.predict_proba(
            policy_points[result.feature_columns]
        )[:, 1]
        policy_probability = probability_calibrator.transform(policy_raw)
        policy_truth = (
            policy_points.rul_raw <= CONFIG.maintenance_horizon
        ).to_numpy()
        operational_threshold, policy_value = best_threshold(
            policy_truth, policy_probability, CONFIG
        )

        endpoint_raw_probability = deployment_classifier.predict_proba(
            operational_endpoints[result.feature_columns]
        )[:, 1]
        operational_predictions["raw_maintenance_probability"] = (
            endpoint_raw_probability
        )
        operational_predictions["calibrated_maintenance_probability"] = (
            probability_calibrator.transform(endpoint_raw_probability)
        )
        operational_predictions["maintenance_actual_evaluation_only"] = (
            test_rul.to_numpy() <= CONFIG.maintenance_horizon
        ).astype(int)

        operational_decision = (
            operational_predictions.calibrated_maintenance_probability
            >= operational_threshold
        ).astype(int)
        operational_value = maintenance_value(
            operational_predictions.maintenance_actual_evaluation_only,
            operational_decision,
            CONFIG,
        )
        calibration_metrics = {
            "operational_threshold": operational_threshold,
            "calibration_engines": len(calibration_units),
            "policy_engines": len(policy_units),
            "calibration_brier_raw": float(
                brier_score_loss(calibration_truth, calibration_raw)
            ),
            "calibration_brier_isotonic": float(
                brier_score_loss(
                    calibration_truth,
                    probability_calibrator.transform(calibration_raw),
                )
            ),
            "policy_expected_value": policy_value["expected_value"],
            "test_expected_value_evaluation_only": operational_value.expected_value,
        }
        display(pd.Series(calibration_metrics, name="Risk calibration").to_frame())

        raw_observed, raw_predicted = calibration_curve(
            calibration_truth, calibration_raw, n_bins=6, strategy="quantile"
        )
        calibrated_observed, calibrated_predicted = calibration_curve(
            calibration_truth,
            probability_calibrator.transform(calibration_raw),
            n_bins=6,
            strategy="quantile",
        )
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        axis.plot([0, 1], [0, 1], "--", color="#737d79", label="Ideal")
        axis.plot(raw_predicted, raw_observed, "o-", label="Raw classifier")
        axis.plot(
            calibrated_predicted,
            calibrated_observed,
            "o-",
            label="Isotonic calibrated",
        )
        axis.set(
            title="Maintenance-risk reliability",
            xlabel="Mean predicted probability",
            ylabel="Observed frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.legend()
        figure.tight_layout()
        plt.show()
        '''
    ),
    markdown("## 13. Out-of-distribution detection and population drift"),
    code(
        '''
        ood_training = development[result.feature_columns].sample(
            n=min(10_000, len(development)),
            random_state=CONFIG.random_state,
        )
        ood_model = IsolationForest(
            n_estimators=150,
            max_samples="auto",
            contamination="auto",
            n_jobs=CONFIG.n_jobs,
            random_state=CONFIG.random_state,
        )
        ood_model.fit(ood_training)
        reference_ood_score = ood_model.score_samples(ood_training)
        ood_threshold = float(
            np.quantile(reference_ood_score, OP_CONFIG.ood_reference_quantile)
        )
        endpoint_ood_score = ood_model.score_samples(
            operational_endpoints[result.feature_columns]
        )
        operational_predictions["ood_score"] = endpoint_ood_score
        operational_predictions["out_of_distribution"] = (
            endpoint_ood_score < ood_threshold
        )

        def population_stability_index(
            reference_values: pd.Series,
            current_values: pd.Series,
            bins: int = 10,
        ) -> float:
            quantiles = np.linspace(0, 1, bins + 1)
            internal = np.unique(
                reference_values.quantile(quantiles[1:-1]).to_numpy()
            )
            edges = np.concatenate(([-np.inf], internal, [np.inf]))
            if len(edges) < 3:
                return 0.0
            reference_count, _ = np.histogram(reference_values, bins=edges)
            current_count, _ = np.histogram(current_values, bins=edges)
            reference_share = np.clip(
                reference_count / max(1, reference_count.sum()), 1e-6, None
            )
            current_share = np.clip(
                current_count / max(1, current_count.sum()), 1e-6, None
            )
            return float(
                np.sum(
                    (current_share - reference_share)
                    * np.log(current_share / reference_share)
                )
            )

        if hasattr(deployment_regressor, "feature_importances_"):
            drift_features = (
                pd.Series(
                    deployment_regressor.feature_importances_,
                    index=result.feature_columns,
                )
                .sort_values(ascending=False)
                .head(12)
                .index.tolist()
            )
        else:
            drift_features = result.feature_columns[:12]

        drift_records = []
        for feature in drift_features:
            psi = population_stability_index(
                development[feature], test_engineered[feature]
            )
            status = (
                "ALERT"
                if psi >= OP_CONFIG.psi_alert
                else "WARN"
                if psi >= OP_CONFIG.psi_warning
                else "PASS"
            )
            drift_records.append({"feature": feature, "psi": psi, "status": status})
        drift_report = pd.DataFrame(drift_records).sort_values(
            "psi", ascending=False
        )
        ood_rate = float(operational_predictions.out_of_distribution.mean())
        display(drift_report.style.format({"psi": "{:.3f}"}))
        print(f"Endpoint OOD rate: {ood_rate:.1%} (threshold {ood_threshold:.4f})")
        '''
    ),
    markdown(
        """
        ## 14. Persistent alerts, safety overrides, and maintenance capacity

        One noisy cycle should not create routine work. Planned maintenance requires a persistent
        threshold breach, while low conservative RUL bounds and very high risk bypass persistence.
        OOD engines are never trusted automatically and go to engineering review. Critical safety
        cases also bypass nominal shop capacity; lower-priority work is explicitly backlogged.
        """
    ),
    code(
        '''
        all_raw_probability = deployment_classifier.predict_proba(
            test_engineered[result.feature_columns]
        )[:, 1]
        risk_history = test_engineered[
            ["unit_number", "time_in_cycles"]
        ].copy()
        risk_history["calibrated_probability"] = (
            probability_calibrator.transform(all_raw_probability)
        )

        persistence_records = []
        for unit, history in risk_history.groupby("unit_number", sort=True):
            recent = history.calibrated_probability.tail(
                OP_CONFIG.persistence_window
            ).to_numpy()
            trailing_count = 0
            for breached in (recent >= operational_threshold)[::-1]:
                if not breached:
                    break
                trailing_count += 1
            persistence_records.append(
                {
                    "unit_number": int(unit),
                    "consecutive_alert_cycles": trailing_count,
                    "persistent_alert": (
                        trailing_count >= OP_CONFIG.persistent_cycles
                    ),
                }
            )
        persistence = pd.DataFrame(persistence_records)
        operational_predictions = operational_predictions.merge(
            persistence, on="unit_number", validate="one_to_one"
        )

        def triage_engine(row: pd.Series) -> tuple[str, str]:
            if row.out_of_distribution:
                return "MANUAL_REVIEW", "Out-of-distribution safety override"
            if (
                row.rul_lower_90 <= OP_CONFIG.critical_rul_cycles
                or row.calibrated_maintenance_probability >= OP_CONFIG.critical_risk
            ):
                return "CRITICAL", "Conservative RUL or risk crossed critical limit"
            if (
                row.rul_lower_90 <= OP_CONFIG.expedited_rul_cycles
                or row.calibrated_maintenance_probability >= OP_CONFIG.expedited_risk
            ):
                return "EXPEDITED", "Short conservative RUL or elevated calibrated risk"
            if row.persistent_alert and (
                row.rul_lower_90 <= OP_CONFIG.planned_rul_cycles
                or row.calibrated_maintenance_probability >= operational_threshold
            ):
                return "PLANNED", "Persistent maintenance threshold breach"
            if (
                row.rul_lower_90 <= OP_CONFIG.planned_rul_cycles
                or row.calibrated_maintenance_probability >= operational_threshold
            ):
                return "MONITOR", "Single-cycle signal awaiting persistence"
            return "ROUTINE", "No operational trigger"

        triage = operational_predictions.apply(triage_engine, axis=1)
        operational_predictions["triage"] = [item[0] for item in triage]
        operational_predictions["reason"] = [item[1] for item in triage]
        operational_predictions["priority_score"] = (
            100 * operational_predictions.calibrated_maintenance_probability
            + 2
            * np.maximum(
                0,
                OP_CONFIG.planned_rul_cycles
                - operational_predictions.rul_lower_90,
            )
            + 5 * operational_predictions.consecutive_alert_cycles
            + 50 * operational_predictions.out_of_distribution.astype(int)
        )

        operational_predictions["capacity_status"] = "Not requested"
        operational_predictions.loc[
            operational_predictions.triage == "MANUAL_REVIEW", "capacity_status"
        ] = "Engineering review"
        operational_predictions.loc[
            operational_predictions.triage == "CRITICAL", "capacity_status"
        ] = "Safety override"

        schedulable = operational_predictions[
            operational_predictions.triage.isin(["EXPEDITED", "PLANNED"])
        ].sort_values("priority_score", ascending=False)
        assigned_ids = set(
            schedulable.head(OP_CONFIG.scheduled_shop_capacity).unit_number
        )
        backlog_ids = set(schedulable.unit_number) - assigned_ids
        operational_predictions.loc[
            operational_predictions.unit_number.isin(assigned_ids),
            "capacity_status",
        ] = "Assigned"
        operational_predictions.loc[
            operational_predictions.unit_number.isin(backlog_ids),
            "capacity_status",
        ] = "Capacity backlog"

        lead_time = {
            "MANUAL_REVIEW": 0,
            "CRITICAL": 0,
            "EXPEDITED": 3,
            "PLANNED": 14,
            "MONITOR": 30,
            "ROUTINE": np.nan,
        }
        operational_predictions["target_action_within_cycles"] = (
            operational_predictions.triage.map(lead_time)
        )
        operational_queue = operational_predictions.sort_values(
            ["priority_score", "calibrated_maintenance_probability"],
            ascending=False,
        )
        display(
            operational_queue[
                [
                    "unit_number",
                    "cycle",
                    "rul_lower_90",
                    "predicted_rul",
                    "calibrated_maintenance_probability",
                    "consecutive_alert_cycles",
                    "out_of_distribution",
                    "triage",
                    "capacity_status",
                    "reason",
                ]
            ].head(30).style.format(
                {
                    "rul_lower_90": "{:.1f}",
                    "predicted_rul": "{:.1f}",
                    "calibrated_maintenance_probability": "{:.1%}",
                }
            )
        )

        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        operational_predictions.triage.value_counts().plot.bar(
            ax=axes[0], color="#356a99"
        )
        axes[0].set(title="Operational triage", xlabel="Action", ylabel="Engines")
        axes[0].tick_params(axis="x", rotation=35)
        axes[1].errorbar(
            operational_predictions.predicted_rul,
            operational_predictions.unit_number,
            xerr=interval_radius,
            fmt="o",
            markersize=3,
            alpha=0.55,
            color="#287271",
        )
        axes[1].axvline(
            OP_CONFIG.planned_rul_cycles,
            color="#b5473e",
            linestyle="--",
            label="Planning horizon",
        )
        axes[1].set(
            title="Fleet RUL with 90% uncertainty",
            xlabel="Remaining cycles",
            ylabel="Engine",
        )
        axes[1].legend()
        figure.tight_layout()
        plt.show()
        '''
    ),
    markdown("## 15. Release gate, audit trail, monitoring report, and model card"),
    code(
        '''
        severe_drift = bool((drift_report.status == "ALERT").any())
        excessive_ood = ood_rate > OP_CONFIG.maximum_ood_rate
        release_reasons = []
        if severe_drift:
            release_reasons.append("Population drift alert")
        if excessive_ood:
            release_reasons.append("OOD rate exceeds limit")
        if interval_coverage < OP_CONFIG.minimum_interval_coverage:
            release_reasons.append("RUL interval coverage below release limit")
        if not hard_failures.empty:
            release_reasons.append("Telemetry contract failure")
        release_status = (
            "HOLD_AUTOMATED_RECOMMENDATIONS"
            if release_reasons
            else "HUMAN_REVIEW_ENABLED"
        )

        run_timestamp = datetime.now(UTC).isoformat()
        data_fingerprint = {
            filename: sha256(DATA_DIR / filename) for filename in FD001_SHA256
        }
        run_material = json.dumps(
            {
                "timestamp": run_timestamp,
                "experiment": asdict(CONFIG),
                "operations": asdict(OP_CONFIG),
                "data": data_fingerprint,
            },
            sort_keys=True,
        ).encode()
        run_id = hashlib.sha256(run_material).hexdigest()[:16]

        operational_bundle_path = OP_OUTPUT_DIR / "operational_models.joblib"
        joblib.dump(
            {
                "rul_model": deployment_regressor,
                "maintenance_model": deployment_classifier,
                "probability_calibrator": probability_calibrator,
                "ood_model": ood_model,
                "feature_columns": result.feature_columns,
                "rul_interval_radius": interval_radius,
                "maintenance_threshold": operational_threshold,
                "ood_threshold": ood_threshold,
                "experiment_config": asdict(CONFIG),
                "operational_config": asdict(OP_CONFIG),
                "run_id": run_id,
            },
            operational_bundle_path,
        )
        model_sha256 = sha256(operational_bundle_path)

        operational_predictions.to_csv(
            OP_OUTPUT_DIR / "evaluation_with_operational_decisions.csv",
            index=False,
        )
        operational_queue.drop(
            columns=[
                "actual_rul_evaluation_only",
                "maintenance_actual_evaluation_only",
            ]
        ).to_csv(OP_OUTPUT_DIR / "live_work_queue.csv", index=False)
        quality_report.to_csv(OP_OUTPUT_DIR / "data_quality_report.csv", index=False)
        drift_report.to_csv(OP_OUTPUT_DIR / "drift_report.csv", index=False)

        monitoring_report = {
            "run_id": run_id,
            "created_at": run_timestamp,
            "release_status": release_status,
            "release_reasons": release_reasons,
            "model_sha256": model_sha256,
            "data_fingerprint": data_fingerprint,
            "data_quality": quality_report.to_dict(orient="records"),
            "uncertainty": interval_metrics,
            "calibration": calibration_metrics,
            "drift": drift_report.to_dict(orient="records"),
            "ood_rate": ood_rate,
            "triage_counts": {
                str(key): int(value)
                for key, value in operational_predictions.triage.value_counts().items()
            },
            "capacity_counts": {
                str(key): int(value)
                for key, value in operational_predictions.capacity_status.value_counts().items()
            },
        }
        (OP_OUTPUT_DIR / "monitoring_report.json").write_text(
            json.dumps(monitoring_report, indent=2) + "\\n",
            encoding="utf-8",
        )

        decision_log_path = OP_OUTPUT_DIR / "decision_log.jsonl"
        with decision_log_path.open("w", encoding="utf-8") as decision_log:
            for record in operational_queue.drop(
                columns=[
                    "actual_rul_evaluation_only",
                    "maintenance_actual_evaluation_only",
                ]
            ).to_dict(orient="records"):
                record["run_id"] = run_id
                record["created_at"] = run_timestamp
                decision_log.write(json.dumps(record) + "\\n")

        test_regression = regression_metrics(
            operational_predictions.actual_rul_evaluation_only.to_numpy(),
            operational_predictions.predicted_rul.to_numpy(),
        )
        model_card = f"""# Operational C-MAPSS Model Card

        ## Status
        - Release gate: {release_status}
        - Run ID: {run_id}
        - Model SHA-256: {model_sha256}
        - Approval: RESEARCH PROTOTYPE ONLY

        ## Intended use
        Human-reviewed fleet maintenance decision support for FD001-like research data.
        It must not independently determine airworthiness or release an aircraft to service.

        ## Performance on official FD001 endpoints
        - MAE: {test_regression.mae:.3f} cycles
        - RMSE: {test_regression.rmse:.3f} cycles
        - R2: {test_regression.r2:.3f}
        - NASA score: {test_regression.nasa_score:.3f}
        - 90% interval coverage: {interval_coverage:.1%}

        ## Operational controls
        - Split-conformal RUL intervals
        - Isotonic maintenance-risk calibration
        - Engine-independent calibration and policy subsets
        - OOD routing to engineering review
        - PSI population-drift monitoring
        - Persistent alerts with critical safety overrides
        - Capacity-aware queue with explicit backlog
        - Versioned data, model, monitoring, and decision records

        ## Known limitations
        FD001 is simulated, has one operating condition and one degradation mode, and has no actual
        maintenance actions, component removals, weather, routes, loads, shop constraints, MEL/CDL,
        or maintenance-program task data. Cycles are not calendar time. Thresholds and costs are
        illustrative. Real deployment requires approved fleet data, hazard assessment, independent
        verification, cybersecurity, configuration control, human-factors validation, and authority
        acceptance under the operator's maintenance and IAHM programs.
        """
        (OP_OUTPUT_DIR / "MODEL_CARD.md").write_text(
            dedent(model_card).strip() + "\\n", encoding="utf-8"
        )

        operational_archive = shutil.make_archive(
            str(WORK_DIR / "cmapss_operational_package"),
            "zip",
            root_dir=OP_OUTPUT_DIR,
        )
        print(f"Release gate: {release_status}")
        print("Reasons:", release_reasons or ["All prototype gates passed"])
        print(f"Operational package: {operational_archive}")
        for path in sorted(OP_OUTPUT_DIR.iterdir()):
            print(f"  {path.name:<45} {path.stat().st_size / 1024:>10.1f} KiB")

        if in_colab:
            print("Colab download command:")
            print(
                "from google.colab import files; files.download(r'"
                + operational_archive
                + "')"
            )
        '''
    ),
    markdown(
        """
        ## 16. What remains before a real aircraft operation

        This prototype now has a credible *shape* for operational decision support, but real-life
        advancement depends more on approved evidence and integration than additional algorithms.

        1. Define the intended function, failure conditions, maintenance authority, and human roles.
        2. Replace FD001 with governed fleet data covering tails, engines, routes, environments,
           maintenance actions, removals, findings, censored outcomes, and sensor configuration.
        3. Establish traceable labels and prevent post-maintenance and cross-tail leakage.
        4. Validate uncertainty and calibration prospectively on unseen fleets and time periods.
        5. Integrate with the operator's approved maintenance program, planning system, electronic
           technical log, parts availability, and engineering-order workflow.
        6. Define alert ownership, response times, escalation, override, appeal, incident response,
           fallback operation, rollback, and model decommissioning.
        7. Perform independent verification, cybersecurity and privacy assessment, human-factors
           trials, safety assessment, configuration control, and regulatory engagement.
        8. Run in shadow mode before any recommendation affects a maintenance plan.

        The model must remain advisory until those controls and approvals exist. A high benchmark
        score is not evidence of airworthiness suitability.
        """
    ),
]

notebook.cells = base_cells + operational_cells
notebook.metadata["operational_profile"] = {
    "assurance": "research-prototype",
    "human_in_the_loop": True,
    "automatic_airworthiness_decision": False,
}

output = Path("notebooks/03_operational_cmapss_decision_support.ipynb")
nbformat.write(notebook, output)
print(f"Wrote {output} with {len(notebook.cells)} cells")
