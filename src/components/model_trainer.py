"""Train the original Random Forest baseline with a valid holdout evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.components.data_transformation import DataTransformation
from src.utils import save_object


class ModelTrainer:
    def __init__(self, n_estimators: int = 256, random_state: int = 42,
                 test_size: float = 0.20, filter_outliers: bool = True):
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.filter_outliers = filter_outliers

    @staticmethod
    def _scores(y_true, predictions) -> dict:
        return {
            "r2": float(r2_score(y_true, predictions)),
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        }

    def train(self, data: pd.DataFrame, output_dir: str | Path = "artifacts",
              competition_data: pd.DataFrame | None = None) -> dict:
        if "y" not in data.columns:
            raise ValueError("Training CSV must have a numeric 'y' (test time) column.")
        if "ID" not in data.columns:
            raise ValueError("Training CSV must have the Kaggle 'ID' column.")
        if len(data) < 20:
            raise ValueError("Training CSV must contain at least 20 labeled observations.")
        if data.columns.has_duplicates:
            raise ValueError("Training CSV has duplicate columns.")

        target = pd.to_numeric(data["y"], errors="raise")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError("Training target 'y' contains missing or non-finite values.")

        features = data.drop(columns=["y"])
        X_fit, X_holdout, y_fit, y_holdout = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )
        before_filter = len(y_fit)
        if self.filter_outliers:
            q1, q3 = y_fit.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                keep = y_fit.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                X_fit, y_fit = X_fit.loc[keep], y_fit.loc[keep]
        if len(y_fit) < 10:
            raise ValueError("Too few training examples remain after outlier filtering.")

        model = Pipeline(steps=[
            ("preprocessor", DataTransformation.create_preprocessor()),
            ("regressor", RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])
        # Only X_fit/y_fit are used to fit the pipeline. Held-out examples do
        # not influence column selection, imputation, encoding, or the forest.
        model.fit(X_fit, y_fit)
        holdout_preds = model.predict(X_holdout)

        baseline = DummyRegressor(strategy="mean").fit(X_fit, y_fit)
        baseline_preds = baseline.predict(X_holdout)

        competition_predictions = None
        if competition_data is not None:
            if "ID" not in competition_data.columns:
                raise ValueError("Competition test CSV must have an 'ID' column.")
            # Validate the entire external schema before persisting artifacts.
            competition_predictions = model.predict(competition_data)

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_object(output / "model.pkl", model)

        cleaner = model.named_steps["preprocessor"].named_steps["feature_cleaner"]
        metrics = {
            "evaluation_protocol": "80/20 random holdout by default; fit and target-IQR filtering on training fold only",
            "seed": self.random_state,
            "test_fraction": self.test_size,
            "n_estimators": self.n_estimators,
            "labeled_rows_total": len(data),
            "training_rows_before_filter": before_filter,
            "training_rows_after_filter": len(y_fit),
            "holdout_rows": len(y_holdout),
            "excluded_input_fields": ["ID", "X4", "y"],
            "dropped_near_constant_binary_features": cleaner.dropped_features_,
            "model_holdout": self._scores(y_holdout, holdout_preds),
            "mean_baseline_holdout": self._scores(y_holdout, baseline_preds),
            "note": "Results depend on supplied data; saved model is the one assessed on the holdout. Competition test labels are not available.",
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame({
            "ID": X_holdout["ID"].to_numpy(),
            "actual_y": y_holdout.to_numpy(),
            "predicted_y": holdout_preds,
        }).to_csv(output / "holdout_predictions.csv", index=False)

        if competition_data is not None:
            pd.DataFrame({
                "ID": competition_data["ID"].to_numpy(),
                "y": competition_predictions,
            }).to_csv(output / "competition_predictions.csv", index=False)

        return metrics
