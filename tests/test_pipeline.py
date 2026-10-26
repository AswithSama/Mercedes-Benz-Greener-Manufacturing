import json

import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import FeatureCleaner
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import run_training


def test_feature_cleaner_uses_training_features_only(sample_data):
    train = sample_data.drop(columns=["y"])
    cleaner = FeatureCleaner().fit(train)
    assert "ID" not in cleaner.feature_columns_
    assert "X4" not in cleaner.feature_columns_
    assert "X12" not in cleaner.feature_columns_
    shuffled = train[["X11", "X10", "ID", "X4", "X0", "X1", "X12"]]
    assert cleaner.transform(shuffled).columns.tolist() == cleaner.feature_columns_
    with pytest.raises(ValueError, match="missing"):
        cleaner.transform(train.drop(columns=["X10"]))


def test_train_evaluate_save_and_infer(sample_data, tmp_path):
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    train_csv = data_dir / "train.csv"
    sample_data.to_csv(train_csv, index=False)
    competition_csv = data_dir / "test.csv"
    inference = sample_data.drop(columns=["y"]).iloc[:6].copy()
    inference.loc[inference.index[0], "X0"] = "never-before-seen"
    inference.to_csv(competition_csv, index=False)

    artifacts = tmp_path / "artifacts"
    metrics = run_training(train_csv, competition_csv, output_dir=artifacts, n_estimators=12)
    assert metrics["holdout_rows"] > 0
    assert metrics["training_rows_after_filter"] < len(sample_data)
    assert all(np.isfinite(v) for v in metrics["model_holdout"].values())
    assert (artifacts / "model.pkl").is_file()
    assert (artifacts / "metrics.json").is_file()
    assert (artifacts / "holdout_predictions.csv").is_file()
    assert (artifacts / "competition_predictions.csv").is_file()
    saved = json.loads((artifacts / "metrics.json").read_text())
    assert saved["holdout_rows"] == metrics["holdout_rows"]

    predictor = PredictPipeline(artifacts / "model.pkl")
    predictions = predictor.predict(inference)
    assert len(predictions) == 6
    assert np.isfinite(predictions).all()
    assert np.allclose(predictions, predictor.predict(inference[inference.columns[::-1]]))
    assert pd.read_csv(artifacts / "competition_predictions.csv").columns.tolist() == ["ID", "y"]

    with pytest.raises(ValueError, match="missing"):
        predictor.predict(inference.drop(columns=["X10"]))


def test_missing_data_or_model(sample_data, tmp_path):
    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        run_training(tmp_path / "not-here.csv")
    with pytest.raises(FileNotFoundError, match="Run python"):
        PredictPipeline(tmp_path / "absent.pkl").predict(sample_data.drop(columns=["y"]))
