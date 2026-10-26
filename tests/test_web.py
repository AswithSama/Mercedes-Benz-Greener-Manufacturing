import io

import pandas as pd
import pytest

pytest.importorskip("flask")
from application import create_app
from src.pipeline.train_pipeline import run_training


def test_upload_preview_download_and_validation(sample_data, tmp_path):
    train_csv = tmp_path / "train.csv"
    sample_data.to_csv(train_csv, index=False)
    artifacts = tmp_path / "artifacts"
    run_training(train_csv, output_dir=artifacts, n_estimators=10)

    app = create_app(artifacts / "model.pkl")
    app.config.update(TESTING=True, PREDICTIONS_DIR=tmp_path / "downloads")
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/download_file").status_code == 302
    features = sample_data.drop(columns=["y"]).iloc[:4]
    res = client.post("/upload_file", data={"file": (io.BytesIO(features.to_csv(index=False).encode()), "test.csv")})
    assert res.status_code == 200
    assert b"Processed 4 rows" in res.data

    exported = client.get("/download_file")
    assert exported.status_code == 200
    actual = pd.read_csv(io.BytesIO(exported.data))
    assert actual.columns.tolist() == ["ID", "Prediction"]
    assert actual["ID"].tolist() == features["ID"].tolist()
    assert len(actual) == 4
    assert client.get("/download_file").status_code == 302

    invalid = features.drop(columns=["X10"])
    res = client.post("/upload_file", data={"file": (io.BytesIO(invalid.to_csv(index=False).encode()), "test.csv")})
    assert res.status_code == 400
    assert b"missing" in res.data
    res = client.post("/upload_file", data={"file": (io.BytesIO(b"not csv"), "file.txt")})
    assert res.status_code == 400
