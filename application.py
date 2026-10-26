"""CSV prediction demo for the trained Mercedes-Benz regression pipeline.

Run from the repository root: python application.py
Production WSGI entry point: application:application
"""

from __future__ import annotations

import io
import os
import re
import secrets
import tempfile
import time
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from src.pipeline.predict_pipeline import PredictPipeline


def create_app(model_path: str | Path = "artifacts/model.pkl") -> Flask:
    web = Flask(__name__)
    web.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        PREDICTIONS_DIR=Path(tempfile.gettempdir()) / "mercedes-benz-predictions",
        MODEL_PATH=str(model_path),
    )

    @web.get("/")
    def index():
        return render_template("index.html")

    @web.post("/upload_file")
    def upload_file():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return render_template("index.html", error="Please choose a CSV file."), 400
        if not upload.filename.lower().endswith(".csv"):
            return render_template("index.html", error="Only .csv files are supported."), 400

        try:
            frame = pd.read_csv(upload.stream)
            if frame.empty:
                raise ValueError("The CSV contains no data rows.")
            predictions = PredictPipeline(web.config["MODEL_PATH"]).predict(frame)
            result = pd.DataFrame({
                **({"ID": frame["ID"].to_numpy()} if "ID" in frame.columns else {}),
                "Prediction": predictions,
            })
        except (ValueError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            return render_template("index.html", error=str(exc)), 400

        # Only a random filename is stored in the signed session cookie, never
        # the potentially large CSV or an arbitrary client-provided path.
        directory = Path(web.config["PREDICTIONS_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        _purge_stale_outputs(directory)
        token = secrets.token_hex(16)
        (directory / f"{token}.csv").write_bytes(result.to_csv(index=False).encode("utf-8"))
        session["prediction_token"] = token
        return render_template(
            "results.html", preview=result.head(10).to_dict(orient="records"),
            row_count=len(result), has_id="ID" in result.columns,
        )

    @web.get("/download_file")
    def download_file():
        token = session.get("prediction_token", "")
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            return redirect(url_for("index"))
        path = Path(web.config["PREDICTIONS_DIR"]) / f"{token}.csv"
        if not path.is_file():
            session.pop("prediction_token", None)
            return redirect(url_for("index"))
        content = path.read_bytes()
        path.unlink(missing_ok=True)
        session.pop("prediction_token", None)
        return send_file(
            io.BytesIO(content), mimetype="text/csv", as_attachment=True,
            download_name="predictions.csv",
        )

    @web.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return render_template("index.html", error="CSV exceeds the 10 MB upload limit."), 413

    return web


def _purge_stale_outputs(directory: Path, max_age_seconds: int = 3600) -> None:
    cutoff = time.time() - max_age_seconds
    for path in directory.glob("*.csv"):
        if re.fullmatch(r"[0-9a-f]{32}\.csv", path.name):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue


application = create_app()
app = application

if __name__ == "__main__":
    application.run(host="127.0.0.1", port=5000, debug=False)
