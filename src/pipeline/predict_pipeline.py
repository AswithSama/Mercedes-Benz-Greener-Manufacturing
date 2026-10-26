"""Inference always applies the exact fitted preprocessing used at training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import load_object


class PredictPipeline:
    def __init__(self, model_path: str | Path = "artifacts/model.pkl"):
        self.model_path = Path(model_path)
        self._model = None

    def predict(self, features: pd.DataFrame):
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("Upload a non-empty CSV with the original feature headers.")
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Trained pipeline not found: {self.model_path}. "
                    "Run python -m src.pipeline.train_pipeline first."
                )
            self._model = load_object(self.model_path)
        if not hasattr(self._model, "named_steps") or "preprocessor" not in self._model.named_steps:
            raise ValueError(
                "Saved model uses an old incompatible format. "
                "Retrain with python -m src.pipeline.train_pipeline."
            )
        return self._model.predict(features)
