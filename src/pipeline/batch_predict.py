"""Generate a competition-style CSV without launching the Flask interface.

python -m src.pipeline.batch_predict --input-csv notebook/data/test.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline


def run_batch_prediction(input_csv: str | Path, output_csv: str | Path,
                         model_path: str | Path = "artifacts/model.pkl") -> Path:
    frame = pd.read_csv(input_csv)
    if "ID" not in frame.columns:
        raise ValueError("For competition-style output, the input CSV must include an ID column.")
    predictions = PredictPipeline(model_path).predict(frame)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": frame["ID"].to_numpy(), "y": predictions}).to_csv(output, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for a Kaggle-format test CSV")
    parser.add_argument("--input-csv", default="notebook/data/test.csv")
    parser.add_argument("--output-csv", default="artifacts/competition_predictions.csv")
    parser.add_argument("--model-path", default="artifacts/model.pkl")
    args = parser.parse_args()
    saved = run_batch_prediction(args.input_csv, args.output_csv, args.model_path)
    print(f"Predictions saved to {saved}")


if __name__ == "__main__":
    main()
