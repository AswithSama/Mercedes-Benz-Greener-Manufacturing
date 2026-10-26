"""Run ingestion (optional), training, held-out evaluation and CSV inference.

python -m src.pipeline.train_pipeline --train-csv notebook/data/train.csv
python -m src.pipeline.train_pipeline --download-kaggle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


def run_training(train_csv: str | Path, test_csv: str | Path | None = None,
                 output_dir: str | Path = "artifacts", n_estimators: int = 256,
                 random_state: int = 42, test_size: float = 0.20,
                 filter_outliers: bool = True) -> dict:
    train_path = Path(train_csv)
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}. "
            "Provide --train-csv PATH or use --download-kaggle."
        )
    train = pd.read_csv(train_path)
    test = None
    if test_csv is not None:
        test_path = Path(test_csv)
        if not test_path.is_file():
            raise FileNotFoundError(f"Competition dataset not found: {test_path}")
        test = pd.read_csv(test_path)

    trainer = ModelTrainer(n_estimators=n_estimators, random_state=random_state,
                           test_size=test_size, filter_outliers=filter_outliers)
    return trainer.train(train, output_dir=output_dir, competition_data=test)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Mercedes-Benz manufacturing regression model")
    parser.add_argument("--train-csv", default="notebook/data/train.csv")
    parser.add_argument("--test-csv", default=None, help="Optional competition test CSV for export")
    parser.add_argument("--download-kaggle", action="store_true", help="Download the competition CSVs using your Kaggle credentials")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--n-estimators", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--no-outlier-filter", action="store_true")
    args = parser.parse_args()

    train_csv = args.train_csv
    test_csv = args.test_csv
    if args.download_kaggle:
        train_csv, downloaded_test = DataIngestion.download_from_kaggle()
        if test_csv is None:
            test_csv = downloaded_test
    elif test_csv is None:
        default_test = Path("notebook/data/test.csv")
        if default_test.exists():
            test_csv = default_test

    metrics = run_training(
        train_csv, test_csv, output_dir=args.output_dir,
        n_estimators=args.n_estimators, random_state=args.seed,
        test_size=args.holdout_fraction, filter_outliers=not args.no_outlier_filter,
    )
    print("Training and holdout evaluation complete.")
    print(json.dumps({
        "model_holdout": metrics["model_holdout"],
        "mean_baseline_holdout": metrics["mean_baseline_holdout"],
        "training_rows": metrics["training_rows_after_filter"],
        "holdout_rows": metrics["holdout_rows"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
