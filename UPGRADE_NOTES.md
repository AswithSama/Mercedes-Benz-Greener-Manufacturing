# Upgrade notes and verification

This ZIP overlays an existing `Mercedes-Benz-Greener-Manufacturing` clone. It does not include Kaggle datasets or trained artifacts. Do not copy an old `model.pkl`: it lacks the new integrated preprocessing and must be regenerated.

## Changed code

- `src/components/data_ingestion.py`: optional Kaggle download; local data no longer requires Kaggle authentication.
- `src/components/data_transformation.py`: fitted, train-only feature selection and reusable preprocessing.
- `src/components/model_trainer.py`: train/holdout separation, IQR filter fit to training targets only, Random Forest and baseline evaluation.
- `src/pipeline/train_pipeline.py`: full CLI training and evaluation orchestration.
- `src/pipeline/predict_pipeline.py`: single-artifact schema-safe inference.
- `src/pipeline/batch_predict.py`: competition CSV prediction CLI.
- `application.py`, `templates/`: upload, preview, one-time downloadable predictions and friendly errors.
- `src/utils.py`, `src/logger.py`, `src/exception.py`: utilities simplified/fixed.
- `tests/`: synthetic training and Flask integration tests.
- `README.md`, `DEEPDIVE.md`: accurate documentation of implemented and unverified pieces.
- `.github/workflows/...`: data-free test workflow, rather than automatic cloud deployment.
- `.gitignore`: exclude local data, trained models and artifacts.

## Verification in the preparation environment

- `python -m pytest -q`: **3 passed, 1 skipped**. The Flask test was skipped because Flask could not be installed in this offline environment.
- `python -m compileall -q application.py src tests`: passed.
- Full CLI smoke test using synthetic Kaggle-shaped CSVs: train, save, holdout evaluation, export competition CSV, reload artifact from a separate Python process: passed.
- Actual Kaggle data and cloud deployment: **not tested**. No real competition scores are included.

## Start on your Mac

```bash
cd ~/Desktop/Mercedes-Benz-Greener-Manufacturing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python -m src.pipeline.train_pipeline --train-csv notebook/data/train.csv --test-csv notebook/data/test.csv
python -m src.pipeline.batch_predict --input-csv notebook/data/test.csv
python application.py
```

Review the staged changes before committing. The included archive has no Kaggle CSVs, credentials, pickle files, Python caches or exploratory CatBoost log files.
