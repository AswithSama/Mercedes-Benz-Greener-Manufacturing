"""Leakage-safe, reusable transformations for manufacturing test-time data."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Learn the feature contract from *training rows only*.

    ID is an identifier, X4 was excluded in the original experiment, and y is
    the target. Near-constant binary columns are dropped using training data;
    inference rows cannot change that selection.
    """

    def __init__(self, dominance_threshold: float = 0.99):
        self.dominance_threshold = dominance_threshold

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Expected a pandas DataFrame with named feature columns.")
        if X.columns.has_duplicates:
            raise ValueError("Duplicate CSV column names are not supported.")
        if not 0.5 < self.dominance_threshold <= 1.0:
            raise ValueError("dominance_threshold must be in (0.5, 1.0].")

        candidates = X.drop(columns=["ID", "X4", "y"], errors="ignore")
        if candidates.empty or len(candidates.columns) == 0:
            raise ValueError("No usable input features were found.")

        dropped = []
        for name in candidates.select_dtypes(include="number").columns:
            values = candidates[name].dropna()
            if not values.empty and values.isin([0, 1]).all():
                if values.value_counts(normalize=True).max() > self.dominance_threshold:
                    dropped.append(name)

        self.dropped_features_ = dropped
        self.feature_columns_ = [name for name in candidates.columns if name not in dropped]
        if not self.feature_columns_:
            raise ValueError("All available features were removed during feature selection.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_columns_")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Expected a pandas DataFrame with named feature columns.")
        if X.columns.has_duplicates:
            raise ValueError("Duplicate CSV column names are not supported.")
        missing = [name for name in self.feature_columns_ if name not in X.columns]
        if missing:
            preview = ", ".join(missing[:12])
            suffix = " ..." if len(missing) > 12 else ""
            raise ValueError(f"CSV is missing {len(missing)} required feature column(s): {preview}{suffix}")
        return X.loc[:, self.feature_columns_].copy()


class DataTransformation:
    """Build a fitted-on-train-only pipeline for categorical and numeric data."""

    @staticmethod
    def create_preprocessor() -> Pipeline:
        numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        column_transformer = ColumnTransformer(
            transformers=[
                ("categorical", categorical, make_column_selector(dtype_include=["object", "category"])),
                ("numeric", numeric, make_column_selector(dtype_include="number")),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        return Pipeline(steps=[
            ("feature_cleaner", FeatureCleaner()),
            ("column_transformer", column_transformer),
        ])
