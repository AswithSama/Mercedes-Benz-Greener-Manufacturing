"""Small Kaggle-shaped dataset for local tests; not real competition results."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(17)
    size = 72
    categories = rng.choice(["aa", "bb", "cc"], size=size)
    numeric = rng.uniform(0, 10, size=size)
    y = 90 + 2 * numeric + np.where(categories == "aa", 5, 0) + rng.normal(0, 1.2, size=size)
    return pd.DataFrame({
        "ID": np.arange(size) + 100,
        "X0": categories,
        "X1": rng.choice(["x", "y"], size=size),
        "X4": ["a"] * size,
        "X10": numeric,
        "X11": rng.integers(0, 2, size=size),
        "X12": np.zeros(size, dtype=int),
        "y": y,
    })
