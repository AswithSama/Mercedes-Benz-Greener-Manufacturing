"""Artifact serialization helpers. Only open locally trusted pickle files."""

from __future__ import annotations

import pickle
from pathlib import Path


def save_object(file_path, obj):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(obj, file_obj, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(file_path):
    path = Path(file_path)
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)
