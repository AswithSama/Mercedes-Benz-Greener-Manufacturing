"""Optional Kaggle download for the original competition data.

Local train/test CSV files are supported without Kaggle credentials.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
from io import BytesIO


class DataIngestion:
    competition = "mercedes-benz-greener-manufacturing"

    @classmethod
    def download_from_kaggle(cls, data_dir: str | Path = "notebook/data") -> tuple[Path, Path]:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise RuntimeError("Install the Kaggle client: python -m pip install kaggle") from exc

        directory = Path(data_dir)
        directory.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()  # only when explicitly requested
        api.competition_download_files(cls.competition, path=str(directory), quiet=False)
        archive = directory / f"{cls.competition}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"Kaggle archive not found at {archive}")

        with ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            for name in ("train.csv", "test.csv"):
                direct = [m for m in members if Path(m.filename).name == name]
                nested = [m for m in members if Path(m.filename).name == name + ".zip"]
                if len(direct) == 1:
                    content = zf.read(direct[0])
                elif len(nested) == 1:
                    # The competition has also been distributed as nested
                    # train.csv.zip / test.csv.zip files.
                    with ZipFile(BytesIO(zf.read(nested[0]))) as inner:
                        inner_members = [m for m in inner.infolist()
                                         if Path(m.filename).name == name and not m.is_dir()]
                        if len(inner_members) != 1:
                            raise ValueError(f"Expected {name} inside {nested[0].filename}")
                        content = inner.read(inner_members[0])
                else:
                    raise ValueError(f"Cannot locate {name} in {archive}")
                (directory / name).write_bytes(content)

        return directory / "train.csv", directory / "test.csv"
