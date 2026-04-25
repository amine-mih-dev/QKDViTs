from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _list_files(directory: Path, suffixes: Sequence[str]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {tuple(suffixes)} found in {directory}"
        )

    return sorted(files)


def _coerce_label(value):
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass

    if isinstance(value, str):
        token = value.strip()
        if token.startswith("tensor(") and token.endswith(")"):
            token = token[7:-1]
        token = token.split(",", 1)[0]
        try:
            return int(float(token))
        except ValueError:
            return value

    return value


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.map(_coerce_label)


def load_fold_csv_directory(input_dir: str | Path) -> pd.DataFrame:
    directory = Path(input_dir)
    csv_paths = _list_files(directory, (".csv",))
    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    if "labels" in df.columns:
        df["labels"] = normalize_label_series(df["labels"])

    return df


def load_fold_pickle_directory(input_dir: str | Path) -> pd.DataFrame:
    directory = Path(input_dir)
    pkl_paths = _list_files(directory, (".pkl", ".pickle"))
    frames = [pd.read_pickle(path) for path in pkl_paths]
    df = pd.concat(frames, ignore_index=True)

    if "labels" in df.columns:
        df["labels"] = normalize_label_series(df["labels"])

    if "Image_ID" in df.columns:
        df["Image_ID"] = df["Image_ID"].map(
            lambda value: value.split("___", 1)[1]
            if isinstance(value, str) and "___" in value
            else value
        )

    return df
