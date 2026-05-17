"""Small persistence helpers for CSV and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def _to_json_cell(value) -> str:
    """Store list/dict-like cells as JSON so round-tripping stays predictable."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return json.dumps([])
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            return json.dumps(value)
    return json.dumps(value)


def _prepare_for_storage(dataframe: pd.DataFrame, json_columns: Iterable[str]) -> pd.DataFrame:
    """Serialize only the columns that need JSON-safe storage."""

    df = dataframe.copy()
    for column in json_columns:
        if column in df:
            df[column] = df[column].map(_to_json_cell)
    return df


def save_dataframe(dataframe: pd.DataFrame, path: Path, json_columns: Iterable[str] = ()) -> Path:
    """Persist a dataframe to CSV after normalizing nested columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_for_storage(dataframe, json_columns)
    prepared.to_csv(path, index=False)
    return path


def save_json(payload: dict, path: Path) -> Path:
    """Persist a JSON artifact with stable indentation for inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
