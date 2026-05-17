from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .cleaning import clean_property_dataframe


class ScraperAdapter(ABC):
    source_name: str

    @abstractmethod
    def fetch_raw_records(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def normalize_records(self) -> pd.DataFrame:
        return clean_property_dataframe(pd.DataFrame(self.fetch_raw_records()))


class MemoryScraperAdapter(ScraperAdapter):
    def __init__(self, source_name: str, records: list[dict[str, Any]]) -> None:
        self.source_name = source_name
        self._records = records

    def fetch_raw_records(self) -> list[dict[str, Any]]:
        normalized = []
        for record in self._records:
            item = dict(record)
            item.setdefault("source", self.source_name)
            normalized.append(item)
        return normalized
