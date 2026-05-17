"""Deterministic cleaning helpers for scraped property and client data."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd


DISTRICT_ALIASES = {
    "maseru": "Maseru",
    "leribe": "Leribe",
    "berea": "Berea",
    "mafeteng": "Mafeteng",
    "mohales hoek": "Mohale's Hoek",
    "mohale's hoek": "Mohale's Hoek",
    "butha-buthe": "Butha-Buthe",
    "quthing": "Quthing",
}


def _decode_nested_serialized_value(value: object, max_depth: int = 4) -> object:
    """Peel back values that were JSON-serialized more than once."""

    current = value
    for _ in range(max_depth):
        if isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                return ""
            if stripped.startswith(("[", "{", '"')) or stripped in {"null", "true", "false"}:
                try:
                    current = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    return current
        break
    return current


def _ensure_list(value: object) -> list[str]:
    """Normalize messy CSV/list fields into a clean list of strings."""

    decoded = _decode_nested_serialized_value(value)
    if isinstance(decoded, list):
        flattened: list[str] = []
        for item in decoded:
            flattened.extend(_ensure_list(item))
        deduplicated: list[str] = []
        seen: set[str] = set()
        for item in flattened:
            normalized = str(item).strip().strip('"').strip("'")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduplicated.append(normalized)
        return deduplicated
    if isinstance(decoded, str):
        stripped = decoded.strip()
        if not stripped:
            return []
        return [item.strip() for item in stripped.split("|") if item.strip()]
    if decoded is None:
        return []
    return [str(decoded)]


def normalize_district(district: object) -> str:
    """Map district aliases to one canonical lecturer-facing label."""

    cleaned = str(district or "").strip()
    key = cleaned.lower()
    return DISTRICT_ALIASES.get(key, cleaned.title() if cleaned else "Maseru")


def _fill_missing_prices(series: pd.Series) -> pd.Series:
    """Impute missing prices with the median observed price."""

    numeric = pd.to_numeric(series, errors="coerce")
    median_price = int(numeric.dropna().median()) if numeric.dropna().shape[0] else 450000
    return numeric.fillna(median_price).astype(int)


def clean_property_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared property-cleaning rules before modeling or display."""

    df = dataframe.copy()
    if df.empty:
        return df
    df["district"] = df["district"].map(normalize_district)
    df["image_paths"] = df["image_paths"].map(_ensure_list)
    df["amenities"] = df.get("amenities", []).map(_ensure_list) if "amenities" in df else [[]] * len(df)
    df["price"] = _fill_missing_prices(df["price"])
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(0).astype(int)
    df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(0).astype(int)
    df["description_en"] = df["description_en"].fillna("").map(str)
    df["description_st"] = df["description_st"].fillna("").map(str)
    fallback = df["title"].fillna("").map(str) + " in " + df["district"]
    df.loc[df["description_en"].str.strip().eq(""), "description_en"] = fallback
    df.loc[df["description_st"].str.strip().eq(""), "description_st"] = fallback
    df["listing_url"] = df["listing_url"].fillna("").map(str)
    if "source" in df:
        df["source"] = df["source"].fillna("unknown").map(str)
    df = df.drop_duplicates(subset=["property_id"])
    df = df.drop_duplicates(subset=["title", "price", "district"])
    return df.reset_index(drop=True)


def clean_client_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize client preference records into the schema expected downstream."""

    df = dataframe.copy()
    list_columns: Iterable[str] = (
        "preferred_districts",
        "preferred_property_types",
        "preferred_channels",
    )
    for column in list_columns:
        df[column] = df[column].map(_ensure_list)
    df["preferred_districts"] = df["preferred_districts"].map(
        lambda values: [normalize_district(item) for item in values]
    )
    df["budget_min"] = pd.to_numeric(df["budget_min"], errors="coerce").fillna(200000).astype(int)
    df["budget_max"] = pd.to_numeric(df["budget_max"], errors="coerce").fillna(900000).astype(int)
    df["preferred_bedrooms"] = pd.to_numeric(
        df["preferred_bedrooms"], errors="coerce"
    ).fillna(2).astype(int)
    df["preferred_language"] = df["preferred_language"].fillna("en").map(str)
    return df.drop_duplicates(subset=["client_id"]).reset_index(drop=True)
