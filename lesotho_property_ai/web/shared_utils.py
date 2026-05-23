"""Shared serialization and formatting helpers for the Flask frontend."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lesotho_property_ai.text_utils import strip_html_text


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def decode_nested_serialized_value(value, max_depth: int = 4):
    current = value
    for _ in range(max_depth):
        if isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                return stripped
            if stripped.startswith(("[", "{", '"')) or stripped in {"null", "true", "false"}:
                try:
                    current = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    return current
        break
    return current


def coerce_image_paths(value) -> list[str]:
    decoded = decode_nested_serialized_value(value)
    if isinstance(decoded, list):
        flattened: list[str] = []
        for item in decoded:
            flattened.extend(coerce_image_paths(item))
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
        stripped = decoded.strip().strip('"').strip("'")
        if not stripped:
            return []
        return [stripped]
    return []


def coerce_text_list(value) -> list[str]:
    decoded = decode_nested_serialized_value(value)
    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded if str(item).strip()]
    if isinstance(decoded, str):
        stripped = decoded.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                return []
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return []


def format_currency(value) -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "LSL -"
    return f"LSL {numeric:,}"


def plain_text_excerpt(value, *, limit: int | None = None) -> str:
    cleaned = strip_html_text(value)
    if limit is None or len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
