"""Shared serialization and formatting helpers for the Flask frontend."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from lesotho_property_ai.text_utils import strip_html_text

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


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


def format_money_input(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    grouped = f"{whole:,}".replace(",", " ")
    if cents:
        return f"{grouped}.{cents:02d}"
    return grouped


def parse_word_money_amount(text: str) -> float | None:
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    if not tokens:
        return None

    total = 0.0
    current = 0.0
    saw_number_word = False
    for token in tokens:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            saw_number_word = True
        elif token in {"and", "ls", "lsl", "maloti", "lotis"}:
            continue
        elif token in {"hundred", "hunderd", "hundered"}:
            current = max(current, 1) * 100
            saw_number_word = True
        elif token in {"thousand", "thousands", "k"}:
            total += max(current, 1) * 1_000
            current = 0
            saw_number_word = True
        elif token in {"million", "millions", "mil", "mi", "m"}:
            total += max(current, 1) * 1_000_000
            current = 0
            saw_number_word = True

    if not saw_number_word:
        return None
    return total + current


def parse_budget_amount(raw_value: str, field_label: str) -> float:
    text = str(raw_value or "").strip()
    if not text:
        return 0.0

    normalized = text.lower().replace(",", " ")
    number_match = re.search(r"\d[\d\s]*(?:\.\d+)?", normalized)
    if number_match:
        number_text = re.sub(r"\s+", "", number_match.group(0))
        amount = float(number_text)
        multiplier = 1
        after_number = normalized[number_match.end() :]
        if re.search(r"\b(?:m|mi|mil|million|millions)\b", after_number):
            multiplier = 1_000_000
        elif re.search(r"\b(?:k|thousand|thousands)\b", after_number):
            multiplier = 1_000
        if re.search(r"\b(?:hundred|hunderd|hundered)\b\s+\b(?:k|thousand|thousands)\b", after_number):
            multiplier = 100_000
        return amount * multiplier

    word_amount = parse_word_money_amount(normalized)
    if word_amount is not None:
        return float(word_amount)

    raise ValueError(
        f"{field_label} must be a number or money phrase, for example 1 200 000, 500k, or 4 million."
    )


def plain_text_excerpt(value, *, limit: int | None = None) -> str:
    cleaned = strip_html_text(value)
    if limit is None or len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
