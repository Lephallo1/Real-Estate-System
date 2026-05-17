"""Shared data shaping helpers for the Flask frontend."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from flask import current_app, url_for

from lesotho_property_ai.artifacts import resolve_artifact_path
from lesotho_property_ai.web.shared_utils import (
    coerce_image_paths,
    coerce_text_list,
    format_currency,
    load_csv,
    load_json,
)
from lesotho_property_ai.marketing import MarketingAutomation


def artifact_dir() -> Path:
    return Path(current_app.config["ARTIFACT_DIR"])


def base_dir() -> Path:
    return Path(current_app.config["BASE_DIR"])


def load_artifact_csv(filename: str) -> pd.DataFrame:
    return load_csv(resolve_artifact_path(artifact_dir(), filename))


def load_artifact_json(filename: str) -> dict[str, object]:
    return load_json(resolve_artifact_path(artifact_dir(), filename))


def preview_frame(frame: pd.DataFrame, columns: list[str], *, limit: int = 25) -> dict[str, object]:
    available = [column for column in columns if column in frame.columns]
    if not available or frame.empty:
        return {"columns": [], "rows": []}
    preview = frame[available].head(limit).fillna("")
    return {"columns": available, "rows": preview.to_dict(orient="records")}


def numeric_value(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _property_image_url(image_paths_value) -> str | None:
    image_paths = coerce_image_paths(image_paths_value)
    if not image_paths:
        return None
    image_root = (base_dir() / "generated" / "images").resolve()
    raw_path = str(image_paths[0]).strip()
    normalized = raw_path.replace("\\", "/")
    marker = "generated/images/"
    if marker in normalized:
        relpath = normalized.split(marker, 1)[1].lstrip("/")
        return url_for("media_image", relpath=relpath)
    candidate = Path(raw_path).resolve()
    try:
        relpath = candidate.relative_to(image_root).as_posix()
    except ValueError:
        return None
    return url_for("media_image", relpath=relpath)


def serialize_property(row: pd.Series) -> dict[str, object]:
    bedrooms = int(numeric_value(row.get("bedrooms", row.get("predicted_bedrooms", 0)), 0))
    bathrooms = int(numeric_value(row.get("bathrooms", 0), 0))
    floor_area = int(numeric_value(row.get("floor_area_sqm", row.get("area_sqm", 0)), 0))
    property_type = str(row.get("property_type", row.get("cnn_property_type", "House")) or "House")
    title = MarketingAutomation._presentation_title(
        str(row.get("title", "House listing")),
        bedrooms=bedrooms,
        property_type=property_type,
    )
    district = str(row.get("district", row.get("district_canonical", "")) or "").strip()
    description = str(row.get("description_en", row.get("enhanced_description", "")) or "").strip()
    amenities = coerce_text_list(row.get("amenities", []))
    listing_intent = str(row.get("listing_intent", "sale") or "sale").lower()
    ai_score = numeric_value(row.get("overall_score", row.get("match_score", row.get("ai_score", 0.0))), 0.0)
    condition = str(row.get("predicted_condition", row.get("condition", "")) or "").strip()

    badges = []
    if listing_intent == "rent":
        badges.append({"label": "FOR RENT", "tone": "blue"})
    else:
        badges.append({"label": "FOR SALE", "tone": "green"})
    if condition.lower() == "new" or "new" in title.lower():
        badges.append({"label": "NEW", "tone": "orange"})

    return {
        "property_id": str(row.get("property_id", "")),
        "title": title,
        "district": district,
        "locality": str(row.get("locality", "") or "").strip(),
        "price": format_currency(row.get("price", 0)),
        "price_value": numeric_value(row.get("price", 0), 0.0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor_area": floor_area,
        "listing_intent": listing_intent,
        "image_url": _property_image_url(row.get("image_paths", [])),
        "source": str(row.get("source", "") or "").strip(),
        "listing_url": str(row.get("listing_url", "") or "").strip(),
        "description": description[:180] + ("..." if len(description) > 180 else ""),
        "amenities": amenities[:6],
        "badges": badges,
        "ai_score": round(ai_score * 100, 0) if ai_score else None,
        "style": str(row.get("predicted_style", row.get("style", "")) or "").strip(),
        "environment": str(row.get("predicted_environment", row.get("environment", "")) or "").strip(),
        "condition": condition,
    }


def load_stock_frame() -> pd.DataFrame:
    stock = load_artifact_csv("house_recommendation_properties.csv")
    if stock.empty:
        stock = load_artifact_csv("properties_house_reviewed.csv")
    return stock.copy()


def apply_stock_filters(frame: pd.DataFrame, params: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    filtered = frame.copy()
    state = {
        "intent": params.get("intent", "all").lower(),
        "district": params.get("district", "all"),
        "price": params.get("price", "all"),
        "beds": params.get("beds", "all"),
        "feature": params.get("feature", "all"),
    }

    if state["intent"] in {"sale", "rent"} and "listing_intent" in filtered.columns:
        filtered = filtered.loc[
            filtered["listing_intent"].fillna("").astype(str).str.lower().eq(state["intent"])
        ].copy()
    if state["district"] != "all":
        district_column = "district" if "district" in filtered.columns else "district_canonical"
        filtered = filtered.loc[
            filtered[district_column].fillna("").astype(str).eq(state["district"])
        ].copy()
    if state["price"] == "under_1m":
        filtered = filtered.loc[pd.to_numeric(filtered.get("price"), errors="coerce").fillna(0).lt(1_000_000)]
    elif state["price"] == "between_1m_3m":
        prices = pd.to_numeric(filtered.get("price"), errors="coerce").fillna(0)
        filtered = filtered.loc[prices.between(1_000_000, 3_000_000, inclusive="both")]
    elif state["price"] == "above_3m":
        filtered = filtered.loc[pd.to_numeric(filtered.get("price"), errors="coerce").fillna(0).gt(3_000_000)]
    if state["beds"] == "3plus":
        filtered = filtered.loc[pd.to_numeric(filtered.get("bedrooms"), errors="coerce").fillna(0).ge(3)]
    elif state["beds"] == "4plus":
        filtered = filtered.loc[pd.to_numeric(filtered.get("bedrooms"), errors="coerce").fillna(0).ge(4)]

    if state["feature"] == "pool":
        text = (
            filtered.get("title", pd.Series(dtype=str)).fillna("").astype(str)
            + " "
            + filtered.get("description_en", pd.Series(dtype=str)).fillna("").astype(str)
            + " "
            + filtered.get("amenities", pd.Series(dtype=str)).fillna("").astype(str)
        )
        filtered = filtered.loc[text.str.contains("pool", case=False, na=False)].copy()
    elif state["feature"] == "new":
        text = (
            filtered.get("title", pd.Series(dtype=str)).fillna("").astype(str)
            + " "
            + filtered.get("description_en", pd.Series(dtype=str)).fillna("").astype(str)
            + " "
            + filtered.get("predicted_condition", pd.Series(dtype=str)).fillna("").astype(str)
        )
        filtered = filtered.loc[text.str.contains("new", case=False, na=False)].copy()

    return filtered, state


def build_stock_chips(endpoint: str, frame: pd.DataFrame, state: dict[str, str]) -> list[dict[str, object]]:
    district_column = "district" if "district" in frame.columns else "district_canonical"
    available_districts = sorted(
        {
            str(value)
            for value in frame.get(district_column, pd.Series(dtype=str)).dropna().tolist()
            if str(value).strip()
        }
    )
    chip_specs = [
        ("All", {}),
        ("For Sale", {"intent": "sale"}),
        ("For Rent", {"intent": "rent"}),
    ]
    for district in [value for value in ["Maseru", "Leribe", "Berea"] if value in available_districts]:
        chip_specs.append((district, {"district": district}))
    chip_specs.extend(
        [
            ("Under M 1M", {"price": "under_1m"}),
            ("M 1M - M 3M", {"price": "between_1m_3m"}),
            ("3+ Beds", {"beds": "3plus"}),
            ("With Pool", {"feature": "pool"}),
            ("New Build", {"feature": "new"}),
        ]
    )

    chips: list[dict[str, object]] = []
    for label, updates in chip_specs:
        next_state = {key: value for key, value in state.items() if value != "all"}
        if not updates:
            next_state = {}
        else:
            next_state.update(updates)
        active = True
        if not updates:
            active = all(value == "all" for value in state.values())
        else:
            active = all(state.get(key, "all") == value for key, value in updates.items())
        chips.append(
            {
                "label": label,
                "href": f"{url_for(endpoint)}?{urlencode(next_state)}" if next_state else url_for(endpoint),
                "active": active,
            }
        )
    return chips


def stock_card_rows(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, object]]:
    preview = frame.head(limit) if limit is not None else frame
    return [serialize_property(row) for _, row in preview.iterrows()]


def load_recommendation_bundle(prefix: str) -> dict[str, object]:
    properties = load_artifact_csv(f"{prefix}_properties.csv")
    matches = load_artifact_csv(f"{prefix}_matches.csv")
    campaigns = load_artifact_csv(f"{prefix}_campaigns.csv")
    metrics = load_artifact_json(f"{prefix}_metrics.json")
    fusion = load_artifact_json(f"{prefix}_fusion_summary.json")
    marketing = load_artifact_json(f"{prefix}_marketing_summary.json")
    return {
        "properties": properties,
        "matches": matches,
        "campaigns": campaigns,
        "metrics": metrics,
        "fusion": fusion,
        "marketing": marketing,
    }


def recommendation_cards(bundle: dict[str, object], *, single_client: bool) -> list[dict[str, object]]:
    properties: pd.DataFrame = bundle["properties"]
    matches: pd.DataFrame = bundle["matches"]
    campaigns: pd.DataFrame = bundle["campaigns"]
    if properties.empty or matches.empty:
        return []

    lookup = properties.set_index("property_id", drop=False)
    cards: list[dict[str, object]] = []
    for _, match in matches.iterrows():
        property_id = match.get("property_id")
        if property_id not in lookup.index:
            continue
        prop = serialize_property(lookup.loc[property_id])
        reasons = coerce_text_list(match.get("recommendation_reasons", []))
        cues = coerce_text_list(match.get("shared_text_cues", []))
        campaign_row = campaigns.loc[campaigns["property_id"] == property_id].head(1)
        message = ""
        if not campaign_row.empty and "message" in campaign_row.columns:
            message = str(campaign_row.iloc[0]["message"])
        cards.append(
            {
                "client_name": str(match.get("client_name", "")),
                "property": prop,
                "rank": int(numeric_value(match.get("rank", 0), 0)),
                "overall_score": round(numeric_value(match.get("overall_score", 0.0), 0.0), 3),
                "structured_score": round(numeric_value(match.get("structured_score", 0.0), 0.0), 3),
                "text_score": round(numeric_value(match.get("text_score", 0.0), 0.0), 3),
                "vision_score": round(numeric_value(match.get("vision_score", 0.0), 0.0), 3),
                "structured_weight_used": round(numeric_value(match.get("structured_weight_used", 0.0), 0.0), 3),
                "text_weight_used": round(numeric_value(match.get("text_weight_used", 0.0), 0.0), 3),
                "vision_weight_used": round(numeric_value(match.get("vision_weight_used", 0.0), 0.0), 3),
                "fusion_reliability": round(numeric_value(match.get("fusion_reliability", 0.0), 0.0), 3),
                "reasons": reasons,
                "cues": cues,
                "message": message,
                "property_id": str(property_id),
                "single_client": single_client,
            }
        )
    return cards


def grouped_cards(cards: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for card in cards:
        grouped.setdefault(card["client_name"] or "Customer", []).append(card)
    return grouped

