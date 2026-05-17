from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .repository import save_dataframe, save_json


BEDROOM_PATTERN = re.compile(r"\b(\d+)\s*(?:bed(?:room)?s?)\b", re.IGNORECASE)
STYLE_RULES = {
    "Modern": ("modern", "newly built", "highly finished", "up-market", "upmarket"),
    "Contemporary": ("contemporary", "semi-furnished", "semi furnished", "minimalist"),
    "Traditional": ("traditional", "classic", "heritage", "stone"),
    "Family": ("family", "spacious", "yard", "secure yard", "garden"),
}
ENVIRONMENT_RULES = {
    "Garden": ("garden", "serapa", "yard", "lawn"),
    "Hillside": ("hillside", "maralleng", "view", "views"),
    "Urban": ("urban", "town", "town centre", "city", "arrival centre"),
    "Suburban": ("suburban", "quiet", "residential", "estate", "thetsane"),
}
CONDITION_RULES = {
    "New": ("new", "newly", "renovated", "highly finished", "under construction"),
    "Good": ("good", "neat", "well-kept", "well kept", "clean"),
    "Renovation Needed": ("renovation", "fixer", "unfinished", "needs work"),
}
NOISE_PATTERNS = (
    ("possible_multi_unit", re.compile(r"\b(?:[2-9]|\d{2,})\s+units?\b", re.IGNORECASE)),
    ("possible_development", re.compile(r"development|under construction", re.IGNORECASE)),
    ("possible_commercial_mix", re.compile(r"office|commercial|guest house|guesthouse", re.IGNORECASE)),
)
MANUAL_SECOND_PASS_RULES = {
    "creative-7275": {
        "review_status": "reviewed",
        "approved_for_training": "yes",
        "notes": "Checked manually: keep for CNN training. Price is likely imputed or missing from source, so use cautiously in recommendation work.",
    },
    "creative-7251": {
        "review_status": "reviewed",
        "approved_for_training": "yes",
        "reviewed_environment": "Urban",
        "notes": "Checked manually: keep for CNN training. Arrival Centre context suggests Urban environment. Rental price still needs later verification for recommendation use.",
    },
    "propmarket-81l4_NDWdSkfdHzXFJwUq": {
        "review_status": "reviewed",
        "approved_for_training": "yes",
        "notes": "Checked manually: keep. Low sale price is plausible because the listing text mentions no water and electricity.",
    },
    "propmarket-T35gU6I6C2_rb4DisUUSd": {
        "review_status": "reviewed",
        "approved_for_training": "yes",
        "notes": "Checked manually: keep. Low price may be plausible for a small 2-room house and weak servicing.",
    },
    "moso-stand-alone-house-tsosane-48": {
        "review_status": "reviewed",
        "approved_for_training": "yes",
        "reviewed_listing_intent": "sale",
        "reviewed_price": 1200000,
        "notes": "Checked manually: source mixes sale and rental text. Keep sale price for structured metadata and treat the rental text as secondary marketing info.",
    },
}


def _parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [item.strip() for item in stripped.split("|") if item.strip()]
    return [str(value)]


def _extract_bedrooms_from_text(text: str) -> int | None:
    match = BEDROOM_PATTERN.search(text)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def _bedroom_class(value: int | None) -> str | None:
    if value is None or value <= 0:
        return None
    if value >= 5:
        return "5+"
    return str(value)


def _normalize_bedroom_class_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _suggest_from_keywords(text: str, rules: dict[str, tuple[str, ...]]) -> str | None:
    lowered = text.lower()
    scores: Counter[str] = Counter()
    for label, keywords in rules.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[label] += 1
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def _to_bool_string(value: object, default: str = "yes") -> str:
    current = str(value or "").strip().lower()
    if current in {"yes", "y", "true", "1"}:
        return "yes"
    if current in {"no", "n", "false", "0"}:
        return "no"
    return default


def _review_priority(flags: list[str]) -> str:
    if not flags:
        return "low"
    if any(flag.startswith(("bedroom_", "possible_", "price_")) for flag in flags):
        return "high"
    return "medium"


def build_house_label_review(
    input_csv: str | Path,
    output_dir: str | Path,
    target_property_type: str = "House",
) -> dict[str, str]:
    dataframe = pd.read_csv(Path(input_csv))
    if "cnn_property_type" in dataframe:
        dataframe = dataframe[dataframe["cnn_property_type"].astype(str) == target_property_type].copy()
    if "is_cnn_candidate" in dataframe:
        dataframe = dataframe[dataframe["is_cnn_candidate"].fillna(False).astype(bool)].copy()
    dataframe = dataframe.reset_index(drop=True)

    review_rows: list[dict[str, object]] = []
    flag_counts: Counter[str] = Counter()
    for row in dataframe.itertuples(index=False):
        image_paths = _parse_list(getattr(row, "image_paths", []))
        text = " ".join(
            [
                str(getattr(row, "title", "")),
                str(getattr(row, "description_en", "")),
                str(getattr(row, "description_st", "")),
                str(getattr(row, "location_text", "")),
            ]
        )

        suggested_bedrooms = _extract_bedrooms_from_text(text)
        suggested_bedroom_class = _bedroom_class(suggested_bedrooms)
        suggested_style = _suggest_from_keywords(text, STYLE_RULES)
        suggested_environment = _suggest_from_keywords(text, ENVIRONMENT_RULES)
        suggested_condition = _suggest_from_keywords(text, CONDITION_RULES)

        flags: list[str] = []
        current_bedroom_class = str(getattr(row, "cnn_bedroom_class", "") or "")
        if suggested_bedroom_class and current_bedroom_class and suggested_bedroom_class != current_bedroom_class:
            flags.append(f"bedroom_mismatch:{current_bedroom_class}->{suggested_bedroom_class}")
        current_style = str(getattr(row, "style", "") or "")
        if suggested_style and current_style and suggested_style != current_style:
            flags.append(f"style_mismatch:{current_style}->{suggested_style}")
        current_environment = str(getattr(row, "environment", "") or "")
        if suggested_environment and current_environment and suggested_environment != current_environment:
            flags.append(f"environment_mismatch:{current_environment}->{suggested_environment}")
        current_condition = str(getattr(row, "condition", "") or "")
        if suggested_condition and current_condition and suggested_condition != current_condition:
            flags.append(f"condition_mismatch:{current_condition}->{suggested_condition}")

        price = float(getattr(row, "price", 0) or 0)
        listing_intent = str(getattr(row, "listing_intent", "") or "").lower()
        if listing_intent == "sale" and 0 < price < 150000:
            flags.append("price_low_for_sale")
        if listing_intent == "rent" and price > 80000:
            flags.append("price_high_for_rent")

        for flag_name, pattern in NOISE_PATTERNS:
            if pattern.search(text):
                flags.append(flag_name)

        if len(image_paths) < 3:
            flags.append("low_image_count")

        for flag in flags:
            flag_counts[flag] += 1

        review_rows.append(
            {
                "property_id": row.property_id,
                "source": getattr(row, "source", ""),
                "title": getattr(row, "title", ""),
                "district_canonical": getattr(row, "district_canonical", getattr(row, "district", "")),
                "locality": getattr(row, "locality", ""),
                "listing_intent": getattr(row, "listing_intent", ""),
                "price": getattr(row, "price", ""),
                "bedrooms": getattr(row, "bedrooms", ""),
                "bathrooms": getattr(row, "bathrooms", ""),
                "image_count": len(image_paths),
                "sample_image_1": image_paths[0] if len(image_paths) > 0 else "",
                "sample_image_2": image_paths[1] if len(image_paths) > 1 else "",
                "sample_image_3": image_paths[2] if len(image_paths) > 2 else "",
                "listing_url": getattr(row, "listing_url", ""),
                "current_cnn_property_type": getattr(row, "cnn_property_type", ""),
                "current_cnn_bedroom_class": current_bedroom_class,
                "current_style": current_style,
                "current_environment": current_environment,
                "current_condition": current_condition,
                "suggested_bedrooms_from_text": suggested_bedrooms,
                "suggested_cnn_bedroom_class": suggested_bedroom_class,
                "suggested_style": suggested_style,
                "suggested_environment": suggested_environment,
                "suggested_condition": suggested_condition,
                "review_flags": "|".join(flags),
                "review_priority": _review_priority(flags),
                "approved_for_training": "yes",
                "review_status": "pending",
                "reviewed_bedrooms": "",
                "reviewed_cnn_bedroom_class": "",
                "reviewed_style": "",
                "reviewed_environment": "",
                "reviewed_condition": "",
                "reviewed_price": "",
                "reviewed_listing_intent": "",
                "reviewer_notes": "",
            }
        )

    output_path = Path(output_dir)
    review_df = pd.DataFrame(review_rows)
    review_csv = save_dataframe(review_df, output_path / "house_label_review.csv")
    summary = {
        "rows": int(len(review_df)),
        "review_priority_counts": review_df["review_priority"].value_counts().to_dict() if not review_df.empty else {},
        "flag_counts": dict(flag_counts),
        "target_property_type": target_property_type,
    }
    summary_json = save_json(summary, output_path / "house_label_review_summary.json")
    return {
        "review_csv": str(review_csv),
        "summary_json": str(summary_json),
    }


def seed_house_label_review(
    review_csv: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    review_path = Path(review_csv)
    review = pd.read_csv(review_path)
    working = review.copy()
    for column in (
        "approved_for_training",
        "review_status",
        "reviewed_bedrooms",
        "reviewed_cnn_bedroom_class",
        "reviewed_style",
        "reviewed_environment",
        "reviewed_condition",
        "reviewed_price",
        "reviewed_listing_intent",
        "reviewer_notes",
    ):
        if column not in working:
            working[column] = ""
        working[column] = working[column].astype("object")

    action_counts: Counter[str] = Counter()
    for index, row in working.iterrows():
        flags = [flag.strip() for flag in str(row.get("review_flags", "") or "").split("|") if flag.strip()]
        current_notes = row.get("reviewer_notes", "")
        if isinstance(current_notes, float) and pd.isna(current_notes):
            current_notes = ""
        current_notes = str(current_notes or "").strip()
        notes: list[str] = [current_notes] if current_notes else []

        if any(
            flag.startswith(("possible_multi_unit", "possible_development", "possible_commercial_mix"))
            for flag in flags
        ):
            working.at[index, "approved_for_training"] = "no"
            working.at[index, "review_status"] = "exclude"
            notes.append("Auto-excluded in first-pass review because the row looks multi-unit, development stock, or mixed commercial inventory.")
            action_counts["excluded_obvious_noise"] += 1
            working.at[index, "reviewer_notes"] = " ".join(notes).strip()
            continue

        suggested_bedrooms = row.get("suggested_bedrooms_from_text")
        suggested_class = _normalize_bedroom_class_value(row.get("suggested_cnn_bedroom_class"))
        current_class = _normalize_bedroom_class_value(row.get("current_cnn_bedroom_class"))
        if (
            suggested_class
            and current_class
            and suggested_class != current_class
            and any(flag.startswith("bedroom_mismatch:") for flag in flags)
        ):
            if suggested_class != "5+":
                try:
                    working.at[index, "reviewed_bedrooms"] = int(float(suggested_bedrooms))
                except (TypeError, ValueError):
                    pass
            working.at[index, "reviewed_cnn_bedroom_class"] = suggested_class
            working.at[index, "review_status"] = "reviewed"
            notes.append(
                f"Auto-corrected bedroom label from {current_class} to {suggested_class} using title/description text."
            )
            action_counts["bedroom_label_updates"] += 1

        if notes:
            working.at[index, "reviewer_notes"] = " ".join(notes).strip()

        property_id = str(row.get("property_id", "") or "").strip()
        override = MANUAL_SECOND_PASS_RULES.get(property_id)
        if override:
            for column in (
                "approved_for_training",
                "review_status",
                "reviewed_bedrooms",
                "reviewed_cnn_bedroom_class",
                "reviewed_style",
                "reviewed_environment",
                "reviewed_condition",
                "reviewed_price",
                "reviewed_listing_intent",
            ):
                if column in override:
                    working.at[index, column] = override[column]
            extra_note = str(override.get("notes", "") or "").strip()
            if extra_note:
                combined_note = str(working.at[index, "reviewer_notes"] or "").strip()
                working.at[index, "reviewer_notes"] = (
                    f"{combined_note} {extra_note}".strip() if combined_note else extra_note
                )
            action_counts["manual_second_pass_overrides"] += 1

    output_path = Path(output_dir) if output_dir is not None else review_path.parent
    seeded_csv = save_dataframe(working, output_path / review_path.name)
    summary = {
        "rows": int(len(working)),
        "action_counts": dict(action_counts),
        "review_status_counts": working["review_status"].fillna("pending").astype(str).value_counts().to_dict(),
        "approved_for_training_counts": working["approved_for_training"].fillna("yes").astype(str).value_counts().to_dict(),
    }
    summary_json = save_json(summary, output_path / "house_label_review_seeded_summary.json")
    return {
        "review_csv": str(seeded_csv),
        "summary_json": str(summary_json),
    }


def apply_house_label_review(
    candidates_csv: str | Path,
    images_csv: str | Path,
    review_csv: str | Path,
    output_dir: str | Path,
    target_property_type: str = "House",
) -> dict[str, str]:
    candidates = pd.read_csv(Path(candidates_csv))
    images = pd.read_csv(Path(images_csv))
    review = pd.read_csv(Path(review_csv))

    review = review.copy()
    for column in (
        "approved_for_training",
        "review_status",
        "reviewed_bedrooms",
        "reviewed_cnn_bedroom_class",
        "reviewed_style",
        "reviewed_environment",
        "reviewed_condition",
        "reviewed_price",
        "reviewed_listing_intent",
        "reviewer_notes",
    ):
        if column not in review:
            review[column] = ""
    review["approved_for_training"] = review["approved_for_training"].map(_to_bool_string)
    review["review_status"] = review["review_status"].fillna("pending").astype(str).str.strip().str.lower()

    working = candidates.copy()
    if "cnn_property_type" in working:
        working = working[working["cnn_property_type"].astype(str) == target_property_type].copy()
    working = working.merge(
        review[
            [
                "property_id",
                "approved_for_training",
                "review_status",
                "reviewed_bedrooms",
                "reviewed_cnn_bedroom_class",
                "reviewed_style",
                "reviewed_environment",
                "reviewed_condition",
                "reviewed_price",
                "reviewed_listing_intent",
                "reviewer_notes",
            ]
        ],
        on="property_id",
        how="left",
    )

    working["reviewed_bedrooms"] = pd.to_numeric(working["reviewed_bedrooms"], errors="coerce")
    if "bedrooms" in working:
        reviewed_bedroom_mask = working["reviewed_bedrooms"].notna()
        working.loc[reviewed_bedroom_mask, "bedrooms"] = (
            working.loc[reviewed_bedroom_mask, "reviewed_bedrooms"].astype(int)
        )

    working["cnn_bedroom_class"] = working["cnn_bedroom_class"].astype("object")
    reviewed_class_mask = working["reviewed_cnn_bedroom_class"].fillna("").astype(str).str.strip().ne("")
    working.loc[reviewed_class_mask, "cnn_bedroom_class"] = (
        working.loc[reviewed_class_mask, "reviewed_cnn_bedroom_class"].astype(str).str.strip()
    )
    auto_class_mask = (~reviewed_class_mask) & working["reviewed_bedrooms"].notna()
    working.loc[auto_class_mask, "cnn_bedroom_class"] = working.loc[auto_class_mask, "bedrooms"].map(
        lambda value: _bedroom_class(int(value)) if pd.notna(value) else None
    )
    working["cnn_bedroom_class"] = working["cnn_bedroom_class"].map(_normalize_bedroom_class_value)

    for column, review_column in (
        ("style", "reviewed_style"),
        ("environment", "reviewed_environment"),
        ("condition", "reviewed_condition"),
    ):
        reviewed_mask = working[review_column].fillna("").astype(str).str.strip().ne("")
        working.loc[reviewed_mask, column] = working.loc[reviewed_mask, review_column].astype(str).str.strip()

    working["reviewed_price"] = pd.to_numeric(working["reviewed_price"], errors="coerce")
    reviewed_price_mask = working["reviewed_price"].notna()
    if "price" in working:
        working.loc[reviewed_price_mask, "price"] = working.loc[reviewed_price_mask, "reviewed_price"].astype(int)

    reviewed_intent_mask = working["reviewed_listing_intent"].fillna("").astype(str).str.strip().ne("")
    if "listing_intent" in working:
        working.loc[reviewed_intent_mask, "listing_intent"] = (
            working.loc[reviewed_intent_mask, "reviewed_listing_intent"].astype(str).str.strip().str.lower()
        )

    working["include_in_reviewed_training"] = (
        working["approved_for_training"].fillna("yes").map(_to_bool_string).eq("yes")
        & working["review_status"].fillna("pending").ne("exclude")
    )
    working["is_cnn_candidate"] = working["include_in_reviewed_training"]

    reviewed_properties = working[
        [
            column
            for column in working.columns
            if column
            not in {
                "approved_for_training",
                "review_status",
                "reviewed_bedrooms",
                "reviewed_cnn_bedroom_class",
                "reviewed_style",
                "reviewed_environment",
                "reviewed_condition",
            }
        ]
    ].copy()

    included_ids = set(
        reviewed_properties.loc[
            reviewed_properties["include_in_reviewed_training"].fillna(False).astype(bool), "property_id"
        ].astype(str)
    )
    reviewed_images = images[images["property_id"].astype(str).isin(included_ids)].copy()
    reviewed_images = reviewed_images.merge(
        reviewed_properties[
            [
                "property_id",
                "cnn_bedroom_class",
                "condition",
                "style",
                "environment",
            ]
        ],
        on="property_id",
        how="left",
        suffixes=("", "_reviewed"),
    )
    for column in ("cnn_bedroom_class", "condition", "style", "environment"):
        reviewed_column = f"{column}_reviewed"
        if reviewed_column in reviewed_images:
            reviewed_images[column] = reviewed_images[reviewed_column]
            reviewed_images = reviewed_images.drop(columns=[reviewed_column])

    output_path = Path(output_dir)
    properties_csv = save_dataframe(
        reviewed_properties,
        output_path / "properties_house_reviewed.csv",
        json_columns=("image_paths", "amenities"),
    )
    images_csv_out = save_dataframe(
        reviewed_images,
        output_path / "properties_house_reviewed_images.csv",
    )

    modified_counts = {
        "bedroom_class_updates": int(reviewed_class_mask.sum() + auto_class_mask.sum()),
        "style_updates": int(working["reviewed_style"].fillna("").astype(str).str.strip().ne("").sum()),
        "environment_updates": int(working["reviewed_environment"].fillna("").astype(str).str.strip().ne("").sum()),
        "condition_updates": int(working["reviewed_condition"].fillna("").astype(str).str.strip().ne("").sum()),
    }
    summary = {
        "rows": int(len(reviewed_properties)),
        "included_training_rows": int(reviewed_properties["include_in_reviewed_training"].sum()),
        "excluded_training_rows": int((~reviewed_properties["include_in_reviewed_training"]).sum()),
        "image_rows": int(len(reviewed_images)),
        "modified_counts": modified_counts,
        "review_status_counts": review["review_status"].value_counts().to_dict(),
    }
    summary_json = save_json(summary, output_path / "house_label_review_applied_summary.json")
    return {
        "reviewed_properties_csv": str(properties_csv),
        "reviewed_images_csv": str(images_csv_out),
        "summary_json": str(summary_json),
    }
