from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaning import clean_property_dataframe, normalize_district
from .repository import save_dataframe, save_json


OFFICIAL_DISTRICTS = (
    "Maseru",
    "Leribe",
    "Berea",
    "Mafeteng",
    "Mohale's Hoek",
    "Quthing",
    "Butha-Buthe",
    "Mokhotlong",
    "Qacha's Nek",
    "Thaba-Tseka",
)

RESIDENTIAL_TYPES = {"House", "Apartment", "Townhouse", "Cottage"}
COMMERCIAL_TOKENS = (
    "commercial",
    "office",
    "guest house",
    "guesthouse",
    "shopping centre",
    "shopping center",
    "shop",
    "warehouse",
    "business",
    "lodge",
    "motel",
)
SITE_TOKENS = ("site", "plot", "land", "vacant")
SOUTH_AFRICA_TOKENS = (
    "ladybrand",
    "ficksburg",
    "mantsopa local municipality",
    "setsoto local municipality",
    "free state",
    "south africa",
)

LOCALITY_DISTRICT_HINTS = {
    "thetsane": "Maseru",
    "thetsane west": "Maseru",
    "thetsane east": "Maseru",
    "lower thetsane": "Maseru",
    "upper thamae": "Maseru",
    "katlehong": "Maseru",
    "moshoeshoe 2": "Maseru",
    "masowe 1b": "Maseru",
    "masowe 2": "Maseru",
    "masowe 3": "Maseru",
    "masowe 4": "Maseru",
    "maseru west": "Maseru",
    "old europa": "Maseru",
    "europa": "Maseru",
    "high court": "Maseru",
    "sea point": "Maseru",
    "leqele": "Maseru",
    "leqele police region": "Maseru",
    "khubetsoana": "Maseru",
    "mabote": "Maseru",
    "mohalalitoe": "Maseru",
    "tsosane": "Maseru",
    "race course": "Maseru",
    "stadium area": "Maseru",
    "mpilo": "Maseru",
    "mpilo estates": "Maseru",
    "halekhowa": "Maseru",
    "lecoop": "Maseru",
    "thetsane lecoop": "Maseru",
    "thetsane lesia": "Maseru",
    "ha thetsane": "Maseru",
    "hillsview": "Maseru",
    "reserve (butha buthe)": "Butha-Buthe",
    "butha buthe": "Butha-Buthe",
    "qacha's nek": "Qacha's Nek",
    "qacha’s nek": "Qacha's Nek",
    "quthing": "Quthing",
    "mokhotlong": "Mokhotlong",
    "thaba-tseka": "Thaba-Tseka",
    "mohale's hoek": "Mohale's Hoek",
    "mohales hoek": "Mohale's Hoek",
}

SOURCE_DEFAULT_DISTRICT = {
    "creativeproperties": "Maseru",
    "mosoholdings": "Maseru",
    "mestech": "Maseru",
    "sotholand": "Maseru",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_jsonish_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        current = value.strip()
        if not current:
            return []
        for _ in range(3):
            if not current.startswith("[") and not current.startswith('"['):
                break
            try:
                parsed = json.loads(current)
            except json.JSONDecodeError:
                break
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
            if isinstance(parsed, str):
                current = parsed
                continue
            break
        return [item.strip() for item in current.split("|") if item.strip()]
    return [str(value)]


def _resolve_generated_image_path(value: str) -> str:
    original = str(value).strip()
    if not original:
        return original
    original_path = Path(original)
    if original_path.exists():
        return str(original_path)

    normalized = original.replace("/", "\\")
    marker = "generated\\images\\"
    lowered = normalized.lower()
    if marker in lowered:
        suffix = normalized[lowered.index(marker) :].replace("\\", "/")
        candidate = PROJECT_ROOT / Path(suffix)
        if candidate.exists():
            return str(candidate)
    return original


def _combined_text(row: pd.Series) -> str:
    parts = (
        row.get("district", ""),
        row.get("location_text", ""),
        row.get("title", ""),
        row.get("description_en", ""),
    )
    return " ".join(str(part) for part in parts if str(part).strip())


def infer_country(row: pd.Series) -> str:
    combined = _combined_text(row).lower()
    if any(token in combined for token in SOUTH_AFRICA_TOKENS):
        return "South Africa"
    return "Lesotho"


def infer_official_district(row: pd.Series) -> tuple[str, str]:
    combined = _combined_text(row)
    combined_lower = combined.lower()

    if infer_country(row) == "South Africa":
        return "Outside Lesotho", "outside_lesotho"

    alias_patterns = (
        ("Mohale's Hoek", ("mohale's hoek", "mohales hoek")),
        ("Qacha's Nek", ("qacha's nek", "qacha’s nek")),
        ("Butha-Buthe", ("butha-buthe", "butha buthe")),
        ("Thaba-Tseka", ("thaba-tseka", "thaba tseka")),
        ("Maseru", ("maseru",)),
        ("Leribe", ("leribe",)),
        ("Berea", ("berea",)),
        ("Mafeteng", ("mafeteng",)),
        ("Quthing", ("quthing",)),
        ("Mokhotlong", ("mokhotlong",)),
    )
    for official, aliases in alias_patterns:
        if any(alias in combined_lower for alias in aliases):
            return official, "text_match"

    for locality, official in LOCALITY_DISTRICT_HINTS.items():
        if locality in combined_lower:
            return official, "locality_match"

    source = str(row.get("source", "")).strip().lower()
    if source in SOURCE_DEFAULT_DISTRICT:
        return SOURCE_DEFAULT_DISTRICT[source], "source_default"

    current = normalize_district(row.get("district", ""))
    if current in OFFICIAL_DISTRICTS:
        return current, "existing_value"

    return "Unknown", "unresolved"


def infer_locality(row: pd.Series) -> str:
    location_text = str(row.get("location_text", "")).strip()
    if not location_text:
        return ""
    parts = [part.strip() for part in location_text.split(",") if part.strip()]
    if parts:
        first = parts[0]
        if not re.fullmatch(r"[0-9,. ]+", first):
            return first
    if not re.fullmatch(r"[0-9,. ]+", location_text):
        return location_text
    return ""


def infer_use_bucket(row: pd.Series) -> str:
    title = str(row.get("title", "")).lower()
    property_type = str(row.get("property_type", "")).strip()
    property_type_lower = property_type.lower()
    description = str(row.get("description_en", "")).lower()

    strong_commercial_tokens = (
        "guest house",
        "guesthouse",
        "warehouse",
        "shopping complex",
        "shopping centre",
        "shopping center",
        "supermarket",
        "butchery",
        "pharmacy",
        "internet shop",
    )
    site_title_pattern = re.compile(r"^\s*[\d,.\s]+(?:m²|sqm|square meters?)\s*$", re.I)

    if property_type == "Commercial" or any(token in title or token in property_type_lower for token in COMMERCIAL_TOKENS):
        return "commercial"
    if property_type == "Site" or any(token in title or token in property_type_lower for token in SITE_TOKENS):
        return "site_land"
    if site_title_pattern.search(title):
        return "site_land"
    if property_type in RESIDENTIAL_TYPES:
        if any(token in title or token in description for token in strong_commercial_tokens):
            return "commercial"
        return "residential"
    return "unknown"


def normalize_cnn_property_type(row: pd.Series) -> str | None:
    if row.get("use_bucket") != "residential":
        return None
    combined = _combined_text(row).lower()
    if "townhouse" in combined or str(row.get("property_type")) == "Townhouse":
        return "Townhouse"
    if any(token in combined for token in ("apartment", "flat", "bachelor")) or str(row.get("property_type")) == "Apartment":
        return "Apartment"
    return "House"


def bedroom_class(value: object) -> str | None:
    try:
        bedrooms = int(float(value))
    except (TypeError, ValueError):
        return None
    if bedrooms <= 0:
        return None
    if bedrooms >= 5:
        return "5+"
    return str(bedrooms)


def assign_split(property_id: str) -> str:
    digest = hashlib.md5(property_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def assign_modeling_splits(dataframe: pd.DataFrame) -> pd.Series:
    splits = pd.Series(dataframe["property_id"].map(assign_split).tolist(), index=dataframe.index, dtype="object")

    house_rows = dataframe[
        (dataframe["country"] == "Lesotho")
        & (dataframe["use_bucket"] == "residential")
        & (dataframe["cnn_property_type"] == "House")
        & (dataframe["cnn_bedroom_class"].notna())
    ].copy()
    if house_rows.empty:
        return splits

    try:
        train_ids, holdout_ids = train_test_split(
            house_rows["property_id"].tolist(),
            test_size=0.30,
            random_state=42,
            stratify=house_rows["cnn_bedroom_class"].astype(str),
        )
        holdout_rows = house_rows[house_rows["property_id"].isin(holdout_ids)].copy()
        val_ids, test_ids = train_test_split(
            holdout_rows["property_id"].tolist(),
            test_size=0.50,
            random_state=42,
            stratify=holdout_rows["cnn_bedroom_class"].astype(str),
        )
        house_split_map = {property_id: "train" for property_id in train_ids}
        house_split_map.update({property_id: "val" for property_id in val_ids})
        house_split_map.update({property_id: "test" for property_id in test_ids})
        house_mask = dataframe["property_id"].isin(house_split_map)
        splits.loc[house_mask] = dataframe.loc[house_mask, "property_id"].map(house_split_map)
    except ValueError:
        return splits
    return splits


def curate_property_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = clean_property_dataframe(dataframe)
    if df.empty:
        return df, {"rows": 0}

    df["image_paths"] = df["image_paths"].map(_parse_jsonish_list).map(
        lambda values: [_resolve_generated_image_path(item) for item in values]
    )
    df["amenities"] = df["amenities"].map(_parse_jsonish_list) if "amenities" in df else [[]] * len(df)
    df["image_count"] = df["image_paths"].map(len)
    df["country"] = df.apply(infer_country, axis=1)
    district_info = df.apply(infer_official_district, axis=1)
    df["district_canonical"] = district_info.map(lambda item: item[0])
    df["district_resolution"] = district_info.map(lambda item: item[1])
    df["locality"] = df.apply(infer_locality, axis=1)
    df["use_bucket"] = df.apply(infer_use_bucket, axis=1)
    df["cnn_property_type"] = df.apply(normalize_cnn_property_type, axis=1)
    df["cnn_bedroom_class"] = df["bedrooms"].map(bedroom_class)
    df["split"] = assign_modeling_splits(df)

    exclusion_reasons: list[list[str]] = []
    for _, row in df.iterrows():
        reasons: list[str] = []
        if row["country"] != "Lesotho":
            reasons.append("outside_lesotho")
        if row["use_bucket"] != "residential":
            reasons.append(f"not_residential:{row['use_bucket']}")
        if int(row["image_count"]) < 1:
            reasons.append("no_images")
        if str(row.get("listing_intent", "")) not in {"sale", "rent"}:
            reasons.append("unsupported_listing_intent")
        if row["district_canonical"] == "Unknown":
            reasons.append("unknown_district")
        if row["cnn_bedroom_class"] is None:
            reasons.append("invalid_bedroom_label")
        exclusion_reasons.append(reasons)

    df["cnn_exclusion_reasons"] = ["|".join(reasons) for reasons in exclusion_reasons]
    df["is_residential_curated"] = (
        (df["country"] == "Lesotho")
        & (df["use_bucket"] == "residential")
        & (df["image_count"] > 0)
        & (df["listing_intent"].isin(["sale", "rent"]))
        & (df["district_canonical"] != "Unknown")
    )
    df["is_cnn_candidate"] = df["is_residential_curated"] & df["cnn_bedroom_class"].notna()

    summary = {
        "rows": int(len(df)),
        "country_counts": {str(k): int(v) for k, v in df["country"].value_counts().to_dict().items()},
        "use_bucket_counts": {str(k): int(v) for k, v in df["use_bucket"].value_counts().to_dict().items()},
        "district_resolution_counts": {
            str(k): int(v) for k, v in df["district_resolution"].value_counts().to_dict().items()
        },
        "residential_curated_rows": int(df["is_residential_curated"].sum()),
        "cnn_candidate_rows": int(df["is_cnn_candidate"].sum()),
        "cnn_property_type_counts": {
            str(k): int(v)
            for k, v in df.loc[df["is_cnn_candidate"], "cnn_property_type"].value_counts().to_dict().items()
        },
        "cnn_bedroom_class_counts": {
            str(k): int(v)
            for k, v in df.loc[df["is_cnn_candidate"], "cnn_bedroom_class"].value_counts().to_dict().items()
        },
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()},
        "image_total": int(df["image_count"].sum()),
    }
    return df, summary


def build_image_level_dataset(properties: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in properties.iterrows():
        if not bool(row.get("is_cnn_candidate")):
            continue
        image_paths = _parse_jsonish_list(row.get("image_paths"))
        for index, image_path in enumerate(image_paths, start=1):
            rows.append(
                {
                    "property_id": row["property_id"],
                    "source": row["source"],
                    "image_index": index,
                    "image_path": image_path,
                    "country": row["country"],
                    "district_canonical": row["district_canonical"],
                    "locality": row["locality"],
                    "listing_intent": row["listing_intent"],
                    "cnn_property_type": row["cnn_property_type"],
                    "cnn_bedroom_class": row["cnn_bedroom_class"],
                    "condition": row["condition"],
                    "style": row["style"],
                    "environment": row["environment"],
                    "split": row["split"],
                }
            )
    return pd.DataFrame(rows)


def prepare_modeling_datasets(
    input_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    dataframe = pd.read_csv(input_path)
    curated, summary = curate_property_dataset(dataframe)

    residential = curated[curated["use_bucket"] == "residential"].reset_index(drop=True)
    commercial = curated[curated["use_bucket"] == "commercial"].reset_index(drop=True)
    site_land = curated[curated["use_bucket"] == "site_land"].reset_index(drop=True)
    outside_lesotho = curated[curated["country"] != "Lesotho"].reset_index(drop=True)
    cnn_candidates = curated[curated["is_cnn_candidate"]].reset_index(drop=True)
    cnn_excluded = curated[
        (curated["use_bucket"] == "residential") & (~curated["is_cnn_candidate"])
    ].reset_index(drop=True)
    cnn_images = build_image_level_dataset(curated)

    paths = {
        "curated_master_csv": str(
            save_dataframe(
                curated,
                output_path / "properties_curated_master.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "residential_csv": str(
            save_dataframe(
                residential,
                output_path / "properties_residential_curated.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "commercial_csv": str(
            save_dataframe(
                commercial,
                output_path / "properties_commercial_curated.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "site_land_csv": str(
            save_dataframe(
                site_land,
                output_path / "properties_site_land_curated.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "outside_lesotho_csv": str(
            save_dataframe(
                outside_lesotho,
                output_path / "properties_outside_lesotho.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "cnn_candidates_csv": str(
            save_dataframe(
                cnn_candidates,
                output_path / "properties_residential_cnn_candidates.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "cnn_excluded_csv": str(
            save_dataframe(
                cnn_excluded,
                output_path / "properties_residential_cnn_excluded.csv",
                json_columns=("image_paths", "amenities"),
            )
        ),
        "cnn_images_csv": str(save_dataframe(cnn_images, output_path / "properties_residential_cnn_images.csv")),
    }

    summary.update(
        {
            "residential_rows": int(len(residential)),
            "commercial_rows": int(len(commercial)),
            "site_land_rows": int(len(site_land)),
            "outside_lesotho_rows": int(len(outside_lesotho)),
            "cnn_image_rows": int(len(cnn_images)),
            "artifact_paths": paths,
        }
    )
    paths["curation_summary_json"] = str(save_json(summary, output_path / "curation_summary.json"))
    return paths
