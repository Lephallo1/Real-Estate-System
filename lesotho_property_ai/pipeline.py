"""Top-level orchestration for the assignment prototype.

This module is where the project ties its major pieces together:
- load curated property inventory
- run vision predictions
- run NLP processing
- rank properties for clients
- generate simulated marketing outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .artifacts import artifact_path, ensure_artifact_dirs, resolve_artifact_path
from .config import AppConfig
from .data import (
    AVAILABLE_LIVE_SOURCES,
    SimulatedDatasetGenerator,
    clean_client_dataframe,
    clean_property_dataframe,
    scrape_live_properties,
)
from .data.repository import save_dataframe, save_json
from .marketing import MarketingAutomation
from .matching.engine import MatchingEngine, MatchingWeights
from .nlp import MultilingualTextProcessor
from .vision import PropertyVisionAnalyzer


@dataclass(slots=True)
class PipelineResult:
    config: AppConfig
    properties: pd.DataFrame
    clients: pd.DataFrame
    matches: pd.DataFrame
    campaigns: pd.DataFrame
    metrics: dict[str, dict[str, object]]
    artifact_paths: dict[str, str]


def _summarize_recommendation_fusion(matches: pd.DataFrame) -> dict[str, object]:
    """Aggregate fusion behavior into lecturer-friendly summary metrics."""

    if matches.empty:
        return {
            "mean_component_scores": {},
            "mean_weights_used": {},
            "mean_fusion_reliability": 0.0,
            "top_reason_counts": {},
        }

    top_matches = matches.loc[matches["rank"] == 1].copy() if "rank" in matches.columns else matches.copy()
    component_means = {
        "structured": round(float(matches["structured_score"].mean()), 4),
        "text": round(float(matches["text_score"].mean()), 4),
        "vision": round(float(matches["vision_score"].mean()), 4),
        "overall": round(float(matches["overall_score"].mean()), 4),
    }
    weight_means = {
        "structured": round(float(matches["structured_weight_used"].mean()), 4),
        "text": round(float(matches["text_weight_used"].mean()), 4),
        "vision": round(float(matches["vision_weight_used"].mean()), 4),
    }

    reason_counts: dict[str, int] = {}
    for value in top_matches.get("recommendation_reasons", pd.Series(dtype=object)).tolist():
        if isinstance(value, list):
            reasons = value
        elif isinstance(value, str):
            reasons = [part.strip() for part in value.strip("[]").split(",") if part.strip()]
        else:
            reasons = []
        for reason in reasons:
            normalized = str(reason).strip().strip("'\"")
            if not normalized:
                continue
            reason_counts[normalized] = reason_counts.get(normalized, 0) + 1
    top_reason_counts = dict(
        sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
    )

    return {
        "mean_component_scores": component_means,
        "mean_weights_used": weight_means,
        "mean_fusion_reliability": round(float(matches["fusion_reliability"].mean()), 4),
        "top_reason_counts": top_reason_counts,
    }


def _summarize_campaigns(campaigns: pd.DataFrame) -> dict[str, object]:
    """Summarize the campaign simulation layer for the dashboard and report."""

    if campaigns.empty:
        return {
            "campaigns_generated": 0,
            "channel_counts": {},
            "language_counts": {},
            "variant_counts": {},
            "mean_match_score": 0.0,
            "mean_estimated_engagement_score": 0.0,
        }

    return {
        "campaigns_generated": int(len(campaigns)),
        "channel_counts": campaigns["channel"].value_counts().to_dict() if "channel" in campaigns else {},
        "language_counts": campaigns["language"].value_counts().to_dict() if "language" in campaigns else {},
        "variant_counts": campaigns["campaign_variant"].value_counts().to_dict()
        if "campaign_variant" in campaigns
        else {},
        "mean_match_score": round(float(campaigns["match_score"].mean()), 4)
        if "match_score" in campaigns
        else 0.0,
        "mean_estimated_engagement_score": round(float(campaigns["estimated_engagement_score"].mean()), 4)
        if "estimated_engagement_score" in campaigns
        else 0.0,
    }


def _resolve_house_input_path(config: AppConfig, input_csv: str | Path) -> Path:
    input_path = Path(input_csv)
    if not input_path.is_absolute():
        input_path = config.base_dir / input_csv
    if not input_path.exists():
        raise FileNotFoundError(f"Curated property dataset not found: {input_path}")
    return input_path


def _resolve_preferred_house_input_path(config: AppConfig, input_csv: str | Path | None) -> Path:
    if input_csv is not None:
        return _resolve_house_input_path(config, input_csv)

    candidate_paths = (
        resolve_artifact_path(config.output_dir, "properties_house_reviewed.csv"),
        resolve_artifact_path(config.output_dir, "properties_residential_cnn_candidates.csv"),
    )
    for candidate in candidate_paths:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No curated house dataset was found. Expected reviewed or candidate property CSV artifacts."
    )


def _filter_demo_house_inventory(properties: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if properties.empty:
        return properties, 0

    text = (
        properties["title"].fillna("").astype(str)
        + " "
        + properties["description_en"].fillna("").astype(str)
    )
    district_text = properties["district"].fillna("").astype(str)
    # These masks remove rows that are still useful in the raw archive but are
    # too noisy for a clean lecturer-facing house recommendation demo.
    development_mask = text.str.contains(
        r"development|under construction|vacant",
        case=False,
        na=False,
        regex=True,
    )
    multi_unit_mask = text.str.contains(
        r"\b(?:[2-9]|\d{2,})\s+units?\b",
        case=False,
        na=False,
        regex=True,
    )
    floor_mask = text.str.contains(
        r"ground\s*floor|staff\s*quarters|arrival\s*centre",
        case=False,
        na=False,
        regex=True,
    )
    invalid_district_mask = district_text.str.fullmatch(r"\d+(?:\.\d+)?", na=False)
    filtered = properties.loc[~(development_mask | multi_unit_mask | floor_mask | invalid_district_mask)].copy()
    return filtered, int(len(properties) - len(filtered))


def _normalize_demo_property_titles(properties: pd.DataFrame) -> pd.DataFrame:
    if properties.empty or "title" not in properties.columns:
        return properties

    normalized = properties.copy()
    if "raw_title" not in normalized.columns:
        normalized["raw_title"] = normalized["title"].fillna("").astype(str)

    def _presentation_row_title(row: pd.Series) -> str:
        bedrooms = row.get("bedrooms", row.get("predicted_bedrooms", 0))
        property_type = row.get("property_type", row.get("predicted_property_type", "House"))
        return MarketingAutomation._presentation_title(
            row.get("title", ""),
            bedrooms=bedrooms,
            property_type=property_type,
        )

    normalized["display_title"] = normalized.apply(_presentation_row_title, axis=1)
    normalized["title"] = normalized["display_title"].fillna(normalized["title"]).astype(str)
    return normalized


def _load_house_inventory(
    config: AppConfig,
    input_csv: str | Path | None,
    listing_intent: str,
    strict_house_only: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    input_path = _resolve_preferred_house_input_path(config, input_csv)
    properties = pd.read_csv(input_path)
    properties = clean_property_dataframe(properties)
    properties = properties.copy()

    if "district_canonical" in properties:
        properties["district"] = (
            properties["district_canonical"]
            .fillna(properties["district"])
            .replace("", pd.NA)
            .fillna(properties["district"])
        )

    if strict_house_only:
        type_column = "cnn_property_type" if "cnn_property_type" in properties else "property_type"
        properties = properties.loc[
            properties[type_column].fillna(properties["property_type"]).astype(str).str.lower().eq("house")
        ].copy()
        properties["property_type"] = "House"

    if "include_in_reviewed_training" in properties.columns:
        # When reviewed labels exist, prefer them over the looser candidate set.
        reviewed_mask = properties["include_in_reviewed_training"].fillna(False).astype(bool)
        if reviewed_mask.any():
            properties = properties.loc[reviewed_mask].copy()

    normalized_intent = listing_intent.lower()
    if normalized_intent not in {"sale", "rent", "both"}:
        raise ValueError("listing_intent must be one of: sale, rent, both")
    if normalized_intent != "both" and "listing_intent" in properties:
        properties = properties.loc[
            properties["listing_intent"].fillna("sale").astype(str).str.lower().eq(normalized_intent)
        ].copy()

    properties, excluded_demo_noise = _filter_demo_house_inventory(properties)
    properties = _normalize_demo_property_titles(properties)
    if properties.empty:
        raise RuntimeError("No curated house properties matched the selected filters.")

    inventory_metrics = {
        "listing_intent": normalized_intent,
        "strict_house_only": bool(strict_house_only),
        "excluded_demo_noise_rows": int(excluded_demo_noise),
        "properties_considered": int(len(properties)),
        "district_counts": properties["district"].value_counts().to_dict(),
    }
    return properties, inventory_metrics


def _run_house_recommendation_pipeline(
    config: AppConfig,
    properties: pd.DataFrame,
    clients: pd.DataFrame,
    top_n: int,
    artifact_prefix: str,
    extra_metrics: dict[str, object] | None = None,
    constraint_mode: str = "soft",
) -> PipelineResult:
    """Run the main multimodal recommendation flow on already-prepared inputs."""
    vision = PropertyVisionAnalyzer().analyze(properties)
    properties = properties.merge(vision.dataframe, on="property_id", how="left")

    text_processor = MultilingualTextProcessor()
    text_result = text_processor.process(properties, clients)
    properties = text_result.properties
    clients = text_result.clients

    matcher = MatchingEngine(
        text_processor=text_processor,
        weights=MatchingWeights(
            structured=config.structured_weight,
            text=config.text_weight,
            vision=config.vision_weight,
        ),
    )
    matches = matcher.rank_for_all_clients(properties, clients, top_n=top_n, constraint_mode=constraint_mode)

    marketer = MarketingAutomation()
    campaigns = marketer.generate(matches, properties, clients)

    top_matches = matches.loc[matches["rank"] == 1].copy()
    fusion_summary = _summarize_recommendation_fusion(matches)
    marketing_summary = _summarize_campaigns(campaigns)
    ensure_artifact_dirs(config.output_dir)
    artifact_paths = {
        "properties_csv": str(
            save_dataframe(
                properties,
                artifact_path(config.output_dir, f"{artifact_prefix}_properties.csv"),
                json_columns=("image_paths", "amenities", "property_keywords", "text_embedding", "vision_embedding"),
            )
        ),
        "clients_csv": str(
            save_dataframe(
                clients,
                artifact_path(config.output_dir, f"{artifact_prefix}_clients.csv"),
                json_columns=(
                    "preferred_districts",
                    "preferred_property_types",
                    "preferred_channels",
                    "client_keywords",
                    "text_embedding",
                ),
            )
        ),
        "matches_csv": str(
            save_dataframe(
                matches,
                artifact_path(config.output_dir, f"{artifact_prefix}_matches.csv"),
                json_columns=("shared_text_cues", "recommendation_reasons"),
            )
        ),
        "campaigns_csv": str(
            save_dataframe(
                campaigns,
                artifact_path(config.output_dir, f"{artifact_prefix}_campaigns.csv"),
                json_columns=("recommendation_reasons",),
            )
        ),
    }

    recommendation_metrics = {
        "properties_considered": int(len(properties)),
        "clients_profiled": int(len(clients)),
        "matches_generated": int(len(matches)),
        "campaigns_generated": int(len(campaigns)),
        "top_n": int(top_n),
        "constraint_mode": constraint_mode,
        "mean_top_match_score": round(float(top_matches["overall_score"].mean()), 4)
        if not top_matches.empty
        else 0.0,
    }
    if extra_metrics:
        recommendation_metrics.update(extra_metrics)

    metrics = {
        "vision": vision.metrics,
        "nlp": text_result.metrics,
        "recommendation": recommendation_metrics,
        "fusion": fusion_summary,
        "marketing": marketing_summary,
    }
    artifact_paths["metrics_json"] = str(
        save_json(metrics, artifact_path(config.output_dir, f"{artifact_prefix}_metrics.json"))
    )
    artifact_paths["fusion_json"] = str(
        save_json(
            fusion_summary,
            artifact_path(config.output_dir, f"{artifact_prefix}_fusion_summary.json"),
        )
    )
    artifact_paths["marketing_json"] = str(
        save_json(
            marketing_summary,
            artifact_path(config.output_dir, f"{artifact_prefix}_marketing_summary.json"),
        )
    )

    return PipelineResult(
        config=config,
        properties=properties,
        clients=clients,
        matches=matches,
        campaigns=campaigns,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )


def _build_house_demo_clients(listing_intent: str, client_count: int) -> pd.DataFrame:
    normalized_intent = listing_intent.lower()
    if normalized_intent == "rent":
        rows = [
            {
                "client_id": "HOUSE-CLIENT-001",
                "name": "Naleli",
                "budget_min": 7000,
                "budget_max": 16000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Looking for a clean 3 bedroom rental house in Maseru with a secure yard and parking.",
                "free_text_preference_st": "Ke batla ntlo ya rente ya dikamore tse tharo Maseru e nang le lebala le sireletsehileng le parking.",
                "preferred_language": "en",
                "preferred_channels": ["email", "social"],
            },
            {
                "client_id": "HOUSE-CLIENT-002",
                "name": "Teboho",
                "budget_min": 5000,
                "budget_max": 11000,
                "preferred_districts": ["Berea", "Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 2,
                "free_text_preference_en": "Need an affordable family rental house close to town and transport.",
                "free_text_preference_st": "Ke hloka ntlo ya rente ya lelapa e theko e tlase e haufi le toropo le dipalangwang.",
                "preferred_language": "st",
                "preferred_channels": ["social", "email"],
            },
            {
                "client_id": "HOUSE-CLIENT-003",
                "name": "Mpho",
                "budget_min": 10000,
                "budget_max": 22000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 4,
                "free_text_preference_en": "Need a spacious rental house in a quiet area for a growing family.",
                "free_text_preference_st": "Ke batla ntlo e pharalletseng ya rente tikolohong e kgutsitseng bakeng sa lelapa.",
                "preferred_language": "en",
                "preferred_channels": ["email"],
            },
            {
                "client_id": "HOUSE-CLIENT-004",
                "name": "Lineo",
                "budget_min": 6000,
                "budget_max": 14000,
                "preferred_districts": ["Leribe", "Berea"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Searching for a neat rental home with a garden and family space.",
                "free_text_preference_st": "Ke batla ntlo e hlwekileng ya rente e nang le serapa le sebaka sa lelapa.",
                "preferred_language": "st",
                "preferred_channels": ["social"],
            },
        ]
    else:
        rows = [
            {
                "client_id": "HOUSE-CLIENT-001",
                "name": "Naleli",
                "budget_min": 350000,
                "budget_max": 950000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Looking for a modern 3 bedroom family house in Maseru with a garden and parking.",
                "free_text_preference_st": "Ke batla ntlo ya kajeno ya lelapa ya dikamore tse tharo Maseru e nang le serapa le parking.",
                "preferred_language": "en",
                "preferred_channels": ["email", "social"],
            },
            {
                "client_id": "HOUSE-CLIENT-002",
                "name": "Teboho",
                "budget_min": 220000,
                "budget_max": 550000,
                "preferred_districts": ["Berea", "Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 2,
                "free_text_preference_en": "Need an affordable starter house near town with secure access.",
                "free_text_preference_st": "Ke hloka ntlo ya pele e theko e tlase e haufi le toropo le tsela e sireletsehileng.",
                "preferred_language": "st",
                "preferred_channels": ["social", "email"],
            },
            {
                "client_id": "HOUSE-CLIENT-003",
                "name": "Masechaba",
                "budget_min": 800000,
                "budget_max": 2200000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 4,
                "free_text_preference_en": "Searching for a spacious hillside house with a yard and room for a large family.",
                "free_text_preference_st": "Ke batla ntlo e kgolo maralleng e nang le lebala bakeng sa lelapa le leholo.",
                "preferred_language": "en",
                "preferred_channels": ["email"],
            },
            {
                "client_id": "HOUSE-CLIENT-004",
                "name": "Khotso",
                "budget_min": 450000,
                "budget_max": 900000,
                "preferred_districts": ["Maseru", "Leribe"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Need a well-kept house close to schools and shops in a suburban area.",
                "free_text_preference_st": "Ke batla ntlo e hlokometsoeng hantle e haufi le dikolo le mabenkele tikolohong ya suburban.",
                "preferred_language": "st",
                "preferred_channels": ["social"],
            },
            {
                "client_id": "HOUSE-CLIENT-005",
                "name": "Lerato",
                "budget_min": 300000,
                "budget_max": 700000,
                "preferred_districts": ["Leribe", "Berea"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Looking for a practical 3 bedroom home for a young family outside central Maseru.",
                "free_text_preference_st": "Ke batla ntlo e sebetsang ya dikamore tse tharo bakeng sa lelapa le lenyane kantle ho Maseru bohareng.",
                "preferred_language": "en",
                "preferred_channels": ["email", "social"],
            },
            {
                "client_id": "HOUSE-CLIENT-006",
                "name": "Palesa",
                "budget_min": 950000,
                "budget_max": 2600000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 4,
                "free_text_preference_en": "Searching for an up-market house with good views, modern finishes, and secure parking.",
                "free_text_preference_st": "Ke batla ntlo ya maemo a hodimo e nang le pono e ntle, diphetho tsa kajeno le parking e sireletsehileng.",
                "preferred_language": "st",
                "preferred_channels": ["email"],
            },
        ]

    return pd.DataFrame(rows[: max(1, min(client_count, len(rows)))])


def run_house_recommendation_demo(
    base_dir: str | Path,
    input_csv: str | Path | None = None,
    top_n: int = 3,
    client_count: int = 6,
    listing_intent: str = "sale",
    strict_house_only: bool = True,
) -> PipelineResult:
    config = AppConfig.from_base_dir(Path(base_dir))
    config.ensure_directories()

    properties, inventory_metrics = _load_house_inventory(
        config=config,
        input_csv=input_csv,
        listing_intent=listing_intent,
        strict_house_only=strict_house_only,
    )
    clients = clean_client_dataframe(
        _build_house_demo_clients(inventory_metrics["listing_intent"], client_count)
    )
    return _run_house_recommendation_pipeline(
        config=config,
        properties=properties,
        clients=clients,
        top_n=top_n,
        artifact_prefix="house_recommendation",
        extra_metrics=inventory_metrics,
    )


def run_house_recommendation_for_clients(
    base_dir: str | Path,
    clients: pd.DataFrame,
    input_csv: str | Path | None = None,
    top_n: int = 3,
    listing_intent: str = "sale",
    strict_house_only: bool = True,
    artifact_prefix: str = "house_user_input",
    constraint_mode: str = "soft",
) -> PipelineResult:
    config = AppConfig.from_base_dir(Path(base_dir))
    config.ensure_directories()

    properties, inventory_metrics = _load_house_inventory(
        config=config,
        input_csv=input_csv,
        listing_intent=listing_intent,
        strict_house_only=strict_house_only,
    )
    cleaned_clients = clean_client_dataframe(clients)
    return _run_house_recommendation_pipeline(
        config=config,
        properties=properties,
        clients=cleaned_clients,
        top_n=top_n,
        artifact_prefix=artifact_prefix,
        extra_metrics=inventory_metrics,
        constraint_mode=constraint_mode,
    )


def run_full_pipeline(
    base_dir: str | Path,
    property_count: int = 18,
    client_count: int = 6,
    top_n: int = 3,
    seed: int = 42,
    data_mode: str = "simulated",
    live_sources: list[str] | None = None,
    live_limit: int = 2,
    include_rentals: bool = False,
    max_images_per_property: int = 2,
) -> PipelineResult:
    config = AppConfig.from_base_dir(Path(base_dir))
    config.ensure_directories()

    mode = data_mode.lower()
    if mode not in {"simulated", "live", "hybrid"}:
        raise ValueError("data_mode must be one of: simulated, live, hybrid")

    source_selection = live_sources or list(AVAILABLE_LIVE_SOURCES)
    property_frames: list[pd.DataFrame] = []
    scrape_report: dict[str, object] = {
        "data_mode": mode,
        "live_sources": source_selection if mode in {"live", "hybrid"} else [],
        "live_limit": live_limit,
        "include_rentals": include_rentals,
        "max_images_per_property": max_images_per_property,
    }

    if mode in {"simulated", "hybrid"}:
        generator = SimulatedDatasetGenerator(config.image_dir)
        raw_properties, raw_clients = generator.generate(
            property_count=property_count,
            client_count=client_count,
            seed=seed,
        )
        property_frames.append(clean_property_dataframe(raw_properties))
        clients = clean_client_dataframe(raw_clients)
    else:
        generator = SimulatedDatasetGenerator(config.image_dir)
        _, raw_clients = generator.generate(
            property_count=0,
            client_count=client_count,
            seed=seed,
        )
        clients = clean_client_dataframe(raw_clients)

    if mode in {"live", "hybrid"}:
        live_properties, live_report = scrape_live_properties(
            image_root=config.image_dir,
            sources=source_selection,
            per_source_limit=live_limit,
            include_rentals=include_rentals,
            max_images_per_property=max_images_per_property,
        )
        property_frames.append(live_properties)
        scrape_report["live_report"] = live_report

    properties = clean_property_dataframe(pd.concat(property_frames, ignore_index=True)) if property_frames else pd.DataFrame()
    if properties.empty:
        raise RuntimeError("No property records were collected. Check the selected data mode or live sources.")
    clients = clean_client_dataframe(raw_clients)

    vision = PropertyVisionAnalyzer().analyze(properties)
    properties = properties.merge(vision.dataframe, on="property_id", how="left")

    text_processor = MultilingualTextProcessor()
    text_result = text_processor.process(properties, clients)
    properties = text_result.properties
    clients = text_result.clients

    matcher = MatchingEngine(
        text_processor=text_processor,
        weights=MatchingWeights(
            structured=config.structured_weight,
            text=config.text_weight,
            vision=config.vision_weight,
        ),
    )
    matches = matcher.rank_for_all_clients(properties, clients, top_n=top_n)

    marketer = MarketingAutomation()
    campaigns = marketer.generate(matches, properties, clients)

    fusion_summary = _summarize_recommendation_fusion(matches)
    marketing_summary = _summarize_campaigns(campaigns)
    ensure_artifact_dirs(config.output_dir)

    artifact_paths = {
        "properties_csv": str(
            save_dataframe(
                properties,
                artifact_path(config.output_dir, "properties.csv"),
                json_columns=("image_paths", "amenities", "property_keywords", "text_embedding", "vision_embedding"),
            )
        ),
        "clients_csv": str(
            save_dataframe(
                clients,
                artifact_path(config.output_dir, "clients.csv"),
                json_columns=(
                    "preferred_districts",
                    "preferred_property_types",
                    "preferred_channels",
                    "client_keywords",
                    "text_embedding",
                ),
            )
        ),
        "matches_csv": str(
            save_dataframe(
                matches,
                artifact_path(config.output_dir, "matches.csv"),
                json_columns=("shared_text_cues", "recommendation_reasons"),
            )
        ),
        "campaigns_csv": str(
            save_dataframe(
                campaigns,
                artifact_path(config.output_dir, "campaigns.csv"),
                json_columns=("recommendation_reasons",),
            )
        ),
    }

    metrics = {
        "vision": vision.metrics,
        "nlp": text_result.metrics,
        "fusion": fusion_summary,
        "marketing": marketing_summary,
        "pipeline": {
            "properties": len(properties),
            "clients": len(clients),
            "matches": len(matches),
            "campaigns": len(campaigns),
        },
        "scrape": scrape_report,
    }
    artifact_paths["metrics_json"] = str(
        save_json(metrics, artifact_path(config.output_dir, "metrics.json"))
    )
    artifact_paths["fusion_json"] = str(
        save_json(fusion_summary, artifact_path(config.output_dir, "fusion_summary.json"))
    )
    artifact_paths["marketing_json"] = str(
        save_json(marketing_summary, artifact_path(config.output_dir, "marketing_summary.json"))
    )

    return PipelineResult(
        config=config,
        properties=properties,
        clients=clients,
        matches=matches,
        campaigns=campaigns,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
