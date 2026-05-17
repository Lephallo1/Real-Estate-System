"""Helpers for organizing generated artifacts by module.

The project started with one flat `generated/artifacts` folder. As the system
grew, that became too noisy to navigate. This module keeps a clear directory
layout while still supporting older flat artifact paths during migration.
"""

from __future__ import annotations

from pathlib import Path


ARTIFACT_CATEGORIES = {
    "scraping",
    "curation",
    "review",
    "vision",
    "nlp",
    "recommendation",
    "pipeline",
}

EXACT_CATEGORY_MAP = {
    "real_only_properties_raw.csv": "scraping",
    "real_only_properties_cleaned.csv": "scraping",
    "real_only_quality_flags.csv": "scraping",
    "real_only_scrape_summary.json": "scraping",
    "scraper_raw_properties.csv": "scraping",
    "scraper_clean_properties.csv": "scraping",
    "scraper_report.json": "scraping",
    "properties_curated_master.csv": "curation",
    "properties_residential_curated.csv": "curation",
    "properties_commercial_curated.csv": "curation",
    "properties_site_land_curated.csv": "curation",
    "properties_outside_lesotho.csv": "curation",
    "properties_residential_cnn_candidates.csv": "curation",
    "properties_residential_cnn_excluded.csv": "curation",
    "properties_residential_cnn_images.csv": "curation",
    "curation_summary.json": "curation",
    "house_label_review.csv": "review",
    "house_label_review_summary.json": "review",
    "house_label_review_seeded_summary.json": "review",
    "house_label_review_applied_summary.json": "review",
    "properties_house_reviewed.csv": "review",
    "properties_house_reviewed_images.csv": "review",
    "house_nlp_metrics.json": "nlp",
    "house_nlp_query_results.csv": "nlp",
    "metrics.json": "pipeline",
    "fusion_summary.json": "pipeline",
    "marketing_summary.json": "pipeline",
    "properties.csv": "pipeline",
    "clients.csv": "pipeline",
    "matches.csv": "pipeline",
    "campaigns.csv": "pipeline",
}


def artifact_category(filename: str) -> str:
    """Return the module folder for a generated artifact filename."""

    name = Path(filename).name
    if name in EXACT_CATEGORY_MAP:
        return EXACT_CATEGORY_MAP[name]
    if name.startswith(("house_vision_", "house_bedroom_", "residential_property_type_")):
        return "vision"
    if name.startswith("house_nlp_"):
        return "nlp"
    if name.startswith(("house_recommendation_", "house_user_input_")):
        return "recommendation"
    if name.startswith("real_only_") or name.startswith("scraper_"):
        return "scraping"
    if name.startswith("house_label_review") or name.startswith("properties_house_reviewed"):
        return "review"
    if name.startswith("properties_") or name == "curation_summary.json":
        return "curation"
    return "pipeline"


def artifact_dir(artifact_root: str | Path, category: str) -> Path:
    """Return the directory for one artifact category."""

    root = Path(artifact_root)
    return root / category


def artifact_path(artifact_root: str | Path, filename: str) -> Path:
    """Return the preferred organized path for an artifact filename."""

    category = artifact_category(filename)
    return artifact_dir(artifact_root, category) / Path(filename).name


def resolve_artifact_path(artifact_root: str | Path, filename: str) -> Path:
    """Find an artifact in the new organized layout or the legacy flat layout."""

    preferred = artifact_path(artifact_root, filename)
    legacy = Path(artifact_root) / Path(filename).name
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def ensure_artifact_dirs(artifact_root: str | Path) -> None:
    """Create all known artifact category folders."""

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    for category in sorted(ARTIFACT_CATEGORIES):
        (root / category).mkdir(parents=True, exist_ok=True)
