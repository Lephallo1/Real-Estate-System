"""Shared admin action definitions for dashboard background jobs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdminActionSpec:
    key: str
    label: str
    description: str
    commands: tuple[tuple[str, tuple[str, ...]], ...]
    local_only: bool = False


ADMIN_ACTIONS: dict[str, AdminActionSpec] = {
    "scraper": AdminActionSpec(
        key="scraper",
        label="Run Scraping Job",
        description="Collect fresh live listings and refresh the cleaned scraping artifacts.",
        commands=(("run_scraper.py", ("--live-limit", "10", "--include-rentals", "--max-images", "3")),),
    ),
    "prepare": AdminActionSpec(
        key="prepare",
        label="Prepare Dataset",
        description="Rebuild the cleaned residential and CNN-ready property datasets.",
        commands=(("prepare_modeling_dataset.py", ()),),
    ),
    "nlp": AdminActionSpec(
        key="nlp",
        label="Evaluate NLP",
        description="Run the bilingual NLP evaluation and refresh the language metrics artifacts.",
        commands=(("evaluate_nlp_module.py", ()),),
    ),
    "vision": AdminActionSpec(
        key="vision",
        label="Train Vision Models",
        description="Train the vision support models and refresh the evaluation outputs. This remains local-only.",
        commands=(
            ("train_house_vision_model.py", ()),
            ("train_house_bedroom_model.py", ()),
            ("train_residential_property_type_model.py", ()),
            ("evaluate_bedroom_improvement.py", ()),
        ),
        local_only=True,
    ),
    "recommendations": AdminActionSpec(
        key="recommendations",
        label="Refresh Recommendation Artifacts",
        description="Re-run the house recommendation demo to refresh matching, fusion, and campaign artifacts.",
        commands=(("run_house_recommendation_demo.py", ("--listing-intent", "sale", "--top-n", "3", "--clients", "6")),),
    ),
}

