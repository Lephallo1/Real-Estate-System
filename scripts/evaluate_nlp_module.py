"""CLI entrypoint for the bilingual NLP evaluation pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_path, resolve_artifact_path
from lesotho_property_ai.data.cleaning import clean_client_dataframe, clean_property_dataframe
from lesotho_property_ai.data.repository import save_dataframe, save_json
from lesotho_property_ai.nlp import MultilingualTextProcessor


def _resolve_input_csv(base_dir: Path, input_csv: str | None) -> Path:
    if input_csv:
        path = Path(input_csv)
        if not path.is_absolute():
            path = base_dir / input_csv
        if not path.exists():
            raise FileNotFoundError(f"NLP input dataset not found: {path}")
        return path

    candidates = [
        resolve_artifact_path(base_dir / "generated" / "artifacts", "properties_house_reviewed.csv"),
        resolve_artifact_path(base_dir / "generated" / "artifacts", "properties_residential_cnn_candidates.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No reviewed or candidate house property dataset was found.")


def _build_demo_nlp_queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "client_id": "NLP-QUERY-001",
                "name": "English Family Query",
                "budget_min": 300000,
                "budget_max": 1200000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Looking for a family house in Maseru with secure parking, a yard, and good condition.",
                "free_text_preference_st": "",
                "preferred_language": "en",
                "preferred_channels": ["dashboard"],
            },
            {
                "client_id": "NLP-QUERY-002",
                "name": "Sesotho Family Query",
                "budget_min": 300000,
                "budget_max": 1200000,
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "",
                "free_text_preference_st": "Ke batla ntlo ya lelapa Maseru e nang le parking e sireletsehileng le serapa.",
                "preferred_language": "st",
                "preferred_channels": ["dashboard"],
            },
            {
                "client_id": "NLP-QUERY-003",
                "name": "Berea Garden Query",
                "budget_min": 250000,
                "budget_max": 900000,
                "preferred_districts": ["Berea"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 2,
                "free_text_preference_en": "Need a 2 bedroom house in Berea with a garden and road access.",
                "free_text_preference_st": "",
                "preferred_language": "en",
                "preferred_channels": ["dashboard"],
            },
            {
                "client_id": "NLP-QUERY-004",
                "name": "Modern View Query",
                "budget_min": 700000,
                "budget_max": 2200000,
                "preferred_districts": ["Maseru", "Berea"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 4,
                "free_text_preference_en": "Searching for a modern house with good views, parking, and quiet surroundings.",
                "free_text_preference_st": "Ke batla ntlo ya kajeno e nang le pono e ntle le tikoloho e kgutsitseng.",
                "preferred_language": "en",
                "preferred_channels": ["dashboard"],
            },
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bilingual NLP evaluation on the curated house dataset."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project base directory.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional property dataset CSV. Defaults to the reviewed house dataset when available.",
    )
    parser.add_argument(
        "--listing-intent",
        choices=["sale", "rent", "both"],
        default="sale",
        help="Filter properties by intent before NLP evaluation.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top properties to keep per query.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir)
    artifact_root = base_dir / "generated" / "artifacts"
    artifact_path(artifact_root, "house_nlp_metrics.json").parent.mkdir(parents=True, exist_ok=True)

    input_csv = _resolve_input_csv(base_dir, args.input_csv)
    properties = clean_property_dataframe(pd.read_csv(input_csv))
    if "cnn_property_type" in properties.columns:
        properties = properties.loc[
            properties["cnn_property_type"].fillna(properties["property_type"]).astype(str).str.lower().eq("house")
        ].copy()
    if args.listing_intent != "both" and "listing_intent" in properties.columns:
        properties = properties.loc[
            properties["listing_intent"].fillna("sale").astype(str).str.lower().eq(args.listing_intent)
        ].copy()
    if properties.empty:
        raise RuntimeError("No house properties matched the NLP evaluation filters.")

    clients = clean_client_dataframe(_build_demo_nlp_queries())
    processor = MultilingualTextProcessor()
    result = processor.process(properties, clients)

    ranking_rows: list[dict[str, object]] = []
    for client in result.clients.itertuples(index=False):
        scored_rows: list[dict[str, object]] = []
        for property_row in result.properties.itertuples(index=False):
            score = processor.score_client_property(client, property_row)
            scored_rows.append(
                {
                    "query_id": client.client_id,
                    "query_name": client.name,
                    "property_id": property_row.property_id,
                    "property_title": property_row.title,
                    "district": property_row.district,
                    "price": property_row.price,
                    "text_score": score["score"],
                    "cosine": score["cosine"],
                    "keyword_overlap": score["keyword_overlap"],
                    "signal_alignment": score["signal_alignment"],
                    "shared_keywords": score["shared_keywords"],
                }
            )
        top_rows = sorted(scored_rows, key=lambda item: item["text_score"], reverse=True)[: args.top_n]
        for rank, item in enumerate(top_rows, start=1):
            item["rank"] = rank
            ranking_rows.append(item)

    rankings = pd.DataFrame(ranking_rows)
    metrics = dict(result.metrics)
    metrics.update(
        {
            "properties_evaluated": int(len(result.properties)),
            "queries_evaluated": int(len(result.clients)),
            "listing_intent": args.listing_intent,
            "top_n": int(args.top_n),
            "input_csv": str(input_csv),
        }
    )

    rankings_csv = save_dataframe(
        rankings,
        artifact_path(artifact_root, "house_nlp_query_results.csv"),
        json_columns=("shared_keywords",),
    )
    metrics_json = save_json(
        metrics,
        artifact_path(artifact_root, "house_nlp_metrics.json"),
    )

    print("NLP evaluation complete")
    print(f"Properties evaluated : {len(result.properties)}")
    print(f"Queries evaluated    : {len(result.clients)}")
    print(f"Vocabulary size      : {metrics.get('vocabulary_size', 0)}")
    print(f"Query success rate   : {metrics.get('query_success_rate', 0.0)}")
    print()
    print("Top query matches")
    for row in rankings.loc[rankings["rank"] == 1].itertuples(index=False):
        print(f"- {row.query_name}: {row.property_title} ({row.district}) text_score={row.text_score}")
    print()
    print("Artifacts")
    print(f"- rankings_csv: {rankings_csv}")
    print(f"- metrics_json: {metrics_json}")


if __name__ == "__main__":
    main()
