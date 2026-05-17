from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.pipeline import run_house_recommendation_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the house-only recommendation demo using curated real property data."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project base directory containing generated artifacts.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional curated property dataset CSV. If omitted, the reviewed house dataset is used when available.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of property recommendations to keep per client.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=6,
        help="Number of demo client profiles to include.",
    )
    parser.add_argument(
        "--listing-intent",
        choices=["sale", "rent", "both"],
        default="sale",
        help="Filter the house dataset by listing intent before matching.",
    )
    parser.add_argument(
        "--allow-non-house",
        action="store_true",
        help="Disable the strict house-only filter. Leave this off for the marker-facing demo.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_house_recommendation_demo(
        base_dir=Path(args.base_dir),
        input_csv=Path(args.input_csv) if args.input_csv else None,
        top_n=args.top_n,
        client_count=args.clients,
        listing_intent=args.listing_intent,
        strict_house_only=not args.allow_non_house,
    )

    print("House recommendation demo complete")
    print(f"Properties considered : {len(result.properties)}")
    print(f"Clients profiled      : {len(result.clients)}")
    print(f"Matches generated     : {len(result.matches)}")
    print(f"Campaigns generated   : {len(result.campaigns)}")
    print(f"Listing intent        : {result.metrics['recommendation']['listing_intent']}")
    print(f"Mean top match score  : {result.metrics['recommendation']['mean_top_match_score']}")
    print()
    print("Top recommendations")
    for row in result.matches[result.matches["rank"] == 1].itertuples(index=False):
        print(
            f"- {row.client_name}: {row.property_title} ({row.district}) "
            f"score={row.overall_score}"
        )
    print()
    print("Artifacts")
    for name, path in result.artifact_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
