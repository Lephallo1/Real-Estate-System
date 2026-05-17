"""Command-line entrypoint for the end-to-end prototype pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from lesotho_property_ai.pipeline import run_full_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Lesotho multimodal real estate AI prototype."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Working directory for generated data, images, and output artifacts.",
    )
    parser.add_argument(
        "--properties",
        type=int,
        default=18,
        help="Number of simulated properties to generate.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=6,
        help="Number of client profiles to generate.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top matches to retain per client.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset generation.",
    )
    parser.add_argument(
        "--data-mode",
        choices=["simulated", "live", "hybrid"],
        default="simulated",
        help="Choose whether to use simulated data only, real live sources only, or a hybrid mix.",
    )
    parser.add_argument(
        "--sources",
        default="creativeproperties,propmarket,sotholand,mosoholdings",
        help="Comma-separated live sources to use in live/hybrid mode.",
    )
    parser.add_argument(
        "--live-limit",
        type=int,
        default=2,
        help="Maximum number of live properties to scrape per selected source.",
    )
    parser.add_argument(
        "--include-rentals",
        action="store_true",
        help="Include rental listings when scraping live sources.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2,
        help="Maximum number of images to download per live property.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_full_pipeline(
        base_dir=Path(args.base_dir),
        property_count=args.properties,
        client_count=args.clients,
        top_n=args.top_n,
        seed=args.seed,
        data_mode=args.data_mode,
        live_sources=[item.strip() for item in args.sources.split(",") if item.strip()],
        live_limit=args.live_limit,
        include_rentals=args.include_rentals,
        max_images_per_property=args.max_images,
    )

    print("Lesotho Multimodal Real Estate AI Prototype")
    print("=" * 48)
    print(f"Data mode           : {args.data_mode}")
    print(f"Properties analysed : {len(result.properties)}")
    print(f"Clients profiled    : {len(result.clients)}")
    print(f"Matches generated   : {len(result.matches)}")
    print(f"Campaigns simulated : {len(result.campaigns)}")
    if "scrape" in result.metrics:
        sources = result.metrics["scrape"].get("live_sources", [])
        if sources:
            print(f"Live sources        : {', '.join(sources)}")
    print()
    print("Artifacts")
    for name, path in result.artifact_paths.items():
        print(f"- {name}: {path}")
    print()
    print("Vision metrics")
    for key, value in result.metrics["vision"].items():
        print(f"- {key}: {value}")
    print()
    print("NLP sanity")
    for key, value in result.metrics["nlp"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
