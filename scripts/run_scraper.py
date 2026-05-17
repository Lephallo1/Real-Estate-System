"""CLI entrypoint for collecting real property data from supported live sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir
from lesotho_property_ai.data import (
    AVAILABLE_LIVE_SOURCES,
    collect_live_property_records,
    scrape_live_properties,
)
from lesotho_property_ai.data.repository import save_dataframe


def run_scraper(
    output_dir: Path,
    image_root: Path,
    sources: list[str],
    live_limit: int,
    include_rentals: bool,
    max_images: int,
) -> dict[str, str]:
    """Save both the raw scrape and the cleaned scrape for inspection."""

    raw_df, raw_report = collect_live_property_records(
        image_root=image_root,
        sources=sources,
        per_source_limit=live_limit,
        include_rentals=include_rentals,
        max_images_per_property=max_images,
    )
    clean_df, clean_report = scrape_live_properties(
        image_root=image_root,
        sources=sources,
        per_source_limit=live_limit,
        include_rentals=include_rentals,
        max_images_per_property=max_images,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = save_dataframe(raw_df, output_dir / "scraper_raw_properties.csv", json_columns=("image_paths", "amenities"))
    clean_csv = save_dataframe(
        clean_df,
        output_dir / "scraper_clean_properties.csv",
        json_columns=("image_paths", "amenities"),
    )
    report_json = output_dir / "scraper_report.json"
    report_json.write_text(
        json.dumps(
            {
                "raw_report": raw_report,
                "clean_report": clean_report,
                "sources": sources,
                "live_limit": live_limit,
                "include_rentals": include_rentals,
                "max_images": max_images,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "raw_csv": str(raw_csv),
        "clean_csv": str(clean_csv),
        "report_json": str(report_json),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Lesotho real-estate web scrapers and save raw + cleaned outputs."
    )
    parser.add_argument(
        "--sources",
        default="creativeproperties,propmarket,sotholand,lesothohousing,mestech,mosoholdings",
        help="Comma-separated list of live sources.",
    )
    parser.add_argument(
        "--live-limit",
        type=int,
        default=50,
        help="Maximum properties per source.",
    )
    parser.add_argument(
        "--include-rentals",
        action="store_true",
        help="Include rental listings in the scrape.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum images to download per property.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "scraping")),
        help="Directory where scraped outputs should be saved.",
    )
    parser.add_argument(
        "--image-root",
        default="generated/images",
        help="Directory where listing images should be downloaded.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    invalid = [source for source in sources if source not in AVAILABLE_LIVE_SOURCES]
    if invalid:
        raise ValueError(
            f"Unsupported sources: {', '.join(invalid)}. Available: {', '.join(AVAILABLE_LIVE_SOURCES)}"
        )

    paths = run_scraper(
        output_dir=Path(args.output_dir),
        image_root=Path(args.image_root),
        sources=sources,
        live_limit=args.live_limit,
        include_rentals=args.include_rentals,
        max_images=args.max_images,
    )
    print("Scraper run complete")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
