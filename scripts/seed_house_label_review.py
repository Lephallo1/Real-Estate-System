from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir, artifact_path
from lesotho_property_ai.data import seed_house_label_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply first-pass automatic review decisions to the house label review sheet."
    )
    parser.add_argument(
        "--review-csv",
        default=str(artifact_path("generated/artifacts", "house_label_review.csv")),
        help="House label review CSV to update in place.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "review")),
        help="Directory where the seeded review summary should be saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = seed_house_label_review(
        review_csv=Path(args.review_csv),
        output_dir=Path(args.output_dir),
    )
    print("Seeded house label review")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
