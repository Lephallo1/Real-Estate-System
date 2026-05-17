from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir, artifact_path
from lesotho_property_ai.data import apply_house_label_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply reviewed house labels to build a refined CNN training dataset."
    )
    parser.add_argument(
        "--candidates-csv",
        default=str(artifact_path("generated/artifacts", "properties_residential_cnn_candidates.csv")),
        help="Curated property-level CNN candidate dataset.",
    )
    parser.add_argument(
        "--images-csv",
        default=str(artifact_path("generated/artifacts", "properties_residential_cnn_images.csv")),
        help="Curated image-level CNN dataset.",
    )
    parser.add_argument(
        "--review-csv",
        default=str(artifact_path("generated/artifacts", "house_label_review.csv")),
        help="Reviewed house-label CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "review")),
        help="Directory where reviewed datasets should be saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = apply_house_label_review(
        candidates_csv=Path(args.candidates_csv),
        images_csv=Path(args.images_csv),
        review_csv=Path(args.review_csv),
        output_dir=Path(args.output_dir),
    )
    print("Applied house label review")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
