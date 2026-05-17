from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir, artifact_path
from lesotho_property_ai.data import build_house_label_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a review sheet for house-label cleanup before CNN retraining."
    )
    parser.add_argument(
        "--input-csv",
        default=str(artifact_path("generated/artifacts", "properties_residential_cnn_candidates.csv")),
        help="Curated property-level CNN candidate dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "review")),
        help="Directory where the review CSV and summary should be saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = build_house_label_review(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )
    print("Prepared house label review files")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
