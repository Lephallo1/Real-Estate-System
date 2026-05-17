from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir, artifact_path
from lesotho_property_ai.data import prepare_modeling_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare curated modeling datasets from the real-only scraped property CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=str(artifact_path("generated/artifacts", "real_only_properties_cleaned.csv")),
        help="Input property CSV to curate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "curation")),
        help="Directory where curated datasets should be saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = prepare_modeling_datasets(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )
    print("Prepared curated modeling datasets")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
