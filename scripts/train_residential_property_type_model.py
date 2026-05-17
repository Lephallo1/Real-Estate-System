"""CLI entrypoint for the auxiliary residential property-type classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_dir, resolve_artifact_path
from lesotho_property_ai.vision import train_residential_property_type_model


def _resolve_default_input_csv() -> Path:
    candidates = [
        resolve_artifact_path("generated/artifacts", "properties_residential_cnn_images.csv"),
        resolve_artifact_path("generated/artifacts", "properties_house_reviewed_images.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No curated residential image dataset was found. Expected candidate image CSV artifacts."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the residential property-type classifier for House, Townhouse, and Apartment."
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional curated image-level dataset CSV. Defaults to the residential candidate image dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(artifact_dir("generated/artifacts", "vision")),
        help="Directory where training outputs should be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs for the CNN path when PyTorch is available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the CNN path when PyTorch is available.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained CNN weights when PyTorch is available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv) if args.input_csv else _resolve_default_input_csv()
    result = train_residential_property_type_model(
        image_dataset_csv=input_csv,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_pretrained=not args.no_pretrained,
    )
    print("Residential property-type training complete")
    print(f"Mode: {result.metrics['mode']}")
    for task, split_metrics in result.metrics["tasks"].items():
        print(f"Task: {task}")
        for split, values in split_metrics.items():
            print(
                f"  {split}: image_accuracy={values['image_accuracy']} "
                f"property_accuracy={values['property_accuracy']}"
            )
    for name, path in result.artifact_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
