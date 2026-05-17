"""Build a before-vs-after comparison for the bedroom-class improvement work."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.artifacts import artifact_path, resolve_artifact_path
from lesotho_property_ai.data.repository import save_dataframe, save_json


def _coarse_bedroom_class(value: object) -> str:
    label = str(value).strip()
    if label in {"1", "2", "1-2"}:
        return "1-2"
    if label == "3":
        return "3"
    if label in {"4", "5", "5+", "4+"}:
        return "4+"
    try:
        numeric = int(float(label))
    except (TypeError, ValueError):
        return label or "unknown"
    if numeric <= 2:
        return "1-2"
    if numeric == 3:
        return "3"
    return "4+"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _property_accuracy(frame: pd.DataFrame, actual_col: str, predicted_col: str) -> float:
    working = frame[["property_id", actual_col, predicted_col]].copy()
    grouped = working.groupby("property_id", sort=False)
    total = 0
    correct = 0
    for _, group in grouped:
        actual = str(group[actual_col].iloc[0])
        predicted = group[predicted_col].astype(str).value_counts().idxmax()
        total += 1
        correct += int(actual == predicted)
    return round(correct / max(total, 1), 3)


def build_bedroom_comparison(artifact_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    main_metrics = _load_json(resolve_artifact_path(artifact_dir, "house_vision_metrics.json"))
    grouped_metrics = _load_json(resolve_artifact_path(artifact_dir, "house_bedroom_metrics.json"))
    main_predictions = pd.read_csv(resolve_artifact_path(artifact_dir, "house_vision_predictions.csv"))
    grouped_predictions = pd.read_csv(resolve_artifact_path(artifact_dir, "house_bedroom_predictions.csv"))

    main_predictions = main_predictions.copy()
    main_predictions["actual_grouped_bedroom_class"] = main_predictions["actual_cnn_bedroom_class"].map(
        _coarse_bedroom_class
    )
    main_predictions["predicted_grouped_bedroom_class"] = main_predictions[
        "predicted_cnn_bedroom_class"
    ].map(_coarse_bedroom_class)

    comparison_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        baseline_rows = main_predictions.loc[main_predictions["split"] == split].copy()
        improved_rows = grouped_predictions.loc[grouped_predictions["split"] == split].copy()

        baseline_grouped_image_accuracy = round(
            float(
                (
                    baseline_rows["actual_grouped_bedroom_class"]
                    == baseline_rows["predicted_grouped_bedroom_class"]
                ).mean()
            ),
            3,
        )
        baseline_grouped_property_accuracy = _property_accuracy(
            baseline_rows,
            "actual_grouped_bedroom_class",
            "predicted_grouped_bedroom_class",
        )

        improved_image_accuracy = float(
            grouped_metrics.get("tasks", {})
            .get("cnn_bedroom_class", {})
            .get(split, {})
            .get("image_accuracy", 0.0)
        )
        improved_property_accuracy = float(
            grouped_metrics.get("tasks", {})
            .get("cnn_bedroom_class", {})
            .get(split, {})
            .get("property_accuracy", 0.0)
        )

        comparison_rows.append(
            {
                "split": split,
                "before_model": "main_house_multitask",
                "before_label_scheme": "exact_labels_remapped_to_groups",
                "before_image_accuracy": baseline_grouped_image_accuracy,
                "before_property_accuracy": baseline_grouped_property_accuracy,
                "after_model": "dedicated_house_bedroom",
                "after_label_scheme": "coarse_groups_1-2_3_4+",
                "after_image_accuracy": round(improved_image_accuracy, 3),
                "after_property_accuracy": round(improved_property_accuracy, 3),
                "image_accuracy_gain": round(improved_image_accuracy - baseline_grouped_image_accuracy, 3),
                "property_accuracy_gain": round(improved_property_accuracy - baseline_grouped_property_accuracy, 3),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    summary = {
        "baseline_source": "house_vision_predictions.csv",
        "improved_source": "house_bedroom_predictions.csv",
        "note": "The baseline is the original multi-task bedroom head remapped into the same grouped bedroom labels used by the improved model.",
        "splits": comparison_rows,
        "main_model_exact_test_property_accuracy": float(
            main_metrics.get("tasks", {})
            .get("cnn_bedroom_class", {})
            .get("test", {})
            .get("property_accuracy", 0.0)
        ),
        "improved_grouped_test_property_accuracy": float(
            grouped_metrics.get("tasks", {})
            .get("cnn_bedroom_class", {})
            .get("test", {})
            .get("property_accuracy", 0.0)
        ),
    }
    return comparison_df, summary


def main() -> None:
    artifact_dir = Path("generated/artifacts")
    comparison_df, summary = build_bedroom_comparison(artifact_dir)
    csv_path = save_dataframe(
        comparison_df,
        artifact_path(artifact_dir, "house_bedroom_comparison.csv"),
    )
    json_path = save_json(
        summary,
        artifact_path(artifact_dir, "house_bedroom_comparison.json"),
    )

    print("Bedroom improvement comparison complete")
    print()
    print(comparison_df.to_string(index=False))
    print()
    print("Artifacts")
    print(f"- comparison_csv: {csv_path}")
    print(f"- comparison_json: {json_path}")


if __name__ == "__main__":
    main()
