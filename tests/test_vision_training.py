from __future__ import annotations

import gc
import shutil
import time
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd
from PIL import Image

from lesotho_property_ai.vision.training import HouseVisionTrainer


class HouseVisionTrainingTests(unittest.TestCase):
    def _workspace_case_dir(self) -> Path:
        path = Path.cwd() / "generated_test_runs" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._cleanup_dir, path)
        return path

    def _cleanup_dir(self, path: Path) -> None:
        if not path.exists():
            return
        gc.collect()
        last_error: Exception | None = None
        for _ in range(5):
            try:
                shutil.rmtree(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.2)
                gc.collect()
        if last_error is not None:
            raise last_error

    def test_fallback_house_vision_training_creates_artifacts(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        output_dir = case_dir / "outputs"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("train", "2", "Suburban", "Traditional", "Good", (120, 90, 90)),
            ("train", "2", "Suburban", "Traditional", "Good", (125, 92, 92)),
            ("train", "3", "Urban", "Family", "New", (60, 110, 160)),
            ("train", "3", "Urban", "Family", "New", (58, 112, 162)),
            ("val", "2", "Suburban", "Traditional", "Good", (122, 94, 94)),
            ("val", "3", "Urban", "Family", "New", (62, 114, 164)),
            ("test", "2", "Suburban", "Traditional", "Good", (124, 96, 96)),
            ("test", "3", "Urban", "Family", "New", (64, 116, 166)),
        ]

        for index, (split, bedroom_class, environment, style, condition, color) in enumerate(samples, start=1):
            image_path = image_dir / f"sample_{index}.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": f"house-{index // 2}",
                    "source": "unit-test",
                    "image_index": 1,
                    "image_path": str(image_path),
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Katlehong",
                    "listing_intent": "sale",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": bedroom_class,
                    "condition": condition,
                    "style": style,
                    "environment": environment,
                    "split": split,
                }
            )

        dataframe = pd.DataFrame(rows)
        trainer = HouseVisionTrainer(
            task_columns=("cnn_bedroom_class", "environment", "style", "condition"),
            target_property_type="House",
            random_state=7,
        )
        trainer.torch_available = False
        result = trainer.train(dataframe, output_dir=output_dir)

        self.assertEqual(result.metrics["mode"], "fallback_image_classifier")
        self.assertIn("cnn_bedroom_class", result.metrics["tasks"])
        self.assertEqual(len(result.predictions), len(dataframe))
        for artifact_path in result.artifact_paths.values():
            self.assertTrue(Path(artifact_path).exists())

    def test_fallback_residential_property_type_training_creates_artifacts(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        output_dir = case_dir / "outputs"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("train", "House", (120, 90, 90)),
            ("train", "House", (122, 92, 92)),
            ("train", "Townhouse", (80, 110, 150)),
            ("train", "Townhouse", (82, 112, 152)),
            ("train", "Apartment", (60, 130, 170)),
            ("train", "Apartment", (62, 132, 172)),
            ("val", "House", (124, 94, 94)),
            ("val", "Townhouse", (84, 114, 154)),
            ("val", "Apartment", (64, 134, 174)),
            ("test", "House", (126, 96, 96)),
            ("test", "Townhouse", (86, 116, 156)),
            ("test", "Apartment", (66, 136, 176)),
        ]

        for index, (split, property_type, color) in enumerate(samples, start=1):
            image_path = image_dir / f"type_sample_{index}.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": f"type-{index}",
                    "source": "unit-test",
                    "image_index": 1,
                    "image_path": str(image_path),
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Katlehong",
                    "listing_intent": "sale",
                    "cnn_property_type": property_type,
                    "cnn_bedroom_class": "3",
                    "condition": "Good",
                    "style": "Modern",
                    "environment": "Suburban",
                    "split": split,
                }
            )

        dataframe = pd.DataFrame(rows)
        trainer = HouseVisionTrainer(
            task_columns=("cnn_property_type",),
            target_property_type=None,
            random_state=7,
            artifact_prefix="residential_property_type",
        )
        trainer.torch_available = False
        result = trainer.train(dataframe, output_dir=output_dir)

        self.assertEqual(result.metrics["mode"], "fallback_image_classifier")
        self.assertIn("cnn_property_type", result.metrics["tasks"])
        self.assertEqual(result.metrics["target_property_type"], "all_residential_types")
        for artifact_path in result.artifact_paths.values():
            self.assertTrue(Path(artifact_path).exists())

    def test_fallback_house_bedroom_training_creates_artifacts(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        output_dir = case_dir / "outputs"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("train", "2", (120, 90, 90)),
            ("train", "2", (122, 92, 92)),
            ("train", "3", (80, 110, 150)),
            ("train", "3", (82, 112, 152)),
            ("train", "4", (60, 130, 170)),
            ("train", "4", (62, 132, 172)),
            ("val", "2", (124, 94, 94)),
            ("val", "3", (84, 114, 154)),
            ("val", "4", (64, 134, 174)),
            ("test", "2", (126, 96, 96)),
            ("test", "3", (86, 116, 156)),
            ("test", "4", (66, 136, 176)),
        ]

        for index, (split, bedroom_class, color) in enumerate(samples, start=1):
            image_path = image_dir / f"bedroom_sample_{index}.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": f"bed-{index}",
                    "source": "unit-test",
                    "image_index": 1,
                    "image_path": str(image_path),
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Katlehong",
                    "listing_intent": "sale",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": bedroom_class,
                    "condition": "Good",
                    "style": "Modern",
                    "environment": "Suburban",
                    "split": split,
                }
            )

        dataframe = pd.DataFrame(rows)
        trainer = HouseVisionTrainer(
            task_columns=("cnn_bedroom_class",),
            target_property_type="House",
            random_state=7,
            artifact_prefix="house_bedroom",
        )
        trainer.torch_available = False
        result = trainer.train(dataframe, output_dir=output_dir)

        self.assertEqual(result.metrics["mode"], "fallback_image_classifier")
        self.assertIn("cnn_bedroom_class", result.metrics["tasks"])
        self.assertEqual(result.metrics["target_property_type"], "House")
        for artifact_path in result.artifact_paths.values():
            self.assertTrue(Path(artifact_path).exists())


if __name__ == "__main__":
    unittest.main()
