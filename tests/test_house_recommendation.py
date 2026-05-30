from __future__ import annotations

import gc
import shutil
import time
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd
from PIL import Image

from lesotho_property_ai.data.repository import save_dataframe
from lesotho_property_ai.pipeline import (
    run_house_recommendation_demo,
    run_house_recommendation_for_clients,
)


class HouseRecommendationPipelineTests(unittest.TestCase):
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

    def test_house_recommendation_demo_runs_on_house_only_sale_data(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("sale-house-1", "sale", "Maseru", "House", 3, 2, "Good", "Modern", "Suburban", (120, 100, 80)),
            ("sale-house-2", "sale", "Berea", "House", 2, 1, "Good", "Traditional", "Garden", (80, 120, 80)),
            ("rent-house-1", "rent", "Maseru", "House", 4, 3, "New", "Family", "Urban", (70, 90, 140)),
        ]

        for property_id, intent, district, property_type, bedrooms, bathrooms, condition, style, environment, color in samples:
            property_dir = image_dir / property_id
            property_dir.mkdir(parents=True, exist_ok=True)
            image_path = property_dir / "front.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": property_id,
                    "source": "unit-test",
                    "title": f"{bedrooms} bedroom {property_type} in {district}",
                    "description_en": f"{bedrooms} bedroom {property_type.lower()} with parking and family yard.",
                    "description_st": f"Ntlo ya dikamore tse {bedrooms} e nang le parking le lebala.",
                    "price": 450000 if intent == "sale" else 12000,
                    "currency": "LSL",
                    "district": district,
                    "location_text": district,
                    "property_type": property_type,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "image_paths": [str(image_path)],
                    "listing_url": f"https://example.com/{property_id}",
                    "condition": condition,
                    "style": style,
                    "environment": environment,
                    "amenities": ["parking", "yard"],
                    "listing_intent": intent,
                    "country": "Lesotho",
                    "district_canonical": district,
                    "cnn_property_type": property_type,
                    "cnn_bedroom_class": "5+" if bedrooms >= 5 else str(bedrooms),
                    "split": "train",
                    "cnn_exclusion_reasons": "",
                    "is_residential_curated": True,
                    "is_cnn_candidate": True,
                }
            )

        input_csv = case_dir / "properties_residential_cnn_candidates.csv"
        save_dataframe(pd.DataFrame(rows), input_csv, json_columns=("image_paths", "amenities"))

        result = run_house_recommendation_demo(
            base_dir=case_dir,
            input_csv=input_csv,
            top_n=2,
            client_count=3,
            listing_intent="sale",
            strict_house_only=True,
        )

        self.assertEqual(len(result.properties), 2)
        self.assertTrue((result.properties["listing_intent"] == "sale").all())
        self.assertTrue((result.properties["property_type"] == "House").all())
        self.assertEqual(len(result.matches), len(result.clients) * 2)
        self.assertEqual(len(result.campaigns), len(result.matches))
        self.assertIn("recommendation", result.metrics)
        self.assertIn("fusion", result.metrics)
        self.assertIn("structured_weight_used", result.matches.columns)
        self.assertIn("text_weight_used", result.matches.columns)
        self.assertIn("vision_weight_used", result.matches.columns)
        self.assertIn("recommendation_reasons", result.matches.columns)
        first_match = result.matches.iloc[0]
        self.assertAlmostEqual(
            float(first_match["structured_weight_used"])
            + float(first_match["text_weight_used"])
            + float(first_match["vision_weight_used"]),
            1.0,
            places=3,
        )
        for artifact_path in result.artifact_paths.values():
            self.assertTrue(Path(artifact_path).exists())

    def test_custom_house_recommendation_accepts_user_client_input(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("sale-house-1", "sale", "Maseru", 3, 2, "Good", "Modern", "Suburban", 650000, (120, 100, 80)),
            ("sale-house-2", "sale", "Berea", 2, 1, "Good", "Traditional", "Garden", 420000, (80, 120, 80)),
            ("sale-house-3", "sale", "Maseru", 4, 3, "New", "Family", "Urban", 980000, (70, 90, 140)),
        ]

        for property_id, intent, district, bedrooms, bathrooms, condition, style, environment, price, color in samples:
            property_dir = image_dir / property_id
            property_dir.mkdir(parents=True, exist_ok=True)
            image_path = property_dir / "front.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": property_id,
                    "source": "unit-test",
                    "title": f"{bedrooms} bedroom house in {district}",
                    "description_en": f"Family house in {district} with parking and good condition.",
                    "description_st": f"Ntlo ya lelapa {district} e nang le parking le boemo bo botle.",
                    "price": price,
                    "currency": "LSL",
                    "district": district,
                    "location_text": district,
                    "property_type": "House",
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "image_paths": [str(image_path)],
                    "listing_url": f"https://example.com/{property_id}",
                    "condition": condition,
                    "style": style,
                    "environment": environment,
                    "amenities": ["parking"],
                    "listing_intent": intent,
                    "country": "Lesotho",
                    "district_canonical": district,
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": str(bedrooms),
                    "split": "train",
                    "cnn_exclusion_reasons": "",
                    "is_residential_curated": True,
                    "is_cnn_candidate": True,
                }
            )

        input_csv = case_dir / "properties_residential_cnn_candidates.csv"
        save_dataframe(pd.DataFrame(rows), input_csv, json_columns=("image_paths", "amenities"))

        clients = pd.DataFrame(
            [
                {
                    "client_id": "USER-1",
                    "name": "Lecturer Demo",
                    "budget_min": 500000,
                    "budget_max": 1000000,
                    "preferred_districts": ["Maseru"],
                    "preferred_property_types": ["House"],
                    "preferred_bedrooms": 3,
                    "free_text_preference_en": "Looking for a house in Maseru with parking and good condition.",
                    "free_text_preference_st": "Ke batla ntlo Maseru e nang le parking le boemo bo botle.",
                    "preferred_language": "en",
                    "preferred_channels": ["dashboard"],
                }
            ]
        )

        result = run_house_recommendation_for_clients(
            base_dir=case_dir,
            clients=clients,
            input_csv=input_csv,
            top_n=2,
            listing_intent="sale",
            strict_house_only=True,
            artifact_prefix="house_user_input",
        )

        self.assertEqual(len(result.clients), 1)
        self.assertEqual(len(result.matches), 2)
        self.assertEqual(result.metrics["recommendation"]["listing_intent"], "sale")
        self.assertTrue((result.matches["district"] == "Maseru").any())
        self.assertTrue(
            result.matches["recommendation_reasons"].map(lambda reasons: len(reasons) >= 1).all()
        )
        self.assertTrue(Path(result.artifact_paths["matches_csv"]).exists())

    def test_blank_text_preferences_exclude_over_budget_homes_and_prioritize_in_range(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("fit-house", "Maseru", 3, 650000, (120, 100, 80)),
            ("cheap-house", "Maseru", 3, 250000, (100, 120, 80)),
            ("expensive-house", "Maseru", 3, 1200000, (80, 100, 140)),
        ]

        for property_id, district, bedrooms, price, color in samples:
            property_dir = image_dir / property_id
            property_dir.mkdir(parents=True, exist_ok=True)
            image_path = property_dir / "front.png"
            Image.new("RGB", (64, 64), color=color).save(image_path)
            rows.append(
                {
                    "property_id": property_id,
                    "source": "unit-test",
                    "title": f"{bedrooms} bedroom house in {district}",
                    "description_en": f"Family house in {district} with parking and yard.",
                    "description_st": f"Ntlo ya lelapa {district} e nang le parking le lebala.",
                    "price": price,
                    "currency": "LSL",
                    "district": district,
                    "location_text": district,
                    "property_type": "House",
                    "bedrooms": bedrooms,
                    "bathrooms": 2,
                    "image_paths": [str(image_path)],
                    "listing_url": f"https://example.com/{property_id}",
                    "condition": "Good",
                    "style": "Modern",
                    "environment": "Suburban",
                    "amenities": ["parking", "yard"],
                    "listing_intent": "sale",
                    "country": "Lesotho",
                    "district_canonical": district,
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": str(bedrooms),
                    "split": "train",
                    "cnn_exclusion_reasons": "",
                    "is_residential_curated": True,
                    "is_cnn_candidate": True,
                }
            )

        input_csv = case_dir / "properties_residential_cnn_candidates.csv"
        save_dataframe(pd.DataFrame(rows), input_csv, json_columns=("image_paths", "amenities"))

        clients = pd.DataFrame(
            [
                {
                    "client_id": "USER-2",
                    "name": "Budget Buyer",
                    "budget_min": 500000,
                    "budget_max": 800000,
                    "preferred_districts": ["Maseru"],
                    "preferred_property_types": ["House"],
                    "preferred_bedrooms": 3,
                    "free_text_preference_en": "",
                    "free_text_preference_st": "",
                    "preferred_language": "en",
                    "preferred_channels": ["dashboard"],
                }
            ]
        )

        result = run_house_recommendation_for_clients(
            base_dir=case_dir,
            clients=clients,
            input_csv=input_csv,
            top_n=3,
            listing_intent="sale",
            strict_house_only=True,
            artifact_prefix="house_user_input",
        )

        self.assertTrue((result.matches["price"] <= 800000).all())
        self.assertEqual(result.matches.iloc[0]["price"], 650000)
        self.assertIn(
            "structured fallback respected the budget ceiling",
            result.matches.iloc[0]["recommendation_reasons"],
        )

    def test_strict_customer_constraints_require_budget_district_and_exact_bedrooms(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("exact-house", "Maseru", 3, 750000),
            ("wrong-bedroom-house", "Maseru", 4, 700000),
            ("wrong-district-house", "Berea", 3, 650000),
            ("over-budget-house", "Maseru", 3, 1200000),
        ]
        for property_id, district, bedrooms, price in samples:
            property_dir = image_dir / property_id
            property_dir.mkdir(parents=True, exist_ok=True)
            image_path = property_dir / "front.png"
            Image.new("RGB", (64, 64), color=(120, 100, 80)).save(image_path)
            rows.append(
                {
                    "property_id": property_id,
                    "source": "unit-test",
                    "title": f"{bedrooms} bedroom house in {district}",
                    "description_en": f"Family house in {district} with parking.",
                    "description_st": f"Ntlo ya lelapa {district} e nang le parking.",
                    "price": price,
                    "currency": "LSL",
                    "district": district,
                    "location_text": district,
                    "property_type": "House",
                    "bedrooms": bedrooms,
                    "bathrooms": 2,
                    "image_paths": [str(image_path)],
                    "listing_url": f"https://example.com/{property_id}",
                    "condition": "Good",
                    "style": "Modern",
                    "environment": "Suburban",
                    "amenities": ["parking"],
                    "listing_intent": "sale",
                    "country": "Lesotho",
                    "district_canonical": district,
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": str(bedrooms),
                    "split": "train",
                    "cnn_exclusion_reasons": "",
                    "is_residential_curated": True,
                    "is_cnn_candidate": True,
                }
            )

        input_csv = case_dir / "properties_residential_cnn_candidates.csv"
        save_dataframe(pd.DataFrame(rows), input_csv, json_columns=("image_paths", "amenities"))
        clients = pd.DataFrame(
            [
                {
                    "client_id": "USER-STRICT",
                    "name": "Strict Buyer",
                    "budget_min": 500000,
                    "budget_max": 900000,
                    "preferred_districts": ["Maseru"],
                    "preferred_property_types": ["House"],
                    "preferred_bedrooms": 3,
                    "free_text_preference_en": "Need a family house with parking.",
                    "free_text_preference_st": "",
                    "preferred_language": "en",
                    "preferred_channels": ["dashboard"],
                }
            ]
        )

        result = run_house_recommendation_for_clients(
            base_dir=case_dir,
            clients=clients,
            input_csv=input_csv,
            top_n=5,
            listing_intent="sale",
            strict_house_only=True,
            artifact_prefix="strict_user_input",
            constraint_mode="strict",
        )

        self.assertEqual(result.matches["property_id"].tolist(), ["exact-house"])

        near_result = run_house_recommendation_for_clients(
            base_dir=case_dir,
            clients=clients,
            input_csv=input_csv,
            top_n=5,
            listing_intent="sale",
            strict_house_only=True,
            artifact_prefix="strict_user_input_near",
            constraint_mode="near",
        )

        self.assertEqual(near_result.matches["property_id"].tolist(), ["wrong-bedroom-house"])

    def test_house_recommendation_filters_noisy_rows_and_normalizes_titles(self) -> None:
        case_dir = self._workspace_case_dir()
        image_dir = case_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        samples = [
            ("good-house", "3 Bedrooms, House", "Maseru", 3, "sale", True),
            ("one-unit-house", "1 Unit, 4 Bedrooms, House", "Berea", 4, "sale", True),
            ("ground-floor-house", "Ground Floor", "Maseru", 3, "sale", True),
            ("bad-district-house", "1 Unit, 3 Bedrooms, House", "000.00", 3, "sale", True),
            ("excluded-review-house", "4 Bedrooms, House", "Maseru", 4, "sale", False),
        ]

        for property_id, title, district, bedrooms, intent, include_in_reviewed_training in samples:
            property_dir = image_dir / property_id
            property_dir.mkdir(parents=True, exist_ok=True)
            image_path = property_dir / "front.png"
            Image.new("RGB", (64, 64), color=(120, 100, 80)).save(image_path)
            rows.append(
                {
                    "property_id": property_id,
                    "source": "unit-test",
                    "title": title,
                    "description_en": "House with parking and family yard.",
                    "description_st": "Ntlo e nang le parking le lebala.",
                    "price": 650000,
                    "currency": "LSL",
                    "district": district,
                    "location_text": district,
                    "property_type": "House",
                    "bedrooms": bedrooms,
                    "bathrooms": 2,
                    "image_paths": [str(image_path)],
                    "listing_url": f"https://example.com/{property_id}",
                    "condition": "Good",
                    "style": "Modern",
                    "environment": "Suburban",
                    "amenities": ["parking", "yard"],
                    "listing_intent": intent,
                    "country": "Lesotho",
                    "district_canonical": district,
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": str(bedrooms),
                    "split": "train",
                    "cnn_exclusion_reasons": "",
                    "is_residential_curated": True,
                    "is_cnn_candidate": True,
                    "include_in_reviewed_training": include_in_reviewed_training,
                }
            )

        input_csv = case_dir / "properties_house_reviewed.csv"
        save_dataframe(
            pd.DataFrame(rows),
            input_csv,
            json_columns=("image_paths", "amenities"),
        )

        result = run_house_recommendation_demo(
            base_dir=case_dir,
            input_csv=input_csv,
            top_n=1,
            client_count=1,
            listing_intent="sale",
            strict_house_only=True,
        )

        remaining_titles = set(result.properties["title"].tolist())
        self.assertEqual(remaining_titles, {"3-Bedroom House", "4-Bedroom House"})
        self.assertNotIn("Ground Floor", remaining_titles)
        self.assertFalse(result.properties["district"].astype(str).str.fullmatch(r"\d+(?:\.\d+)?").any())


if __name__ == "__main__":
    unittest.main()
