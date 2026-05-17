from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from lesotho_property_ai.data.label_review import (
    apply_house_label_review,
    build_house_label_review,
    seed_house_label_review,
)


class HouseLabelReviewTests(unittest.TestCase):
    def _workspace_case_dir(self) -> Path:
        path = Path.cwd() / "generated_test_runs" / uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._cleanup_dir, path)
        return path

    def _cleanup_dir(self, path: Path) -> None:
        if not path.exists():
            return
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()

    def test_build_house_label_review_creates_flagged_review_rows(self) -> None:
        case_dir = self._workspace_case_dir()
        candidates_csv = case_dir / "candidates.csv"
        dataframe = pd.DataFrame(
            [
                {
                    "property_id": "house-1",
                    "source": "unit-test",
                    "title": "Modern 4 bedroom house",
                    "description_en": "Modern 4 bedroom house with garden and parking.",
                    "description_st": "",
                    "price": 450000,
                    "currency": "LSL",
                    "district": "Maseru",
                    "location_text": "Maseru",
                    "property_type": "House",
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "image_paths": json.dumps(["a.jpg", "b.jpg"]),
                    "listing_url": "https://example.com/house-1",
                    "condition": "Good",
                    "style": "Traditional",
                    "environment": "Urban",
                    "amenities": json.dumps(["parking"]),
                    "listing_intent": "sale",
                    "image_count": 2,
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Maseru",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": "3",
                    "split": "train",
                    "is_cnn_candidate": True,
                }
            ]
        )
        dataframe.to_csv(candidates_csv, index=False)

        paths = build_house_label_review(candidates_csv, case_dir)
        review_df = pd.read_csv(paths["review_csv"])

        self.assertEqual(len(review_df), 1)
        self.assertEqual(str(review_df.iloc[0]["suggested_cnn_bedroom_class"]), "4")
        self.assertIn("bedroom_mismatch", str(review_df.iloc[0]["review_flags"]))
        self.assertEqual(review_df.iloc[0]["review_priority"], "high")

    def test_apply_house_label_review_updates_labels_and_excludes_rows(self) -> None:
        case_dir = self._workspace_case_dir()
        candidates_csv = case_dir / "candidates.csv"
        images_csv = case_dir / "images.csv"
        review_csv = case_dir / "review.csv"

        candidates = pd.DataFrame(
            [
                {
                    "property_id": "house-1",
                    "source": "unit-test",
                    "title": "House one",
                    "description_en": "House one",
                    "description_st": "",
                    "price": 450000,
                    "currency": "LSL",
                    "district": "Maseru",
                    "location_text": "Maseru",
                    "property_type": "House",
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "image_paths": json.dumps(["a.jpg", "b.jpg"]),
                    "listing_url": "https://example.com/house-1",
                    "condition": "Good",
                    "style": "Traditional",
                    "environment": "Urban",
                    "amenities": json.dumps(["parking"]),
                    "listing_intent": "sale",
                    "image_count": 2,
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Maseru",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": "3",
                    "split": "train",
                    "is_cnn_candidate": True,
                },
                {
                    "property_id": "house-2",
                    "source": "unit-test",
                    "title": "House two",
                    "description_en": "House two",
                    "description_st": "",
                    "price": 650000,
                    "currency": "LSL",
                    "district": "Berea",
                    "location_text": "Berea",
                    "property_type": "House",
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "image_paths": json.dumps(["c.jpg"]),
                    "listing_url": "https://example.com/house-2",
                    "condition": "Good",
                    "style": "Family",
                    "environment": "Suburban",
                    "amenities": json.dumps(["yard"]),
                    "listing_intent": "sale",
                    "image_count": 1,
                    "country": "Lesotho",
                    "district_canonical": "Berea",
                    "locality": "Berea",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": "2",
                    "split": "val",
                    "is_cnn_candidate": True,
                },
            ]
        )
        images = pd.DataFrame(
            [
                {
                    "property_id": "house-1",
                    "source": "unit-test",
                    "image_index": 1,
                    "image_path": "a.jpg",
                    "country": "Lesotho",
                    "district_canonical": "Maseru",
                    "locality": "Maseru",
                    "listing_intent": "sale",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": "3",
                    "condition": "Good",
                    "style": "Traditional",
                    "environment": "Urban",
                    "split": "train",
                },
                {
                    "property_id": "house-2",
                    "source": "unit-test",
                    "image_index": 1,
                    "image_path": "c.jpg",
                    "country": "Lesotho",
                    "district_canonical": "Berea",
                    "locality": "Berea",
                    "listing_intent": "sale",
                    "cnn_property_type": "House",
                    "cnn_bedroom_class": "2",
                    "condition": "Good",
                    "style": "Family",
                    "environment": "Suburban",
                    "split": "val",
                },
            ]
        )
        review = pd.DataFrame(
            [
                {
                    "property_id": "house-1",
                    "approved_for_training": "yes",
                    "review_status": "reviewed",
                    "reviewed_bedrooms": 4,
                    "reviewed_cnn_bedroom_class": "4",
                    "reviewed_style": "Modern",
                    "reviewed_environment": "Garden",
                    "reviewed_condition": "",
                    "reviewed_price": "",
                    "reviewed_listing_intent": "",
                    "reviewer_notes": "Updated from manual check",
                },
                {
                    "property_id": "house-2",
                    "approved_for_training": "no",
                    "review_status": "exclude",
                    "reviewed_bedrooms": "",
                    "reviewed_cnn_bedroom_class": "",
                    "reviewed_style": "",
                    "reviewed_environment": "",
                    "reviewed_condition": "",
                    "reviewed_price": "",
                    "reviewed_listing_intent": "",
                    "reviewer_notes": "Exclude from training",
                },
            ]
        )

        candidates.to_csv(candidates_csv, index=False)
        images.to_csv(images_csv, index=False)
        review.to_csv(review_csv, index=False)

        paths = apply_house_label_review(candidates_csv, images_csv, review_csv, case_dir)
        reviewed_properties = pd.read_csv(paths["reviewed_properties_csv"])
        reviewed_images = pd.read_csv(paths["reviewed_images_csv"])

        house_one = reviewed_properties.loc[reviewed_properties["property_id"] == "house-1"].iloc[0]
        self.assertEqual(int(house_one["bedrooms"]), 4)
        self.assertEqual(str(house_one["cnn_bedroom_class"]), "4")
        self.assertEqual(str(house_one["style"]), "Modern")
        self.assertEqual(str(house_one["environment"]), "Garden")

        house_two = reviewed_properties.loc[reviewed_properties["property_id"] == "house-2"].iloc[0]
        self.assertEqual(str(house_two["include_in_reviewed_training"]).strip().lower(), "false")
        self.assertEqual(set(reviewed_images["property_id"]), {"house-1"})

    def test_seed_house_label_review_excludes_obvious_noise_and_updates_bedrooms(self) -> None:
        case_dir = self._workspace_case_dir()
        review_csv = case_dir / "house_label_review.csv"
        review = pd.DataFrame(
            [
                {
                    "property_id": "noise-1",
                    "approved_for_training": "yes",
                    "review_status": "pending",
                    "current_cnn_bedroom_class": "2",
                    "suggested_bedrooms_from_text": "",
                    "suggested_cnn_bedroom_class": "",
                    "review_flags": "possible_multi_unit|low_image_count",
                    "reviewed_price": "",
                    "reviewed_listing_intent": "",
                    "reviewer_notes": "",
                },
                {
                    "property_id": "bedroom-fix-1",
                    "approved_for_training": "yes",
                    "review_status": "pending",
                    "current_cnn_bedroom_class": "2",
                    "suggested_bedrooms_from_text": 3,
                    "suggested_cnn_bedroom_class": "3",
                    "review_flags": "bedroom_mismatch:2->3",
                    "reviewed_price": "",
                    "reviewed_listing_intent": "",
                    "reviewer_notes": "",
                },
            ]
        )
        review.to_csv(review_csv, index=False)

        paths = seed_house_label_review(review_csv, case_dir)
        seeded = pd.read_csv(paths["review_csv"])

        noise_row = seeded.loc[seeded["property_id"] == "noise-1"].iloc[0]
        self.assertEqual(str(noise_row["approved_for_training"]).strip().lower(), "no")
        self.assertEqual(str(noise_row["review_status"]).strip().lower(), "exclude")

        bedroom_row = seeded.loc[seeded["property_id"] == "bedroom-fix-1"].iloc[0]
        self.assertEqual(str(bedroom_row["review_status"]).strip().lower(), "reviewed")
        self.assertEqual(str(bedroom_row["reviewed_cnn_bedroom_class"]).replace(".0", ""), "3")
        self.assertEqual(int(float(bedroom_row["reviewed_bedrooms"])), 3)


if __name__ == "__main__":
    unittest.main()
