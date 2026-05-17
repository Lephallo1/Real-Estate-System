from __future__ import annotations

import unittest

import pandas as pd

from lesotho_property_ai.data.curation import build_image_level_dataset, curate_property_dataset


class DatasetCurationTests(unittest.TestCase):
    def test_curate_property_dataset_separates_residential_and_outside_rows(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "property_id": "res-1",
                    "source": "mosoholdings",
                    "title": "Duplex",
                    "description_en": "Bedrooms: 3 / Baths: 2 / Fully fitted",
                    "description_st": "",
                    "price": 7000,
                    "currency": "LSL",
                    "district": "Thetsane West",
                    "location_text": "Thetsane West",
                    "property_type": "House",
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "image_paths": ["a.jpg", "b.jpg"],
                    "listing_url": "https://example.com/res-1",
                    "condition": "Good",
                    "style": "Family",
                    "environment": "Suburban",
                    "amenities": ["parking"],
                    "listing_intent": "rent",
                },
                {
                    "property_id": "com-1",
                    "source": "propmarket",
                    "title": "Guest House",
                    "description_en": "Guest house in Maseru with conference rooms and parking.",
                    "description_st": "",
                    "price": 800,
                    "currency": "LSL",
                    "district": "Maseru",
                    "location_text": "Katlehong, Maseru",
                    "property_type": "House",
                    "bedrooms": 2,
                    "bathrooms": 1,
                    "image_paths": ["c.jpg"],
                    "listing_url": "https://example.com/com-1",
                    "condition": "Good",
                    "style": "Traditional",
                    "environment": "Urban",
                    "amenities": [],
                    "listing_intent": "sale",
                },
                {
                    "property_id": "out-1",
                    "source": "propmarket",
                    "title": "4 Bedrooms, House",
                    "description_en": "Double storey house in Ladybrand.",
                    "description_st": "",
                    "price": 3500000,
                    "currency": "LSL",
                    "district": "Mantsopa Local Municipality",
                    "location_text": "Ladybrand, Mantsopa Local Municipality",
                    "property_type": "House",
                    "bedrooms": 4,
                    "bathrooms": 2,
                    "image_paths": ["d.jpg"],
                    "listing_url": "https://example.com/out-1",
                    "condition": "Good",
                    "style": "Traditional",
                    "environment": "Suburban",
                    "amenities": [],
                    "listing_intent": "sale",
                },
            ]
        )

        curated, summary = curate_property_dataset(dataframe)

        residential = curated.loc[curated["property_id"] == "res-1"].iloc[0]
        commercial = curated.loc[curated["property_id"] == "com-1"].iloc[0]
        outside = curated.loc[curated["property_id"] == "out-1"].iloc[0]

        self.assertEqual(residential["district_canonical"], "Maseru")
        self.assertEqual(residential["use_bucket"], "residential")
        self.assertTrue(residential["is_cnn_candidate"])
        self.assertEqual(residential["cnn_bedroom_class"], "3")

        self.assertEqual(commercial["use_bucket"], "commercial")
        self.assertFalse(commercial["is_cnn_candidate"])

        self.assertEqual(outside["country"], "South Africa")
        self.assertFalse(outside["is_cnn_candidate"])
        self.assertIn("outside_lesotho", outside["cnn_exclusion_reasons"])

        self.assertEqual(summary["cnn_candidate_rows"], 1)

    def test_build_image_level_dataset_explodes_candidate_images(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "property_id": "res-2",
                    "source": "creativeproperties",
                    "title": "Family House",
                    "description_en": "Three bedrooms and two bathrooms.",
                    "description_st": "",
                    "price": 500000,
                    "currency": "LSL",
                    "district": "Maseru",
                    "location_text": "Katlehong, Maseru",
                    "property_type": "House",
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "image_paths": ["x.jpg", "y.jpg", "z.jpg"],
                    "listing_url": "https://example.com/res-2",
                    "condition": "Good",
                    "style": "Family",
                    "environment": "Urban",
                    "amenities": [],
                    "listing_intent": "sale",
                }
            ]
        )
        curated, _ = curate_property_dataset(dataframe)
        images = build_image_level_dataset(curated)

        self.assertEqual(len(images), 3)
        self.assertTrue((images["property_id"] == "res-2").all())
        self.assertTrue((images["split"] == curated.iloc[0]["split"]).all())


if __name__ == "__main__":
    unittest.main()
