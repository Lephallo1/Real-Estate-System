from __future__ import annotations

import unittest

import pandas as pd

from lesotho_property_ai.data.cleaning import clean_property_dataframe
from lesotho_property_ai.web.helpers import serialize_property


class PropertyCleaningTests(unittest.TestCase):
    def test_html_descriptions_are_stripped_for_data_and_ui(self) -> None:
        frame = clean_property_dataframe(
            pd.DataFrame(
                [
                    {
                        "property_id": "html-row",
                        "title": "3 bedroom house",
                        "description_en": "<p>HOUSE FOR SALE</p><p><strong>Master bedroom</strong> with shower.</p>",
                        "description_st": "<p>Ntlo e ntle</p>",
                        "district": "Maseru",
                        "location_text": "Maseru",
                        "property_type": "House",
                        "bedrooms": 3,
                        "bathrooms": 2,
                        "image_paths": [],
                        "listing_url": "https://example.com/html-row",
                        "condition": "Good",
                        "style": "Modern",
                        "environment": "Suburban",
                        "amenities": ["parking"],
                        "price": 650000,
                    }
                ]
            )
        )

        cleaned = frame.iloc[0]
        card = serialize_property(cleaned)

        self.assertNotIn("<p>", cleaned["description_en"])
        self.assertNotIn("</strong>", cleaned["description_en"])
        self.assertIn("Master bedroom", cleaned["description_en"])
        self.assertNotIn("<p>", card["description"])
        self.assertIn("Master bedroom", card["description"])


if __name__ == "__main__":
    unittest.main()
