from __future__ import annotations

import unittest

import pandas as pd

from lesotho_property_ai.data.cleaning import clean_client_dataframe, clean_property_dataframe
from lesotho_property_ai.nlp import MultilingualTextProcessor


class MultilingualTextProcessorTests(unittest.TestCase):
    def test_tokenize_normalizes_sesotho_bedroom_and_parking_phrases(self) -> None:
        processor = MultilingualTextProcessor()

        tokens = processor.tokenize(
            "Ke batla ntlo ya dikamore tse tharo e nang le parking e sireletsehileng le serapa."
        )

        self.assertIn("house", tokens)
        self.assertIn("bedroom_3", tokens)
        self.assertIn("parking", tokens)
        self.assertIn("secure", tokens)
        self.assertIn("garden", tokens)

    def test_score_client_property_prefers_matching_family_house(self) -> None:
        properties = clean_property_dataframe(
            pd.DataFrame(
                [
                    {
                        "property_id": "house-1",
                        "title": "3 bedroom family house in Maseru",
                        "description_en": "Modern family house with secure parking, yard, and good condition.",
                        "description_st": "Ntlo ya lelapa e nang le parking e sireletsehileng le serapa.",
                        "district": "Maseru",
                        "location_text": "Maseru",
                        "property_type": "House",
                        "bedrooms": 3,
                        "bathrooms": 2,
                        "image_paths": [],
                        "listing_url": "https://example.com/house-1",
                        "style": "Modern",
                        "environment": "Suburban",
                        "condition": "Good",
                        "amenities": ["parking", "yard"],
                        "price": 650000,
                    },
                    {
                        "property_id": "house-2",
                        "title": "2 bedroom house in Berea",
                        "description_en": "Affordable house close to town with basic road access.",
                        "description_st": "Ntlo e fumaneha Berea haufi le toropo.",
                        "district": "Berea",
                        "location_text": "Berea",
                        "property_type": "House",
                        "bedrooms": 2,
                        "bathrooms": 1,
                        "image_paths": [],
                        "listing_url": "https://example.com/house-2",
                        "style": "Traditional",
                        "environment": "Urban",
                        "condition": "Fair",
                        "amenities": ["road access"],
                        "price": 420000,
                    },
                ]
            )
        )

        clients = clean_client_dataframe(
            pd.DataFrame(
                [
                    {
                        "client_id": "client-1",
                        "name": "Demo Buyer",
                        "budget_min": 500000,
                        "budget_max": 900000,
                        "preferred_districts": ["Maseru"],
                        "preferred_property_types": ["House"],
                        "preferred_bedrooms": 3,
                        "free_text_preference_en": "Looking for a family house in Maseru with secure parking and a yard.",
                        "free_text_preference_st": "Ke batla ntlo ya lelapa Maseru e nang le parking e sireletsehileng le serapa.",
                        "preferred_language": "en",
                        "preferred_channels": ["dashboard"],
                    }
                ]
            )
        )

        processor = MultilingualTextProcessor()
        result = processor.process(properties, clients)
        client_row = result.clients.iloc[0]
        strong_match = processor.score_client_property(client_row, result.properties.iloc[0])
        weak_match = processor.score_client_property(client_row, result.properties.iloc[1])

        self.assertGreater(strong_match["score"], weak_match["score"])
        self.assertIn("parking", strong_match["shared_keywords"])
        self.assertTrue(result.metrics["query_success_rate"] >= 0.5)


if __name__ == "__main__":
    unittest.main()
