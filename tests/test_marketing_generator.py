from __future__ import annotations

import unittest

import pandas as pd

from lesotho_property_ai.marketing import MarketingAutomation


class MarketingAutomationTests(unittest.TestCase):
    def test_generate_uses_cleaned_titles_and_more_natural_message_text(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "client_id": "client-1",
                    "client_name": "Naleli",
                    "property_id": "property-1",
                    "property_title": "1 Unit, 3 Bedrooms, House",
                    "district": "Maseru",
                    "recommendation_reasons": [
                        "budget is closely aligned",
                        "it matches the preferred district",
                    ],
                    "overall_score": 0.684,
                    "rank": 1,
                }
            ]
        )
        properties = pd.DataFrame(
            [
                {
                    "property_id": "property-1",
                    "title": "1 Unit, 3 Bedrooms, House",
                    "district": "Maseru",
                    "locality": "Lower Thetsane",
                    "bedrooms": 3,
                    "price": 650000,
                    "predicted_environment": "Urban",
                    "predicted_condition": "Good",
                    "amenities": ["parking", "yard"],
                }
            ]
        )
        clients = pd.DataFrame(
            [
                {
                    "client_id": "client-1",
                    "name": "Naleli",
                    "preferred_language": "en",
                    "preferred_channels": ["email"],
                    "free_text_preference_en": "Looking for parking and a yard.",
                    "free_text_preference_st": "",
                }
            ]
        )

        campaigns = MarketingAutomation().generate(matches, properties, clients)
        message = campaigns.iloc[0]["message"]
        first_campaign = campaigns.iloc[0]

        self.assertIn("3-Bedroom House", message)
        self.assertIn("LSL 650,000", message)
        self.assertIn("an urban area", message)
        self.assertIn("It stands out because", message)
        self.assertNotIn("a urban", message)
        self.assertEqual(first_campaign["campaign_variant"], "email_property_spotlight")
        self.assertIn("Naleli", first_campaign["subject_line"])
        self.assertIn("3-Bedroom House", first_campaign["preview_text"])
        self.assertIn("Reply to this message", first_campaign["call_to_action"])
        self.assertGreater(first_campaign["estimated_engagement_score"], first_campaign["match_score"])


if __name__ == "__main__":
    unittest.main()
