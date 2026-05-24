from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lesotho_property_ai.vision.analyzer import PropertyVisionAnalyzer
from lesotho_property_ai.web.admin import _presentation_property_type_rows


class UploadedVisionDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.image_path = self.temp_dir / "sample.png"
        Image.new("RGB", (64, 64), color=(180, 180, 180)).save(self.image_path)

    def test_house_similarity_can_override_weak_residential_type_label(self) -> None:
        analyzer = PropertyVisionAnalyzer()
        analyzer._analyze_image_with_llm = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

        analyzer._analyze_uploaded_scope_fast = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "supported": True,
            "support_score": 0.93,
            "top_similarity": 0.94,
            "mean_top_similarity": 0.93,
            "house_similarity": 0.99,
            "scene_hint": "Garden / outdoor residential scene",
            "scene_description": "The uploaded image resembles a house scene from the reviewed training examples.",
            "property_type": "Apartment",
            "property_type_confidence": 0.96,
            "condition": "New",
            "style": "Modern",
            "environment": "Hillside",
            "cnn_bedroom_class": "4+",
            "locality": "Maseru West",
        }

        result = analyzer.analyze_uploaded_images([str(self.image_path)])

        self.assertTrue(result["supported_for_property_workflow"])
        self.assertTrue(result["allow_nlp_send"])
        self.assertEqual(result["predicted_property_type"], "House")
        self.assertEqual(result["predicted_condition"], "New")
        self.assertEqual(result["predicted_style"], "Modern")
        self.assertEqual(result["predicted_environment"], "Hillside")
        self.assertEqual(result["predicted_bedrooms"], 4)

    def test_out_of_scope_upload_does_not_invent_house_attributes(self) -> None:
        analyzer = PropertyVisionAnalyzer()
        analyzer._analyze_image_with_llm = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
        analyzer._analyze_uploaded_scope_fast = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "supported": False,
            "support_score": 0.31,
            "top_similarity": 0.42,
            "mean_top_similarity": 0.39,
            "house_similarity": 0.28,
            "scene_hint": "Detected sports car",
            "scene_description": "The uploaded image appears to show a vehicle rather than a residential property.",
            "property_type": "Out of scope",
            "property_type_confidence": 0.82,
            "condition": "",
            "style": "",
            "environment": "",
            "cnn_bedroom_class": "",
            "locality": "",
        }

        result = analyzer.analyze_uploaded_images([str(self.image_path)])

        self.assertFalse(result["supported_for_property_workflow"])
        self.assertFalse(result["allow_nlp_send"])
        self.assertEqual(result["predicted_property_type"], "Out of scope")
        self.assertEqual(result["predicted_bedrooms"], "Not available")

    def test_llm_house_result_is_used_when_available(self) -> None:
        analyzer = PropertyVisionAnalyzer()
        analyzer._analyze_image_with_llm = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "is_property": True,
            "property_type": "House",
            "condition": "Good",
            "style": "Modern",
            "environment": "Suburban",
            "scene_description": "A modern single-storey house with a paved yard is visible.",
            "confidence": 0.91,
            "analysis_source": "gemini",
            "analysis_source_label": "Gemini",
        }

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("Fallback analyzer should not run when the hosted LLM returns a valid result.")

        analyzer._analyze_uploaded_scope_fast = should_not_run  # type: ignore[method-assign]

        result = analyzer.analyze_uploaded_images([str(self.image_path)])

        self.assertTrue(result["supported_for_property_workflow"])
        self.assertTrue(result["allow_nlp_send"])
        self.assertEqual(result["predicted_property_type"], "House")
        self.assertEqual(result["predicted_condition"], "Good")
        self.assertEqual(result["predicted_style"], "Modern")
        self.assertEqual(result["predicted_environment"], "Suburban")
        self.assertEqual(result["scene_description"], "A modern single-storey house with a paved yard is visible.")

    def test_llm_non_property_result_is_used_when_available(self) -> None:
        analyzer = PropertyVisionAnalyzer()
        analyzer._analyze_image_with_llm = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "is_property": False,
            "property_type": "Not a property",
            "condition": "N/A",
            "style": "N/A",
            "environment": "N/A",
            "scene_description": "The image shows a silver hatchback parked indoors.",
            "confidence": 0.94,
            "analysis_source": "gemini",
            "analysis_source_label": "Gemini",
        }

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("Fallback analyzer should not run when the hosted LLM returns a valid result.")

        analyzer._analyze_uploaded_scope_fast = should_not_run  # type: ignore[method-assign]

        result = analyzer.analyze_uploaded_images([str(self.image_path)])

        self.assertFalse(result["supported_for_property_workflow"])
        self.assertFalse(result["allow_nlp_send"])
        self.assertEqual(result["predicted_property_type"], "Not a property")
        self.assertEqual(result["scene_description"], "The image shows a silver hatchback parked indoors.")


class AdminPresentationFilterTests(unittest.TestCase):
    def test_site_rows_are_hidden_from_presentation_summary(self) -> None:
        rows = _presentation_property_type_rows(
            {
                "House": 197,
                "Site": 89,
                "Townhouse": 23,
            }
        )

        labels = [row["label"] for row in rows]
        self.assertIn("House", labels)
        self.assertIn("Townhouse", labels)
        self.assertNotIn("Site", labels)


if __name__ == "__main__":
    unittest.main()
