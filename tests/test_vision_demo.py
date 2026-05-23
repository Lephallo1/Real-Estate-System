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

        def fake_predict(path: Path, artifact_prefix: str):
            if artifact_prefix == "residential_property_type":
                return {"cnn_property_type": {"label": "Apartment", "confidence": 0.96}}
            if artifact_prefix == "house_vision":
                return {
                    "condition": {"label": "New", "confidence": 0.92},
                    "style": {"label": "Modern", "confidence": 0.88},
                    "environment": {"label": "Hillside", "confidence": 0.81},
                    "cnn_bedroom_class": {"label": "4", "confidence": 0.79},
                }
            if artifact_prefix == "house_bedroom":
                return {"cnn_bedroom_class": {"label": "4+", "confidence": 0.91}}
            return {}

        analyzer._predict_saved_tasks = fake_predict  # type: ignore[method-assign]
        analyzer._assess_uploaded_scope = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "supported": True,
            "support_score": 0.93,
            "top_similarity": 0.94,
            "mean_top_similarity": 0.93,
            "house_similarity": 0.99,
            "scene_hint": "Garden / outdoor residential scene",
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
        analyzer._predict_saved_tasks = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "cnn_property_type": {"label": "Apartment", "confidence": 0.82}
        }
        analyzer._assess_uploaded_scope = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            "supported": False,
            "support_score": 0.31,
            "top_similarity": 0.42,
            "mean_top_similarity": 0.39,
            "house_similarity": 0.28,
            "scene_hint": "Detected sports car",
        }

        result = analyzer.analyze_uploaded_images([str(self.image_path)])

        self.assertFalse(result["supported_for_property_workflow"])
        self.assertFalse(result["allow_nlp_send"])
        self.assertEqual(result["predicted_property_type"], "Out of scope")
        self.assertEqual(result["predicted_bedrooms"], "Not available")


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
