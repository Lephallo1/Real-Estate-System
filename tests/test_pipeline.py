from __future__ import annotations

import gc
import shutil
import time
import unittest
from pathlib import Path
from uuid import uuid4

from lesotho_property_ai.pipeline import run_full_pipeline


class PipelineIntegrationTests(unittest.TestCase):
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

    def test_pipeline_runs_end_to_end(self) -> None:
        case_dir = self._workspace_case_dir()
        result = run_full_pipeline(case_dir, property_count=12, client_count=6, top_n=3, seed=7)

        self.assertEqual(len(result.properties), 12)
        self.assertGreaterEqual(len(result.clients), 5)
        self.assertEqual(len(result.matches), 18)
        self.assertEqual(len(result.campaigns), len(result.clients))
        self.assertIn("predicted_property_type", result.properties.columns)
        self.assertIn("text_embedding", result.properties.columns)
        self.assertTrue(all(Path(path).exists() for paths in result.properties["image_paths"] for path in paths))

    def test_metrics_and_artifacts_exist(self) -> None:
        case_dir = self._workspace_case_dir()
        result = run_full_pipeline(case_dir, property_count=10, client_count=5, top_n=2, seed=11)

        self.assertIn("vision", result.metrics)
        self.assertIn("nlp", result.metrics)
        self.assertIn("fusion", result.metrics)
        self.assertIn("marketing", result.metrics)
        self.assertTrue(result.metrics["vision"]["property_type_accuracy"] >= 0.0)
        self.assertIn("english_query_matches_maseru", result.metrics["nlp"])
        self.assertGreaterEqual(result.metrics["marketing"]["campaigns_generated"], 1)
        for artifact_path in result.artifact_paths.values():
            self.assertTrue(Path(artifact_path).exists())


if __name__ == "__main__":
    unittest.main()
