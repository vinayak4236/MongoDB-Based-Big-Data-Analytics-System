from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from config.config import DEFAULT_CSV_PATH
from tests.test_dataset_analysis import compatible_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DashboardRenderingTests(unittest.TestCase):
    def test_all_pages_render_with_project_dataset(self) -> None:
        pages = [
            "Overview",
            "Performance Analytics",
            "Placement Analytics",
            "Correlation Patterns",
            "Placement Prediction",
            "Model Performance",
            "Data View",
            "Dataset Upload",
        ]
        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py")).run()
        self.assertEqual(app.exception, [])

        for page in pages:
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertEqual(app.exception, [])
                self.assertTrue(app.title)

    def test_all_analytics_pages_render_uploaded_dataset(self) -> None:
        pages = [
            "Overview",
            "Performance Analytics",
            "Placement Analytics",
            "Correlation Patterns",
            "Placement Prediction",
            "Model Performance",
            "Data View",
        ]
        dataframe = pd.read_csv(DEFAULT_CSV_PATH).head(100)
        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py")).run()
        app.session_state["uploaded_dataframe"] = dataframe
        app.session_state["uploaded_name"] = "test-upload.csv"
        app.session_state["dataset_mode"] = "Upload New Dataset"

        for page in pages:
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertEqual(app.exception, [])
                self.assertEqual(app.title[-1].value, {
                    "Overview": "Overview",
                    "Performance Analytics": "Student Performance",
                    "Placement Analytics": "Placement Analytics",
                    "Correlation Patterns": "Correlation & Patterns",
                    "Placement Prediction": "Placement Prediction",
                    "Model Performance": "Model Performance",
                    "Data View": "Data Explorer",
                }[page])

        app.radio[0].set_value("Overview").run()
        total_students = next(metric.value for metric in app.metric if metric.label == "Total Students")
        self.assertEqual(total_students, "100")

    def test_dataset_upload_page_compares_uploaded_data_to_project_data(self) -> None:
        project_data = pd.read_csv(DEFAULT_CSV_PATH)
        uploaded_data = project_data.head(100).copy()
        uploaded_data["cgpa"] = uploaded_data["cgpa"] + 0.25

        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py")).run()
        app.session_state["uploaded_dataframe"] = uploaded_data
        app.session_state["uploaded_name"] = "comparison-upload.csv"
        app.radio[0].set_value("Dataset Upload").run()

        self.assertEqual(app.exception, [])
        self.assertIn("Project vs uploaded dataset", [item.value for item in app.subheader])
        feature_selector = next(item for item in app.selectbox if item.label == "Shared numeric feature")
        self.assertIn("Cgpa", feature_selector.options)

    def test_upload_persists_dataset_and_trains_compatible_data(self) -> None:
        dataframe = compatible_dataset()
        csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
        upload_id = hashlib.sha256(csv_bytes).hexdigest()
        database_result = {
            "database": "student_analytics",
            "collection": "uploaded_students",
            "upload_id": upload_id,
            "rows_saved": len(dataframe),
        }
        training_result = {
            "best_model": "random_forest",
            "model_path": "temporary-model.joblib",
            "models": {
                name: {
                    "accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "roc_auc": 0.8,
                }
                for name in ("logistic_regression", "decision_tree", "random_forest")
            },
        }
        app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py")).run()
        app.radio[0].set_value("Dataset Upload").run()

        with (
            patch("database.uploaded_datasets.persist_uploaded_dataset", return_value=database_result) as save_dataset,
            patch("database.uploaded_datasets.save_uploaded_training_result") as save_training_result,
            patch("dashboard.dataset_analysis.train_uploaded_models", return_value=training_result) as train_models,
        ):
            app.file_uploader[0].upload("upload.csv", csv_bytes, "text/csv").run()

        self.assertEqual(app.exception, [])
        save_dataset.assert_called_once()
        self.assertEqual(save_dataset.call_args.args[1:], (upload_id, "upload.csv"))
        self.assertEqual(len(save_dataset.call_args.args[0]), len(dataframe))
        train_models.assert_called_once()
        self.assertEqual(len(train_models.call_args.args[0]), len(dataframe))
        save_training_result.assert_called_once_with(
            upload_id, "trained", result=training_result
        )
        self.assertTrue(
            any("Automatic training complete" in item.value for item in app.success)
        )


if __name__ == "__main__":
    unittest.main()
