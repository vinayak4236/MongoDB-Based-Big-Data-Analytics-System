from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dashboard.dataset_analysis import train_uploaded_models
from ml.evaluate_models import CLASS_LABELS, evaluate_model, select_best_model
from ml.feature_engineering import FEATURE_COLUMNS, prepare_training_data
from tests.test_dataset_analysis import compatible_dataset


class TrainingDataTests(unittest.TestCase):
    def test_prepare_training_data_selects_features_and_does_not_mutate_input(self) -> None:
        dataframe = compatible_dataset()
        original_columns = dataframe.columns.tolist()

        features, target = prepare_training_data(dataframe)

        self.assertEqual(features.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(len(features), len(dataframe))
        self.assertEqual(set(target.unique()), set(CLASS_LABELS))
        self.assertEqual(dataframe.columns.tolist(), original_columns)

    def test_prepare_training_data_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "department"):
            prepare_training_data(pd.DataFrame({"placement_status": ["Placed"]}))

    def test_prepare_training_data_rejects_unknown_target_values(self) -> None:
        dataframe = compatible_dataset()
        dataframe.loc[0, "placement_status"] = "Pending"

        with self.assertRaisesRegex(ValueError, "Unsupported placement labels"):
            prepare_training_data(dataframe)

    def test_prepare_training_data_requires_both_target_classes(self) -> None:
        dataframe = compatible_dataset()
        dataframe["placement_status"] = "Placed"

        with self.assertRaisesRegex(ValueError, "both Placed and Not Placed"):
            prepare_training_data(dataframe)

    def test_evaluate_model_returns_expected_metrics_and_matrix_shape(self) -> None:
        class FixedModel:
            classes_ = CLASS_LABELS

            def predict(self, _features):
                return ["Not Placed", "Placed", "Placed", "Not Placed"]

            def predict_proba(self, _features):
                return np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]])

        result = evaluate_model(
            FixedModel(),
            pd.DataFrame({"feature": [1, 2, 3, 4]}),
            pd.Series(["Not Placed", "Placed", "Placed", "Not Placed"]),
        )

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(result["test_records"], 4)

    def test_evaluate_model_rejects_unexpected_class_order(self) -> None:
        class ReversedModel:
            classes_ = list(reversed(CLASS_LABELS))

            def predict(self, _features):
                return CLASS_LABELS

            def predict_proba(self, _features):
                return np.array([[0.9, 0.1], [0.1, 0.9]])

        with self.assertRaisesRegex(ValueError, "Unexpected model class order"):
            evaluate_model(
                ReversedModel(),
                pd.DataFrame({"feature": [1, 2]}),
                pd.Series(CLASS_LABELS),
            )

    def test_select_best_model_uses_f1_then_auc_then_accuracy(self) -> None:
        metrics = {
            "lower_f1": {"f1": 0.8, "roc_auc": 0.99, "accuracy": 0.99},
            "higher_auc": {"f1": 0.9, "roc_auc": 0.7, "accuracy": 0.7},
            "higher_accuracy": {"f1": 0.9, "roc_auc": 0.7, "accuracy": 0.8},
        }

        self.assertEqual(select_best_model(metrics), "higher_accuracy")
        with self.assertRaisesRegex(ValueError, "empty metrics"):
            select_best_model({})

    def test_uploaded_model_training_saves_selected_model_and_metadata(self) -> None:
        dataframe = compatible_dataset(rows=80)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = train_uploaded_models(dataframe, temporary_directory)
            model_path = Path(result["model_path"])
            metadata_path = Path(temporary_directory) / "training_metadata.json"

            self.assertTrue(model_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertIsNotNone(joblib.load(model_path))
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_metadata["records"], 80)
            self.assertEqual(saved_metadata["best_model"], result["best_model"])
            self.assertEqual(
                set(result["models"]),
                {"logistic_regression", "decision_tree", "random_forest"},
            )
            for model_result in result["models"].values():
                self.assertEqual(model_result["test_records"], 16)

    def test_uploaded_model_training_rejects_too_few_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "at least 10 records"):
                train_uploaded_models(compatible_dataset(rows=8), temporary_directory)


if __name__ == "__main__":
    unittest.main()
