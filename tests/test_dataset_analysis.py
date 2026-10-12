from __future__ import annotations

import unittest

import pandas as pd

from dashboard.dataset_analysis import (
    dataset_profile,
    dynamic_numeric_columns,
    identify_columns,
    load_uploaded_csv,
    validate_uploaded_dataset,
)
from ml.feature_engineering import FEATURE_COLUMNS


def compatible_dataset(rows: int = 40) -> pd.DataFrame:
    data = {
        column: [index + 1 for index in range(rows)]
        for column in FEATURE_COLUMNS
        if column not in {"department", "gender"}
    }
    data["department"] = ["CSE" if index % 2 else "ECE" for index in range(rows)]
    data["gender"] = ["Female" if index % 2 else "Male" for index in range(rows)]
    data["placement_status"] = ["Placed" if index % 2 else "Not Placed" for index in range(rows)]
    data["student_id"] = [f"STU{index:05d}" for index in range(rows)]
    return pd.DataFrame(data)


class UploadedDatasetTests(unittest.TestCase):
    def test_load_uploaded_csv_preserves_rows_and_columns(self) -> None:
        source = b"student_id,cgpa\nS1,8.5\nS2,7.0\n"

        result = load_uploaded_csv(source)

        self.assertEqual(result["student_id"].tolist(), ["S1", "S2"])
        self.assertEqual(result["cgpa"].tolist(), [8.5, 7.0])

    def test_load_uploaded_csv_rejects_empty_and_invalid_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            load_uploaded_csv(b"")
        with self.assertRaisesRegex(ValueError, "parse"):
            load_uploaded_csv(b"\x80\x81")

    def test_load_uploaded_csv_rejects_header_without_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "no data rows"):
            load_uploaded_csv(b"student_id,cgpa\n")

    def test_dataset_profile_counts_missing_values_and_duplicates(self) -> None:
        dataframe = pd.DataFrame(
            {"student_id": ["S1", "S1", "S3"], "cgpa": [8.0, 8.0, None]}
        )

        profile = dataset_profile(dataframe, 2048)

        self.assertEqual(profile["file_size_kb"], 2.0)
        self.assertEqual(profile["rows"], 3)
        self.assertEqual(profile["columns"], 2)
        self.assertEqual(profile["duplicate_rows"], 1)
        self.assertEqual(profile["missing_cells"], 1)
        self.assertEqual(profile["missing_by_column"], {"cgpa": 1})

    def test_column_identification_accepts_common_aliases(self) -> None:
        dataframe = pd.DataFrame(
            columns=["Student ID", "Branch", "Placement Status", "GPA", "Attendance Percentage"]
        )

        identified = identify_columns(dataframe)

        self.assertEqual(
            identified,
            {
                "student_id": "Student ID",
                "department": "Branch",
                "placement_status": "Placement Status",
                "cgpa": "GPA",
                "attendance": "Attendance Percentage",
            },
        )

    def test_validation_recognizes_complete_ml_schema(self) -> None:
        validation = validate_uploaded_dataset(compatible_dataset())

        self.assertTrue(validation["valid_for_analysis"])
        self.assertEqual(validation["missing_expected_columns"], [])
        self.assertTrue(validation["ml_compatible"])
        self.assertEqual(validation["ml_missing_columns"], [])
        self.assertIsNone(validation["ml_dependency_error"])

    def test_validation_reports_missing_ml_features_without_rejecting_analytics(self) -> None:
        dataframe = pd.DataFrame(
            {
                "student_id": ["S1"],
                "department": ["CSE"],
                "placement_status": ["Placed"],
                "cgpa": [8.0],
                "attendance": [90],
            }
        )

        validation = validate_uploaded_dataset(dataframe)

        self.assertTrue(validation["valid_for_analysis"])
        self.assertFalse(validation["ml_compatible"])
        self.assertIn("semester", validation["ml_missing_columns"])
        self.assertIsNone(validation["ml_dependency_error"])

    def test_dynamic_numeric_columns_returns_only_numeric_fields(self) -> None:
        dataframe = pd.DataFrame({"student_id": ["S1"], "cgpa": [8.0], "attendance": [90]})

        self.assertEqual(dynamic_numeric_columns(dataframe), ["cgpa", "attendance"])


if __name__ == "__main__":
    unittest.main()
