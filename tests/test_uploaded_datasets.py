from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from pymongo.errors import PyMongoError

from database.uploaded_datasets import (
    UPLOAD_BATCHES_COLLECTION,
    UPLOADED_STUDENTS_COLLECTION,
    persist_uploaded_dataset,
    save_uploaded_training_result,
)


class UploadedDatasetPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = MagicMock()
        self.database.name = "student_analytics"
        self.students = MagicMock()
        self.batches = MagicMock()
        self.database.__getitem__.side_effect = {
            UPLOADED_STUDENTS_COLLECTION: self.students,
            UPLOAD_BATCHES_COLLECTION: self.batches,
        }.__getitem__

    @patch("database.uploaded_datasets.get_database")
    def test_persistence_uses_idempotent_upserts_and_separate_collections(
        self, get_database: MagicMock
    ) -> None:
        get_database.return_value = self.database
        dataframe = pd.DataFrame(
            {"student_id": ["S1", "S2"], "cgpa": [8.5, 7.0]}
        )

        result = persist_uploaded_dataset(dataframe, "upload-hash", "students.csv")

        self.assertEqual(result["database"], "student_analytics")
        self.assertEqual(result["collection"], UPLOADED_STUDENTS_COLLECTION)
        self.assertEqual(result["rows_saved"], 2)
        self.assertEqual(self.database.__getitem__.call_args_list[0].args[0], UPLOADED_STUDENTS_COLLECTION)
        self.students.create_index.assert_called_once_with(
            [("_upload_id", 1), ("_row_number", 1)], unique=True
        )
        operations = self.students.bulk_write.call_args.args[0]
        self.assertEqual(len(operations), 2)
        self.assertTrue(all(operation._upsert for operation in operations))
        self.assertEqual(operations[0]._filter, {"_upload_id": "upload-hash", "_row_number": 0})
        self.assertEqual(operations[0]._doc["$set"]["_source_filename"], "students.csv")
        self.batches.update_one.assert_called_once()
        manifest_filter, manifest_update = self.batches.update_one.call_args.args
        self.assertEqual(manifest_filter, {"_id": "upload-hash"})
        self.assertEqual(manifest_update["$set"]["row_count"], 2)
        self.assertEqual(manifest_update["$set"]["dataset_status"], "saved")
        self.assertTrue(self.batches.update_one.call_args.kwargs["upsert"])

    @patch("database.uploaded_datasets.get_database")
    def test_large_upload_is_written_in_bounded_batches(self, get_database: MagicMock) -> None:
        get_database.return_value = self.database
        dataframe = pd.DataFrame({"student_id": [f"S{index}" for index in range(1001)]})

        result = persist_uploaded_dataset(dataframe, "large-upload", "large.csv")

        self.assertEqual(result["rows_saved"], 1001)
        self.assertEqual(self.students.bulk_write.call_count, 3)
        self.assertEqual(
            [len(call.args[0]) for call in self.students.bulk_write.call_args_list],
            [500, 500, 1],
        )

    @patch("database.uploaded_datasets.get_database")
    def test_persistence_surfaces_mongodb_errors(self, get_database: MagicMock) -> None:
        get_database.return_value = self.database
        self.students.create_index.side_effect = PyMongoError("database unavailable")

        with self.assertRaisesRegex(RuntimeError, "Could not save the uploaded dataset"):
            persist_uploaded_dataset(pd.DataFrame({"student_id": ["S1"]}), "bad-upload", "bad.csv")

    @patch("database.uploaded_datasets.get_database")
    def test_training_result_status_is_saved_in_batch_manifest(
        self, get_database: MagicMock
    ) -> None:
        get_database.return_value = self.database
        training_result = {"best_model": "random_forest", "records": 50}

        save_uploaded_training_result("upload-hash", "trained", result=training_result)

        self.database.__getitem__.assert_called_once_with(UPLOAD_BATCHES_COLLECTION)
        update_filter, update_document = self.batches.update_one.call_args.args
        self.assertEqual(update_filter, {"_id": "upload-hash"})
        self.assertEqual(update_document["$set"]["training_status"], "trained")
        self.assertEqual(update_document["$set"]["training_result"], training_result)

    @patch("database.uploaded_datasets.get_database")
    def test_training_status_persistence_surfaces_mongodb_errors(
        self, get_database: MagicMock
    ) -> None:
        self.batches.update_one.side_effect = PyMongoError("write failed")
        get_database.return_value = self.database

        with self.assertRaisesRegex(RuntimeError, "Could not save uploaded-dataset training status"):
            save_uploaded_training_result("upload-hash", "failed", error="training error")


if __name__ == "__main__":
    unittest.main()
