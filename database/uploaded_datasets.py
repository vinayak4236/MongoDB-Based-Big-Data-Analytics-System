from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from database.mongo_connection import get_database

UPLOADED_STUDENTS_COLLECTION = "uploaded_students"
UPLOAD_BATCHES_COLLECTION = "uploaded_dataset_batches"


def persist_uploaded_dataset(
    dataframe: pd.DataFrame,
    upload_id: str,
    filename: str,
) -> dict[str, Any]:
    database = get_database()
    students = database[UPLOADED_STUDENTS_COLLECTION]
    batches = database[UPLOAD_BATCHES_COLLECTION]
    try:
        students.create_index([("_upload_id", 1), ("_row_number", 1)], unique=True)
        for start in range(0, len(dataframe), 500):
            records = json.loads(dataframe.iloc[start : start + 500].to_json(orient="records", date_format="iso"))
            operations = []
            for offset, record in enumerate(records):
                record.pop("_id", None)
                record.update(
                    {
                        "_upload_id": upload_id,
                        "_row_number": start + offset,
                        "_source_filename": filename,
                    }
                )
                operations.append(
                    UpdateOne(
                        {"_upload_id": upload_id, "_row_number": start + offset},
                        {"$set": record},
                        upsert=True,
                    )
                )
            students.bulk_write(operations, ordered=False)

        batches.update_one(
            {"_id": upload_id},
            {
                "$set": {
                    "source_filename": filename,
                    "row_count": len(dataframe),
                    "columns": [str(column) for column in dataframe.columns],
                    "dataset_status": "saved",
                },
                "$setOnInsert": {"uploaded_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
    except PyMongoError as exc:
        raise RuntimeError(f"Could not save the uploaded dataset to MongoDB: {exc}") from exc

    return {
        "database": database.name,
        "collection": UPLOADED_STUDENTS_COLLECTION,
        "upload_id": upload_id,
        "rows_saved": len(dataframe),
    }


def save_uploaded_training_result(
    upload_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    update: dict[str, Any] = {"training_status": status}
    if result is not None:
        update["training_result"] = result
    if error is not None:
        update["training_error"] = error
    try:
        get_database()[UPLOAD_BATCHES_COLLECTION].update_one(
            {"_id": upload_id},
            {"$set": update},
            upsert=True,
        )
    except PyMongoError as exc:
        raise RuntimeError(f"Could not save uploaded-dataset training status to MongoDB: {exc}") from exc
