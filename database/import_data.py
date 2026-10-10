from pathlib import Path

import pandas as pd
from pymongo.errors import PyMongoError

from config.config import MONGODB_COLLECTION, MONGODB_DB_NAME, MONGODB_URI
from database.mongo_connection import get_database


def import_csv_to_mongodb(
    csv_path: str | Path,
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    drop_existing: bool = True,
):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_file}")

    records = df.to_dict(orient="records")
    database = get_database(uri=uri, db_name=db_name)
    collection = database[collection_name or MONGODB_COLLECTION]

    if drop_existing:
        collection.delete_many({})

    try:
        result = collection.insert_many(records)
        return {
            "database": db_name or MONGODB_DB_NAME,
            "collection": collection_name or MONGODB_COLLECTION,
            "rows_inserted": len(result.inserted_ids),
            "source_file": str(csv_file),
        }
    except PyMongoError as exc:
        raise RuntimeError(f"Failed to insert records into MongoDB: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import CSV data into MongoDB.")
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--uri", default=MONGODB_URI, help="MongoDB connection URI.")
    parser.add_argument("--db", default=MONGODB_DB_NAME, help="MongoDB database name.")
    parser.add_argument(
        "--collection",
        default=MONGODB_COLLECTION,
        help="MongoDB collection name.",
    )
    args = parser.parse_args()

    result = import_csv_to_mongodb(args.csv, uri=args.uri, db_name=args.db, collection_name=args.collection)
    print(result)
