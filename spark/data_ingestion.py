from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from config.config import MONGODB_COLLECTION, MONGODB_DB_NAME, MONGODB_URI
from spark.spark_session import get_spark_session


def load_student_records_from_mongodb(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
):
    mongo_uri = uri or MONGODB_URI
    database_name = db_name or MONGODB_DB_NAME
    collection = collection_name or MONGODB_COLLECTION

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        documents = list(client[database_name][collection].find({}, {"_id": 0}))
        if not documents:
            raise ValueError(f"No documents found in MongoDB collection '{collection}' in database '{database_name}'.")
        return documents
    except PyMongoError as exc:
        raise RuntimeError(f"Unable to fetch document records from MongoDB: {exc}") from exc
    finally:
        client.close()


def load_student_dataframe_from_mongodb(
    uri: str | None = None,
    db_name: str | None = None,
    collection_name: str | None = None,
    spark=None,
):
    if spark is None:
        spark = get_spark_session()

    documents = load_student_records_from_mongodb(uri=uri, db_name=db_name, collection_name=collection_name)
    pandas_df = pd.DataFrame(documents)
    return spark.createDataFrame(pandas_df)


def load_student_dataframe_from_csv(csv_path: str | Path, spark=None):
    if spark is None:
        spark = get_spark_session()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(csv_file))
    )
