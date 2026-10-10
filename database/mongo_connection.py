from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

from config.config import MONGODB_DB_NAME, MONGODB_URI


def get_database(uri: str | None = None, db_name: str | None = None):
    client = MongoClient(uri or MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[db_name or MONGODB_DB_NAME]
    return db


def ping_mongodb(uri: str | None = None) -> bool:
    client = MongoClient(uri or MONGODB_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        return True
    except (ConnectionFailure, PyMongoError):
        return False
