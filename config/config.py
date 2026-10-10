import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable '{name}' is not set.")
    return value


MONGODB_URI = get_env("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = get_env("MONGODB_DB_NAME", "student_analytics")
MONGODB_COLLECTION = get_env("MONGODB_COLLECTION", "students")
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "students.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "trained_model"
