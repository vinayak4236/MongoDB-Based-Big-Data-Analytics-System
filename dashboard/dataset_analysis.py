from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = {
    "student_id",
    "department",
    "placement_status",
    "cgpa",
    "attendance",
}

ALIASES = {
    "student_id": ["student_id", "student id", "id", "studentid"],
    "department": ["department", "dept", "branch", "stream"],
    "placement_status": ["placement_status", "placement status", "placed", "placement", "outcome"],
    "cgpa": ["cgpa", "gpa", "grade point average"],
    "attendance": ["attendance", "attendance percentage", "attendance_percent"],
}


def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("The uploaded CSV file is empty.")
    try:
        dataframe = pd.read_csv(BytesIO(file_bytes))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse the uploaded CSV: {exc}") from exc
    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    return dataframe


def dataset_profile(dataframe: pd.DataFrame, file_size_bytes: int) -> dict[str, Any]:
    return {
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_cells": int(dataframe.isna().sum().sum()),
        "missing_by_column": dataframe.isna().sum().loc[lambda values: values > 0].to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
    }


def identify_columns(dataframe: pd.DataFrame) -> dict[str, str | None]:
    normalized = {str(column).strip().lower().replace("-", " ").replace("_", " "): column for column in dataframe.columns}
    identified = {}
    for field, aliases in ALIASES.items():
        identified[field] = next((normalized[alias] for alias in aliases if alias in normalized), None)
    return identified


def validate_uploaded_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    identified = identify_columns(dataframe)
    missing_expected = [field for field in EXPECTED_COLUMNS if identified[field] is None]
    try:
        from ml.feature_engineering import FEATURE_COLUMNS
    except ImportError as exc:
        ml_missing_columns = []
        ml_dependency_error = str(exc)
        ml_compatible = False
    else:
        ml_missing_columns = [column for column in FEATURE_COLUMNS + ["placement_status"] if column not in dataframe.columns]
        ml_dependency_error = None
        ml_compatible = not ml_missing_columns
    return {
        "valid_for_analysis": len(dataframe) > 0,
        "identified_columns": identified,
        "missing_expected_columns": missing_expected,
        "ml_compatible": ml_compatible,
        "ml_missing_columns": ml_missing_columns,
        "ml_dependency_error": ml_dependency_error,
    }


def dynamic_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include="number").columns.tolist()


def train_uploaded_models(dataframe: pd.DataFrame) -> dict[str, Any]:
    from ml.evaluate_models import evaluate_model, select_best_model
    from ml.feature_engineering import prepare_training_data
    from ml.train_models import _model_pipelines

    x, y = prepare_training_data(dataframe)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    metrics = {}
    pipelines = _model_pipelines(42)
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        metrics[name] = evaluate_model(pipeline, x_test, y_test)
    return {"models": metrics, "best_model": select_best_model(metrics), "records": len(dataframe)}
