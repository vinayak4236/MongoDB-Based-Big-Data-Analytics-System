from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "placement_status"
FEATURE_COLUMNS = [
    "department",
    "semester",
    "gender",
    "10th_percentage",
    "12th_percentage",
    "cgpa",
    "attendance",
    "backlogs",
    "programming_score",
    "aptitude_score",
    "communication_score",
    "projects",
    "internships",
    "certifications",
    "hackathons",
    "extracurricular_score",
    "placement_training",
]
CATEGORICAL_FEATURES = ["department", "gender"]
NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]


def prepare_training_data(dataframe: pd.DataFrame):
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing_columns)}")

    data = dataframe.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip()
    invalid_targets = sorted(set(data[TARGET_COLUMN].unique()) - {"Placed", "Not Placed"})
    if invalid_targets:
        raise ValueError(f"Unsupported placement labels: {invalid_targets}")

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    if y.nunique() < 2:
        raise ValueError("Training data must contain both Placed and Not Placed records.")
    return x, y


def build_feature_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def normalize_prediction_input(values: dict[str, Any]) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in values]
    if missing_columns:
        raise ValueError(f"Prediction input is missing required fields: {', '.join(missing_columns)}")
    row = {column: values[column] for column in FEATURE_COLUMNS}
    dataframe = pd.DataFrame([row])
    for column in NUMERIC_FEATURES:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe
