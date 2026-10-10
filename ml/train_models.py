from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config.config import DEFAULT_CSV_PATH, MODEL_DIR
from ml.evaluate_models import CLASS_LABELS, POSITIVE_CLASS, evaluate_model, select_best_model
from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_preprocessor, prepare_training_data


def _model_pipelines(random_state: int) -> dict[str, Pipeline]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_feature_preprocessor()), ("classifier", model)])
        for name, model in models.items()
    }


def train_and_save_models(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    model_dir: str | Path = MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_file}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    dataframe = pd.read_csv(csv_file)
    x, y = prepare_training_data(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipelines = _model_pipelines(random_state)
    metrics = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        metrics[name] = evaluate_model(pipeline, x_test, y_test)

    best_model_name = select_best_model(metrics)
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "placement_model.joblib"
    metadata_path = destination / "training_metadata.json"
    joblib.dump(pipelines[best_model_name], model_path)

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "records": int(len(dataframe)),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "class_distribution": {str(key): int(value) for key, value in y.value_counts().items()},
        "class_labels": CLASS_LABELS,
        "positive_class": POSITIVE_CLASS,
        "models": metrics,
        "best_model": best_model_name,
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate placement prediction models.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Training CSV path.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Directory for the selected model and metadata.")
    args = parser.parse_args()
    result = train_and_save_models(args.csv, args.model_dir)
    print(json.dumps(result, indent=2))
