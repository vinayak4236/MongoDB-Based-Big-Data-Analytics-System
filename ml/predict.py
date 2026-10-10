from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from config.config import MODEL_DIR
from ml.evaluate_models import CLASS_LABELS, POSITIVE_CLASS
from ml.feature_engineering import normalize_prediction_input


def predict_placement(values: dict[str, Any], model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path) if model_path else MODEL_DIR / "placement_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Run Phase 3 training first.")

    model = joblib.load(path)
    class_labels = list(model.classes_)
    if class_labels != CLASS_LABELS:
        raise ValueError(f"Unexpected model class order: {class_labels}; expected {CLASS_LABELS}")
    input_data = normalize_prediction_input(values)
    prediction = str(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    placed_index = class_labels.index(POSITIVE_CLASS)
    return {
        "predicted_status": prediction,
        "class_labels": class_labels,
        "placement_probability": round(float(probabilities[placed_index]), 4),
        "placement_probability_percent": round(float(probabilities[placed_index]) * 100, 2),
    }
