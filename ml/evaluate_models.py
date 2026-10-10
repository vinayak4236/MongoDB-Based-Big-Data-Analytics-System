from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_CLASS = "Placed"
CLASS_LABELS = ["Not Placed", POSITIVE_CLASS]


def evaluate_model(model, x_test, y_test) -> dict[str, Any]:
    predictions = model.predict(x_test)
    class_labels = list(model.classes_)
    if class_labels != CLASS_LABELS:
        raise ValueError(f"Unexpected model class order: {class_labels}; expected {CLASS_LABELS}")
    probabilities = model.predict_proba(x_test)[:, class_labels.index(POSITIVE_CLASS)]
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)

    return {
        "class_labels": class_labels,
        "positive_class": POSITIVE_CLASS,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, pos_label=POSITIVE_CLASS, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score((y_test == POSITIVE_CLASS).astype(int), probabilities)), 4),
        "confusion_matrix": matrix.tolist(),
        "test_records": int(len(y_test)),
    }


def select_best_model(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        raise ValueError("Cannot select a model from an empty metrics collection.")
    return max(metrics, key=lambda name: (metrics[name]["f1"], metrics[name]["roc_auc"], metrics[name]["accuracy"]))
