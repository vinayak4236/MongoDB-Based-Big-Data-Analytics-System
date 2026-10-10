from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FEATURE_COLUMNS": (".feature_engineering", "FEATURE_COLUMNS"),
    "TARGET_COLUMN": (".feature_engineering", "TARGET_COLUMN"),
    "build_feature_preprocessor": (".feature_engineering", "build_feature_preprocessor"),
    "prepare_training_data": (".feature_engineering", "prepare_training_data"),
    "predict_placement": (".predict", "predict_placement"),
    "train_and_save_models": (".train_models", "train_and_save_models"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
