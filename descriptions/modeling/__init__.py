"""Modeling Module.

This module provides tools for building, training, evaluating, and managing
machine learning models for movie genre classification.
"""

from __future__ import annotations

# Resolve package version from installed metadata.
try:  # Why: avoid import-time failure when running from source/uninstalled.
    from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:
    __version__ = version("modeling") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"

# Model building and management
# Evaluation utilities
from .evaluate import (
    evaluate_model,
    save_metrics,
)
from .model import (
    build_model,
    build_pipeline,
    get_model_name,
    get_params,
    load_model,
    save_model,
)

# Preprocessing utilities
from .preprocess import (
    build_preprocessor,
    load_preprocessors,
    save_preprocessors,
)

# Training utilities
from .train import (
    split_data,
    train_model,
    train_test_split_data,
)

__all__ = [
    # Model building and management
    "build_model",
    "build_pipeline",
    "get_model_name",
    "get_params",
    "load_model",
    "save_model",
    # Preprocessing
    "build_preprocessor",
    "load_preprocessors",
    "save_preprocessors",
    # Training
    "split_data",
    "train_model",
    "train_test_split_data",
    # Evaluation
    "evaluate_model",
    "save_metrics",
]
