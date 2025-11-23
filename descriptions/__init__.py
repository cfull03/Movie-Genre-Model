"""Movie Genre Classification Package.

This package provides tools for predicting movie genres based on their descriptions
using NLP and machine learning techniques.
"""

from __future__ import annotations

# Resolve package version from installed metadata.
try:  # Why: avoid import-time failure when running from source/uninstalled.
    from importlib.metadata import PackageNotFoundError, version  # Python 3.8+
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:
    __version__ = version("descriptions") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0"

# Configuration paths
from .config import (
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
)

# Dataset utilities
from .dataset import (
    load_data,
    load_interim,
    load_processed,
    to_interim,
    to_processed,
)

__all__ = [
    # Configuration
    "DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "FIGURES_DIR",
    "INTERIM_DATA_DIR",
    "MODELS_DIR",
    "PROCESSED_DATA_DIR",
    "PROJ_ROOT",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    # Dataset utilities
    "load_data",
    "load_interim",
    "load_processed",
    "to_interim",
    "to_processed",
]
