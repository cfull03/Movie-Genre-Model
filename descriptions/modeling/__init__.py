from __future__ import annotations 

# Resolve package version from installed metadata.
try:  # Why: avoid import-time failure when running from source/uninstalled.
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:
    __version__ = version("modeling") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0" 

from .model import build_model, build_pipeline, save_model, load_model
from .train import train_model
from .preprocess import build_preprocessor, load_preprocessors, save_preprocessors

__all__ = [
    "build_model",
    "build_pipeline",
    "build_preprocessor",
    "load_preprocessors",
    "save_preprocessors",
    "save_model",
    "load_model",
    "train_model",
]