from __future__ import annotations 

# Resolve package version from installed metadata.
try:  # Why: avoid import-time failure when running from source/uninstalled.
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except Exception:  # pragma: no cover
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

try:
    __version__ = version("descriptions") if version else "0.0.0"
except PackageNotFoundError:
    __version__ = "0.0.0" 

from .config import (
    DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    MODELS_DIR,
    INTERIM_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR
)

from .dataset import (
    load_data,
    load_interim,
    load_model,
    load_processed,
    save_model,
    to_interim,
    to_processed
)

from .features import (
    generate_descriptions,
    generate_targets
)
