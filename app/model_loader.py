"""Model loading and caching for Flask API."""

from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from descriptions.config import MODELS_DIR
from descriptions.modeling.model import load_model
from descriptions.modeling.preprocess import load_preprocessors


class ModelLoader:
    """Singleton for loading and caching models."""

    _instance = None
    _model = None
    _vectorizer = None
    _mlb = None
    _feature_selector = None
    _model_path = None
    _loaded = False

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """
        Load model and preprocessors.

        Args:
            model_path: Path to model file. If None, auto-detects.
        """
        if self._loaded and model_path == self._model_path:
            logger.debug("Model already loaded, skipping...")
            return

        logger.info("Loading model and preprocessors...")

        # Auto-detect model if not provided
        if model_path is None:
            model_files = list(MODELS_DIR.glob("*.joblib"))
            model_files = [
                f
                for f in model_files
                if f.name
                not in {
                    "tfidf_vectorizer.joblib",
                    "genre_binarizer.joblib",
                    "feature_selector.joblib",
                }
            ]
            if not model_files:
                raise FileNotFoundError(
                    f"No model found in {MODELS_DIR}. Please train a model first."
                )
            model_path = model_files[0]
            logger.info(f"Auto-detected model: {model_path.name}")

        # Load model
        self._model = load_model(model_path)
        self._model_path = model_path
        logger.success(f"✓ Model loaded: {model_path.name}")

        # Load preprocessors
        self._vectorizer, self._mlb, self._feature_selector = load_preprocessors()
        logger.success("✓ Preprocessors loaded")

        self._loaded = True

    def get_model(self):
        """Get the loaded model."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    def get_preprocessors(self) -> Tuple:
        """Get the loaded preprocessors."""
        if not self._loaded:
            raise RuntimeError("Preprocessors not loaded. Call load_model() first.")
        return self._vectorizer, self._mlb, self._feature_selector

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_info(self) -> dict:
        """Get model metadata."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": str(self._model_path) if self._model_path else None,
            "model_name": self._model_path.name if self._model_path else None,
            "num_genres": len(self._mlb.classes_) if self._mlb else None,
            "genres": list(self._mlb.classes_) if self._mlb else None,
        }


# Global instance
model_loader = ModelLoader()
