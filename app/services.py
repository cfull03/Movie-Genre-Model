"""Service layer for prediction logic."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from descriptions.config import MODELS_DIR
from descriptions.modeling.model import load_model
from descriptions.modeling.preprocess import load_preprocessors


class PredictionService:
    """Service for handling genre predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.model = None
        self.model_path = None
        self.vectorizer = None
        self.mlb = None
        self.normalizer = None
        self.feature_selector = None
        self._is_loaded = False
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model and preprocessors.
        
        Args:
            model_path: Path to model file. If None, uses default.
        """
        if self._is_loaded and model_path == self.model_path:
            logger.debug("Model already loaded, skipping reload")
            return
        
        try:
            # Load model
            if model_path is None:
                logger.debug("No model path provided, searching for default model...")
                model_files = list(MODELS_DIR.glob("*.joblib"))
                model_files = [
                    f
                    for f in model_files
                    if f.name not in {"tfidf_vectorizer.joblib", "genre_binarizer.joblib", "normalizer.joblib", "feature_selector.joblib"}
                ]
                if not model_files:
                    raise FileNotFoundError(f"No model found in {MODELS_DIR}. Please train a model first.")
                model_path = model_files[0]
                logger.info(f"Using default model: {model_path.name}")
            else:
                model_path = Path(model_path)
                if not model_path.is_absolute():
                    model_path = MODELS_DIR / model_path
                logger.info(f"Loading model from {model_path}...")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = load_model(model_path)
            self.model_path = str(model_path)
            logger.success(f"✓ Model loaded successfully: {model_path.name}")
            
            # Load preprocessors
            logger.info("Loading preprocessors...")
            self.vectorizer, self.mlb, self.normalizer, self.feature_selector = load_preprocessors()
            logger.success("✓ Preprocessors loaded successfully")
            
            self._is_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._is_loaded = False
            raise
    
    def predict(
        self,
        descriptions: List[str],
        threshold: float = 0.55,
        top_k: int = 3,
        model_path: Optional[str] = None,
    ) -> List[List[str]]:
        """
        Predict genres for descriptions.
        
        Args:
            descriptions: List of movie descriptions
            threshold: Probability threshold for predictions. Only genres above
                      this threshold will be included.
            top_k: Maximum number of top genres to select (default: 3).
                   The top k genres by probability will be selected, but only
                   those above the threshold will be returned.
            model_path: Optional model path (will reload if different)
        
        Returns:
            List of lists of predicted genres
        """
        from scipy.special import expit
        
        # Load model if not loaded or if different model requested
        if not self._is_loaded or (model_path and model_path != self.model_path):
            self.load_model(model_path)
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Transform descriptions to TF-IDF features
        X = self.vectorizer.transform(descriptions)
        
        # Apply L2 normalization
        X = self.normalizer.transform(X)
        
        # Apply feature selection
        X = self.feature_selector.transform(X)
        
        # Convert to dense array for LinearSVC (handle both sparse and dense)
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Get prediction scores
        y_scores = self.model.decision_function(X_dense)
        
        # Convert scores to probabilities using sigmoid function
        y_proba = expit(y_scores)
        
        # Select top-k genres per sample, but only include those above threshold
        y_pred_binary = np.zeros_like(y_proba, dtype=int)
        
        for i in range(y_proba.shape[0]):
            # Get top-k indices for this sample (sorted by probability descending)
            top_k_indices = np.argsort(y_proba[i])[-top_k:][::-1]
            
            # Only include genres that are above threshold
            for idx in top_k_indices:
                if y_proba[i, idx] >= threshold:
                    y_pred_binary[i, idx] = 1
        
        # Decode predictions back to genre labels
        predicted_genres = self.mlb.inverse_transform(y_pred_binary)
        
        return predicted_genres
    
    def is_ready(self) -> bool:
        """Check if service is ready (model loaded)."""
        return self._is_loaded


# Global service instance
prediction_service = PredictionService()
