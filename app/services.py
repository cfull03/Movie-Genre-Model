"""Service layer for prediction logic."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim
from descriptions.modeling.evaluate import load_per_label_thresholds
from descriptions.modeling.model import load_model
from descriptions.modeling.preprocess import load_preprocessors
from sklearn.pipeline import Pipeline


class PredictionService:
    """Service for handling genre predictions."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.model = None
        self.pipeline = None  # Pipeline (new approach)
        self.model_path = None
        self.vectorizer = None  # For backward compatibility
        self.mlb = None
        self.normalizer = None  # For backward compatibility
        self.feature_selector = None  # For backward compatibility
        self._is_loaded = False
        self._use_pipeline = False  # Flag to track if using Pipeline
    
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
            
            loaded_model = load_model(model_path)
            self.model_path = str(model_path)
            
            # Check if loaded model is a Pipeline (new approach) or separate components (backward compatibility)
            if isinstance(loaded_model, Pipeline):
                logger.info("Using Pipeline model (new approach)")
                self.pipeline = loaded_model
                self.model = loaded_model.named_steps['classifier']
                self._use_pipeline = True
                logger.success(f"✓ Pipeline loaded successfully: {model_path.name}")
                
                # Load mlb separately (for label encoding/decoding)
                logger.info("Loading MultiLabelBinarizer for label encoding/decoding...")
                try:
                    self.mlb = load_model("genre_binarizer")
                    logger.success("✓ MultiLabelBinarizer loaded successfully")
                except FileNotFoundError:
                    # Fallback: try loading all preprocessors
                    _, self.mlb, _, _ = load_preprocessors()
                    logger.info("✓ MultiLabelBinarizer loaded from preprocessors")
            else:
                # Backward compatibility: individual components
                logger.info("Using individual model components (backward compatibility mode)")
                self.model = loaded_model
                self._use_pipeline = False
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
    
    def _preprocess_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """
        Preprocess descriptions through the full pipeline.
        
        Args:
            descriptions: List of movie descriptions
        
        Returns:
            Dense array of preprocessed features ready for model prediction
        """
        if self._use_pipeline:
            # Use Pipeline for preprocessing (new approach)
            X_preprocessed = self.pipeline.named_steps['feature_selector'].transform(
                self.pipeline.named_steps['normalizer'].transform(
                    self.pipeline.named_steps['tfidf'].transform(descriptions)
                )
            )
            # Convert to dense array for LinearSVC (handle both sparse and dense)
            if hasattr(X_preprocessed, 'toarray'):
                return X_preprocessed.toarray()
            return X_preprocessed
        else:
            # Backward compatibility: individual components
            # Transform descriptions to TF-IDF features
            X = self.vectorizer.transform(descriptions)
            
            # Apply L2 normalization
            X = self.normalizer.transform(X)
            
            # Apply feature selection
            X = self.feature_selector.transform(X)
            
            # Convert to dense array for LinearSVC (handle both sparse and dense)
            if hasattr(X, 'toarray'):
                return X.toarray()
            return X
    
    def _get_prediction_probabilities(self, X_dense: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from model.
        
        Args:
            X_dense: Dense array of preprocessed features
        
        Returns:
            Array of prediction probabilities (n_samples, n_classes)
        """
        from scipy.special import expit
        
        # Get prediction scores
        y_scores = self.model.decision_function(X_dense)
        
        # Convert scores to probabilities using sigmoid function
        return expit(y_scores)
    
    def _determine_thresholds(
        self,
        threshold: Optional[Union[float, Dict[str, float]]],
        use_global_threshold: bool
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """
        Determine which thresholds to use for predictions.
        
        Args:
            threshold: Threshold value or dict of per-label thresholds
            use_global_threshold: Force use of global threshold
        
        Returns:
            Tuple of (per_label_thresholds, global_threshold)
        """
        per_label_thresholds: Optional[Dict[str, float]] = None
        global_threshold = 0.55
        
        if use_global_threshold:
            # Force use of global threshold
            if threshold is None or isinstance(threshold, dict):
                global_threshold = 0.55
            else:
                global_threshold = threshold
            logger.debug(f"Using global threshold: {global_threshold}")
        elif isinstance(threshold, dict):
            # Explicit per-label thresholds provided
            per_label_thresholds = threshold
            logger.debug(f"Using provided per-label thresholds for {len(per_label_thresholds)} labels")
        elif threshold is not None:
            # Explicit global threshold provided
            global_threshold = threshold
            logger.debug(f"Using global threshold: {global_threshold}")
        else:
            # Try to load per-label thresholds by default
            per_label_thresholds = load_per_label_thresholds()
            if per_label_thresholds:
                logger.debug(f"Using per-label thresholds for {len(per_label_thresholds)} labels")
            else:
                global_threshold = 0.55
                logger.debug(f"Per-label thresholds not found, using global threshold: {global_threshold}")
        
        return per_label_thresholds, global_threshold
    
    def _apply_thresholds(
        self,
        y_proba: np.ndarray,
        per_label_thresholds: Optional[Dict[str, float]],
        global_threshold: float
    ) -> np.ndarray:
        """
        Apply thresholds to probability predictions.
        
        Args:
            y_proba: Probability predictions (n_samples, n_classes)
            per_label_thresholds: Per-label thresholds dict or None
            global_threshold: Global threshold value
        
        Returns:
            Binary predictions array (n_samples, n_classes)
        """
        y_pred_binary = np.zeros_like(y_proba, dtype=int)
        n_classes = y_proba.shape[1]
        
        if per_label_thresholds:
            # Use per-label thresholds
            # Iterate only over the actual number of classes in the probability array
            for label_idx in range(min(n_classes, len(self.mlb.classes_))):
                label_name = self.mlb.classes_[label_idx]
                if label_name in per_label_thresholds:
                    label_threshold = per_label_thresholds[label_name]
                    y_pred_binary[:, label_idx] = (y_proba[:, label_idx] >= label_threshold).astype(int)
                else:
                    logger.debug(
                        f"Threshold not found for label '{label_name}', using global threshold {global_threshold}"
                    )
                    y_pred_binary[:, label_idx] = (y_proba[:, label_idx] >= global_threshold).astype(int)
        else:
            # Use global threshold
            y_pred_binary = (y_proba >= global_threshold).astype(int)
        
        return y_pred_binary
    
    def _apply_top_k_selection(
        self,
        y_proba: np.ndarray,
        y_pred_binary: np.ndarray,
        top_k: int
    ) -> np.ndarray:
        """
        Apply top-k selection to binary predictions.
        
        Args:
            y_proba: Probability predictions (n_samples, n_classes)
            y_pred_binary: Binary predictions (n_samples, n_classes)
            top_k: Maximum number of genres to select
        
        Returns:
            Binary predictions with top-k selection applied
        """
        for i in range(y_proba.shape[0]):
            # Get indices of genres that passed threshold
            passed_indices = np.where(y_pred_binary[i] == 1)[0]
            
            if len(passed_indices) > top_k:
                # Select top-k by probability from those that passed
                passed_proba = y_proba[i, passed_indices]
                top_k_passed_indices = passed_indices[np.argsort(passed_proba)[-top_k:][::-1]]
                
                # Reset all predictions for this sample
                y_pred_binary[i, :] = 0
                # Set only the top-k
                y_pred_binary[i, top_k_passed_indices] = 1
        
        return y_pred_binary
    
    def predict(
        self,
        descriptions: List[str],
        threshold: Optional[Union[float, Dict[str, float]]] = None,
        top_k: int = 3,
        model_path: Optional[str] = None,
        use_global_threshold: bool = False,
    ) -> List[List[str]]:
        """
        Predict genres for descriptions.
        
        Args:
            descriptions: List of movie descriptions
            threshold: Probability threshold for predictions. Can be:
                      - None: Uses per-label thresholds if available, else 0.55
                      - float: Global threshold value (overrides per-label thresholds)
                      - Dict[str, float]: Per-label thresholds (genre name -> threshold)
            top_k: Maximum number of top genres to select (default: 3).
                   The top k genres by probability will be selected, but only
                   those above their threshold will be returned.
            model_path: Optional model path (will reload if different)
            use_global_threshold: If True, forces use of global threshold even if
                                 per-label thresholds are available.
        
        Returns:
            List of lists of predicted genres
        """
        # Load model if not loaded or if different model requested
        if not self._is_loaded or (model_path and model_path != self.model_path):
            self.load_model(model_path)
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Preprocess descriptions
        X_dense = self._preprocess_descriptions(descriptions)
        
        # Get prediction probabilities
        y_proba = self._get_prediction_probabilities(X_dense)
        
        # Determine thresholds
        per_label_thresholds, global_threshold = self._determine_thresholds(threshold, use_global_threshold)
        
        # Apply thresholds
        y_pred_binary = self._apply_thresholds(y_proba, per_label_thresholds, global_threshold)
        
        # Apply top-k selection
        y_pred_binary = self._apply_top_k_selection(y_proba, y_pred_binary, top_k)
        
        # Decode predictions back to genre labels
        predicted_genres = self.mlb.inverse_transform(y_pred_binary)
        
        return predicted_genres
    
    def predict_with_confidence(
        self,
        descriptions: List[str],
        threshold: Optional[Union[float, Dict[str, float]]] = None,
        top_k: int = 3,
        model_path: Optional[str] = None,
        use_global_threshold: bool = False,
    ) -> Tuple[List[List[str]], List[Dict[str, float]]]:
        """
        Predict genres for descriptions with confidence scores.
        
        Args:
            descriptions: List of movie descriptions
            threshold: Probability threshold for predictions. Can be:
                      - None: Uses per-label thresholds if available, else 0.55
                      - float: Global threshold value (overrides per-label thresholds)
                      - Dict[str, float]: Per-label thresholds (genre name -> threshold)
            top_k: Maximum number of top genres to select (default: 3).
                   The top k genres by probability will be selected, but only
                   those above their threshold will be returned.
            model_path: Optional model path (will reload if different)
            use_global_threshold: If True, forces use of global threshold even if
                                 per-label thresholds are available.
        
        Returns:
            Tuple of (predicted genres list, confidence scores list)
            - predicted genres: List of lists of predicted genre names
            - confidence scores: List of dicts mapping genre name to confidence score
        """
        # Load model if not loaded or if different model requested
        if not self._is_loaded or (model_path and model_path != self.model_path):
            self.load_model(model_path)
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Preprocess descriptions
        X_dense = self._preprocess_descriptions(descriptions)
        
        # Get prediction probabilities
        y_proba = self._get_prediction_probabilities(X_dense)
        
        # Determine thresholds
        per_label_thresholds, global_threshold = self._determine_thresholds(threshold, use_global_threshold)
        
        # Apply thresholds
        y_pred_binary = self._apply_thresholds(y_proba, per_label_thresholds, global_threshold)
        
        # Apply top-k selection
        y_pred_binary = self._apply_top_k_selection(y_proba, y_pred_binary, top_k)
        
        # Decode predictions back to genre labels
        predicted_genres = self.mlb.inverse_transform(y_pred_binary)
        
        # Build confidence scores for each prediction
        confidence_scores = []
        for i, genres in enumerate(predicted_genres):
            conf_dict = {}
            for genre in genres:
                # Find the index of this genre
                if genre in self.mlb.classes_:
                    genre_idx = list(self.mlb.classes_).index(genre)
                    conf_dict[genre] = float(y_proba[i, genre_idx])
            confidence_scores.append(conf_dict)
        
        return predicted_genres, confidence_scores
    
    def get_description_length_stats(self) -> Optional[Dict[str, float]]:
        """
        Analyze description lengths from training data.
        
        Returns:
            Dictionary with description length statistics, or None if data not available
        """
        try:
            # Try to load interim data
            interim_path = INTERIM_DATA_DIR / "cleaned_movies.csv"
            if not interim_path.exists():
                logger.warning(f"Training data not found at {interim_path}")
                return None
            
            data = load_interim(interim_path)
            
            if "description" not in data.columns:
                logger.warning("'description' column not found in training data")
                return None
            
            # Calculate description lengths
            lengths = data["description"].fillna("").astype(str).str.len()
            lengths = lengths[lengths > 0]  # Remove empty descriptions
            
            if len(lengths) == 0:
                logger.warning("No valid descriptions found in training data")
                return None
            
            stats = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "median": float(lengths.median()),
                "q25": float(lengths.quantile(0.25)),
                "q75": float(lengths.quantile(0.75)),
                "optimal_min": int(lengths.quantile(0.25)),  # Use Q1 as recommended minimum
                "optimal_max": int(lengths.quantile(0.75)),  # Use Q3 as recommended maximum
            }
            
            logger.debug(f"Description length stats calculated: {stats}")
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating description length stats: {e}")
            return None
    
    def validate_description_length(
        self, 
        description: str,
        optimal_min: Optional[int] = None,
        optimal_max: Optional[int] = None
    ) -> Dict[str, Union[int, bool, Optional[str]]]:
        """
        Validate description length and provide recommendations.
        
        Args:
            description: Description text to validate
            optimal_min: Recommended minimum length (if None, will calculate from stats)
            optimal_max: Recommended maximum length (if None, will calculate from stats)
        
        Returns:
            Dictionary with validation results
        """
        length = len(description)
        
        # Get optimal ranges if not provided
        if optimal_min is None or optimal_max is None:
            stats = self.get_description_length_stats()
            if stats:
                optimal_min = stats.get("optimal_min", 100)
                optimal_max = stats.get("optimal_max", 500)
            else:
                # Default values if stats not available
                optimal_min = 100
                optimal_max = 500
        
        is_optimal = optimal_min <= length <= optimal_max
        recommendation = None
        
        if not is_optimal:
            if length < optimal_min:
                recommendation = (
                    f"Description is too short ({length} chars). "
                    f"Recommended minimum: {optimal_min} characters. "
                    f"Shorter descriptions may result in less accurate predictions."
                )
            elif length > optimal_max:
                recommendation = (
                    f"Description is quite long ({length} chars). "
                    f"Recommended maximum: {optimal_max} characters. "
                    f"Very long descriptions may not improve accuracy."
                )
        
        return {
            "length": length,
            "is_optimal": is_optimal,
            "recommendation": recommendation,
            "optimal_min": optimal_min,
            "optimal_max": optimal_max,
        }
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information including metrics and stats
        """
        # Get number of features
        if self._use_pipeline and self.pipeline:
            n_features = self.pipeline.named_steps['feature_selector'].k
        elif self.feature_selector:
            n_features = self.feature_selector.k
        else:
            n_features = 0
        
        info = {
            "model_name": Path(self.model_path).name if self.model_path else "unknown",
            "model_path": self.model_path or "not loaded",
            "model_loaded": self._is_loaded,
            "n_classes": len(self.mlb.classes_) if self.mlb else 0,
            "n_features": n_features,
            "metrics": None,
            "description_stats": None,
            "threshold_type": "unknown",
        }
        
        # Try to load metrics
        try:
            metrics_path = MODELS_DIR / "metrics_linearsvc.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    info["metrics"] = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load metrics: {e}")
        
        # Get description stats
        info["description_stats"] = self.get_description_length_stats()
        
        # Determine threshold type
        per_label_thresholds = load_per_label_thresholds()
        if per_label_thresholds:
            info["threshold_type"] = "per-label"
        else:
            info["threshold_type"] = "global"
        
        return info
    
    def is_ready(self) -> bool:
        """Check if service is ready (model loaded)."""
        return self._is_loaded


# Global service instance
prediction_service = PredictionService()
