from pathlib import Path
from typing import Any, Optional, Union

import joblib
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from descriptions.config import MODELS_DIR
from .preprocess import build_preprocessor, load_preprocessors

__all__ = ["build_model", "build_pipeline", "save_model", "load_model"]


# ---- PUBLIC API ----
def save_model(
    model: Any, 
    model_name: Union[str, Path]
) -> None:
    """
    Save a model to the models directory.
    
    Args:
        model: The model object to save (can be any sklearn model, preprocessor, etc.)
        model_name: Name of the model file (with or without .joblib extension) or full Path
    
    Raises:
        ValueError: If model_name is empty or invalid
        OSError: If the file cannot be written
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Handle Path objects
    if isinstance(model_name, Path):
        output_path = model_name
        # Ensure it has .joblib extension if not already present
        if output_path.suffix != ".joblib":
            output_path = output_path.with_suffix(".joblib")
    else:
        # Handle string names
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        model_name = model_name.strip()
        # Ensure it has an extension
        if not model_name.endswith(".joblib"):
            model_name = f"{model_name}.joblib"
        
        output_path = MODELS_DIR / model_name

    try:
        logger.info(f"Saving model to {output_path}...")
        joblib.dump(model, output_path)
        logger.success(f"Model saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {output_path}: {e}")
        raise


def load_model(model_name: Union[str, Path]) -> Any:
    """
    Load a saved model from the models directory.
    
    Args:
        model_name: Name of the model file (with or without .joblib extension) or full Path
    
    Returns:
        The loaded model object
    
    Raises:
        FileNotFoundError: If the model file does not exist
        ValueError: If model_name is empty or invalid
        Exception: If there's an error loading the model (e.g., corrupted file)
    """
    # Handle Path objects
    if isinstance(model_name, Path):
        input_path = model_name
        # Ensure it has .joblib extension if not already present
        if input_path.suffix != ".joblib":
            input_path = input_path.with_suffix(".joblib")
    else:
        # Handle string names
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        model_name = model_name.strip()
        # Ensure model_name has .joblib extension
        if not model_name.endswith(".joblib"):
            model_name = f"{model_name}.joblib"
        
        input_path = MODELS_DIR / model_name
    
    if not input_path.exists():
        error_msg = f"Model file not found at {input_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading model from {input_path}...")
        model = joblib.load(input_path)
        logger.success(f"Model loaded successfully from {input_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {input_path}: {e}")
        raise


def build_model(
    C: float = 1.0,
    penalty: str = 'l1',
    solver: str = 'liblinear',
    max_iter: int = 1000,
) -> OneVsRestClassifier:
    """
    Build the main classification model for multi-label genre prediction.
    
    Uses OneVsRestClassifier strategy with LogisticRegression as the base estimator.
    This allows predicting multiple genres (labels) for each movie description.
    
    Args:
        C: Inverse of regularization strength (default: 1.0). Smaller values specify stronger regularization.
        penalty: Type of regularization penalty ('l1' or 'l2', default: 'l1')
        solver: Algorithm to use for optimization (default: 'liblinear', required for L1 penalty)
        max_iter: Maximum number of iterations for convergence (default: 1000)
    
    Returns:
        OneVsRestClassifier with LogisticRegression base estimator
    
    Note:
        The 'liblinear' solver is required when using L1 penalty. For L2 penalty,
        you can use 'lbfgs' or 'sag' solvers as well.
    """
    base_estimator = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    clf = OneVsRestClassifier(base_estimator)
    return clf


def build_pipeline(
    use_fitted_preprocessor: bool = False,
    vectorizer_path: Optional[Path] = None,
    mlb_path: Optional[Path] = None,
    model_params: Optional[dict] = None,
) -> Pipeline:
    """
    Build the complete pipeline (preprocessor + model) for movie genre classification.
    
    The pipeline consists of:
    1. TfidfVectorizer: Transforms text descriptions into TF-IDF features
    2. OneVsRestClassifier: Multi-label classifier for genre prediction
    
    Note: The MultiLabelBinarizer (mlb) is not included in the pipeline as it's used
    separately for encoding/decoding target labels. It should be loaded separately
    when needed for transforming labels.
    
    Args:
        use_fitted_preprocessor: If True, load pre-fitted TfidfVectorizer. 
            If False, create a new unfitted one that will be fitted during training.
        vectorizer_path: Path to saved TfidfVectorizer (only used if use_fitted_preprocessor=True).
            If None, uses default path from MODELS_DIR.
        mlb_path: Path to saved MultiLabelBinarizer (only used if use_fitted_preprocessor=True).
            If None, uses default path from MODELS_DIR. Note: mlb is loaded but not used in pipeline.
        model_params: Optional dictionary of parameters to pass to build_model().
            Keys can include: 'C', 'penalty', 'solver', 'max_iter'.
    
    Returns:
        Pipeline object with 'vectorizer' and 'model' steps.
    
    Example:
        >>> # For training (unfitted preprocessor)
        >>> pipeline = build_pipeline(use_fitted_preprocessor=False)
        >>> pipeline.fit(X_train, y_train)
        
        >>> # For inference (fitted preprocessor)
        >>> pipeline = build_pipeline(use_fitted_preprocessor=True)
        >>> predictions = pipeline.predict(X_test)
    """
    if use_fitted_preprocessor:
        vectorizer, mlb = load_preprocessors(vectorizer_path, mlb_path)
        # Note: mlb is loaded but not used in pipeline - it's for label encoding/decoding
        logger.debug("Loaded fitted preprocessors (mlb loaded but not used in pipeline)")
    else:
        vectorizer, _ = build_preprocessor()
        logger.debug("Created new unfitted preprocessor")
    
    # Build model with optional parameters
    if model_params is None:
        model_params = {}
    model = build_model(**model_params)
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', model)
    ])
    
    logger.info("Pipeline built successfully with vectorizer and model steps")
    return pipeline
