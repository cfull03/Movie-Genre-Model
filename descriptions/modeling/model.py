from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from descriptions.config import MODELS_DIR

__all__ = [
    "build_model",
    "build_pipeline",
    "save_model",
    "load_model",
    "get_params",
    "get_model_name",
]


# ---- PUBLIC API ----
def save_model(model: Any, model_name: Union[str, Path]) -> None:
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
    C: float = 0.1,
    penalty: str = "l2",
    loss: str = "squared_hinge",
    max_iter: int = 1000,
    tol: float = 1e-3,
    class_weight: str = "balanced",
    dual: bool = False,
    random_state: int = 42,
) -> OneVsRestClassifier:
    """
    Build the main classification model for multi-label genre prediction.

    Uses OneVsRestClassifier strategy with LinearSVC as the base estimator.
    This allows predicting multiple genres (labels) for each movie description.

    Default parameters are optimized via cross-validation (see model_testing.ipynb).

    Args:
        C: Regularization strength (default: 0.1). Larger values = less regularization.
        penalty: Type of regularization penalty ('l1' or 'l2', default: 'l2')
        loss: Loss function ('hinge' or 'squared_hinge', default: 'squared_hinge')
        max_iter: Maximum number of iterations for convergence (default: 1000)
        tol: Tolerance for stopping criteria (default: 1e-3)
        class_weight: Class weight strategy ('balanced' or None, default: 'balanced')
        dual: Whether to solve the dual or primal optimization problem (default: False)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        OneVsRestClassifier with LinearSVC base estimator

    Note:
        Best parameters from cross-validation (model_testing.ipynb):
        - C=0.1, penalty='l2', loss='squared_hinge', max_iter=1000
        - tol=1e-3, class_weight='balanced', dual=False
    """
    logger.debug(
        f"Building OneVsRestClassifier with LinearSVC: "
        f"C={C}, penalty={penalty}, loss={loss}, "
        f"max_iter={max_iter}, tol={tol}, class_weight={class_weight}, dual={dual}"
    )
    base_estimator = LinearSVC(
        C=C,
        penalty=penalty,
        loss=loss,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        dual=dual,
        random_state=random_state,
    )
    clf = OneVsRestClassifier(base_estimator)
    logger.debug("OneVsRestClassifier built successfully with LinearSVC")
    return clf


def build_pipeline(
    model_params: Optional[dict] = None,
    k_features: int = 4500,
    vectorizer_params: Optional[dict] = None,
) -> Pipeline:
    """
    Build the complete prediction pipeline for movie genre classification.

    The pipeline consists of:
    1. TfidfVectorizer: Transforms text descriptions into TF-IDF features
    2. Normalizer: Applies L2 normalization to feature vectors
    3. SelectKBest: Feature selection using chi2 test
    4. OneVsRestClassifier: Multi-label classifier for genre prediction

    Note: The MultiLabelBinarizer (mlb) is NOT included in the pipeline as it's used
    separately for encoding/decoding target labels. It should be loaded separately
    when needed for transforming labels.

    Args:
        model_params: Optional dictionary of parameters to pass to build_model().
            Keys can include: 'C', 'penalty', 'loss', 'max_iter', 'tol', 'class_weight', 'dual', 'random_state'.
        k_features: Number of features to select with SelectKBest (default: 4500)
        vectorizer_params: Optional dictionary of parameters for TfidfVectorizer.
            If None, uses default parameters from preprocess.build_preprocessor().

    Returns:
        Pipeline object with all preprocessing and model steps.

    Example:
        >>> # For training
        >>> pipeline = build_pipeline(model_params={'C': 0.1}, k_features=4500)
        >>> pipeline.fit(X_train_text, y_train_binary)

        >>> # For inference
        >>> pipeline = load_model("genre_pipeline")
        >>> y_scores = pipeline.decision_function(X_test_text)
    """
    # Build vectorizer with default or custom parameters
    if vectorizer_params is None:
        vectorizer_params = {
            "max_features": 10000,
            "stop_words": "english",
            "ngram_range": (1, 3),
            "sublinear_tf": True,
            "max_df": 0.7,
            "min_df": 3,
            "use_idf": True,
        }
    
    vectorizer = TfidfVectorizer(**vectorizer_params)
    logger.debug(f"TfidfVectorizer configured: {vectorizer_params}")

    # Build normalizer
    normalizer = Normalizer(norm="l2")
    logger.debug("Normalizer configured: L2 norm")

    # Build feature selector
    feature_selector = SelectKBest(score_func=chi2, k=k_features)
    logger.debug(f"SelectKBest configured: chi2 test, k={k_features} features")

    # Build model with optional parameters
    if model_params is None:
        model_params = {}
    model = build_model(**model_params)
    logger.debug(f"Model built with parameters: {model_params}")

    # Create pipeline with all steps
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("normalizer", normalizer),
        ("feature_selector", feature_selector),
        ("classifier", model),
    ])

    logger.info("Pipeline built successfully with tfidf, normalizer, feature_selector, and classifier steps")
    return pipeline


def get_params(model: Any) -> Dict[str, Any]:
    """
    Extract key parameters from a model's base estimator.

    For OneVsRestClassifier models, extracts parameters from the underlying
    base estimator (e.g., LinearSVC).

    Args:
        model: Model object (OneVsRestClassifier, Pipeline, or base estimator)

    Returns:
        Dictionary of parameter names and values
    """
    # Handle Pipeline objects - extract the model step
    if isinstance(model, Pipeline):
        # Try to find a 'model' step
        if "model" in model.named_steps:
            model = model.named_steps["model"]
        else:
            # Use the last step if no 'model' step found
            model = model.steps[-1][1]

    # Extract base estimator from OneVsRestClassifier
    if isinstance(model, OneVsRestClassifier):
        base_estimator = model.estimator
    elif hasattr(model, "estimator"):
        base_estimator = model.estimator
    elif hasattr(model, "base_estimator"):
        base_estimator = model.base_estimator
    else:
        # Use the model itself if it's already a base estimator
        base_estimator = model

    # Extract key parameters
    params = {}

    # LinearSVC parameters
    if hasattr(base_estimator, "C"):
        params["C"] = base_estimator.C
    if hasattr(base_estimator, "penalty"):
        params["penalty"] = base_estimator.penalty
    if hasattr(base_estimator, "loss"):
        params["loss"] = base_estimator.loss
    if hasattr(base_estimator, "max_iter"):
        params["max_iter"] = base_estimator.max_iter
    if hasattr(base_estimator, "tol"):
        params["tol"] = base_estimator.tol
    if hasattr(base_estimator, "class_weight"):
        params["class_weight"] = base_estimator.class_weight
    if hasattr(base_estimator, "dual"):
        params["dual"] = base_estimator.dual
    if hasattr(base_estimator, "random_state"):
        params["random_state"] = base_estimator.random_state

    # Legacy LogisticRegression parameters (for backward compatibility)
    if hasattr(base_estimator, "solver"):
        params["solver"] = base_estimator.solver
    if hasattr(base_estimator, "l1_ratio"):
        params["l1_ratio"] = base_estimator.l1_ratio

    return params


def get_model_name(model: Any) -> str:
    """
    Generate a model name based on the base classifier and its key parameters.

    Args:
        model: Model object (OneVsRestClassifier, Pipeline, or base estimator)

    Returns:
        String name for the model file (without extension)
    """
    # Handle Pipeline objects - extract the model step
    if isinstance(model, Pipeline):
        # Try to find a 'model' step
        if "model" in model.named_steps:
            model = model.named_steps["model"]
        else:
            # Use the last step if no 'model' step found
            model = model.steps[-1][1]

    # Extract base estimator from OneVsRestClassifier
    if isinstance(model, OneVsRestClassifier):
        base_estimator = model.estimator
    elif hasattr(model, "estimator"):
        base_estimator = model.estimator
    elif hasattr(model, "base_estimator"):
        base_estimator = model.base_estimator
    else:
        # Use the model itself if it's already a base estimator
        base_estimator = model

    # Get the classifier class name
    classifier_name = base_estimator.__class__.__name__
    model_name = classifier_name

    # Convert to lowercase and replace special characters
    model_name = model_name.lower().replace(" ", "_")
    return model_name
