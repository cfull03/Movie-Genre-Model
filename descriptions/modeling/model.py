from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
from loguru import logger
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

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
    C: float = 1.0,
    penalty: str = "elasticnet",
    solver: str = "saga",
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-3,
    class_weight: str = "balanced",
    # Legacy SGDClassifier parameters (for backward compatibility)
    loss: str = None,
    alpha: float = None,
    learning_rate: str = None,
    early_stopping: bool = None,
) -> OneVsRestClassifier:
    """
    Build the main classification model for multi-label genre prediction.

    Uses OneVsRestClassifier strategy with LogisticRegression as the base estimator.
    This allows predicting multiple genres (labels) for each movie description.

    Default parameters are optimized via cross-validation (see overfitting_testing.ipynb).

    Args:
        C: Regularization strength (default: 1.0). Larger values = less regularization.
        penalty: Type of regularization penalty ('l1', 'l2', 'elasticnet', default: 'elasticnet')
        solver: Algorithm to use ('lbfgs', 'liblinear', 'saga', default: 'saga' for elasticnet)
        l1_ratio: Balance between L1 and L2 for elasticnet (default: 0.5)
        max_iter: Maximum number of iterations for convergence (default: 1000)
        tol: Tolerance for stopping criteria (default: 1e-3)
        class_weight: Class weight strategy ('balanced' or None, default: 'balanced')
        loss: Legacy parameter (ignored, kept for backward compatibility)
        alpha: Legacy parameter (ignored, kept for backward compatibility)
        learning_rate: Legacy parameter (ignored, kept for backward compatibility)
        early_stopping: Legacy parameter (ignored, kept for backward compatibility)

    Returns:
        OneVsRestClassifier with LogisticRegression base estimator

    Note:
        Best parameters from cross-validation:
        - C=1.0, penalty='elasticnet', solver='saga', l1_ratio=0.5
        - max_iter=1000, tol=1e-3, class_weight='balanced'
    """
    logger.debug(
        f"Building OneVsRestClassifier with LogisticRegression: "
        f"C={C}, penalty={penalty}, solver={solver}, "
        f"l1_ratio={l1_ratio}, max_iter={max_iter}, class_weight={class_weight}"
    )
    base_estimator = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    clf = OneVsRestClassifier(base_estimator)
    logger.debug("OneVsRestClassifier built successfully with LogisticRegression")
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
            Keys can include: 'loss', 'penalty', 'alpha', 'learning_rate', 'max_iter', 'tol', 'early_stopping'.

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
    # Import here to avoid circular import with preprocess.py
    from .preprocess import build_preprocessor, load_preprocessors

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

    pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

    logger.info("Pipeline built successfully with vectorizer and model steps")
    return pipeline


def get_params(model: Any) -> Dict[str, Any]:
    """
    Extract key parameters from a model's base estimator.

    For OneVsRestClassifier models, extracts parameters from the underlying
    base estimator (e.g., SGDClassifier).

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

    # LogisticRegression parameters
    if hasattr(base_estimator, "C"):
        params["C"] = base_estimator.C
    if hasattr(base_estimator, "penalty"):
        params["penalty"] = base_estimator.penalty
    if hasattr(base_estimator, "solver"):
        params["solver"] = base_estimator.solver
    if hasattr(base_estimator, "l1_ratio"):
        params["l1_ratio"] = base_estimator.l1_ratio
    if hasattr(base_estimator, "max_iter"):
        params["max_iter"] = base_estimator.max_iter
    if hasattr(base_estimator, "tol"):
        params["tol"] = base_estimator.tol
    if hasattr(base_estimator, "class_weight"):
        params["class_weight"] = base_estimator.class_weight
    if hasattr(base_estimator, "random_state"):
        params["random_state"] = base_estimator.random_state
    
    # Legacy SGDClassifier parameters (for backward compatibility)
    if hasattr(base_estimator, "loss"):
        params["loss"] = base_estimator.loss
    if hasattr(base_estimator, "alpha"):
        params["alpha"] = base_estimator.alpha
    if hasattr(base_estimator, "learning_rate"):
        params["learning_rate"] = base_estimator.learning_rate
    if hasattr(base_estimator, "early_stopping"):
        params["early_stopping"] = base_estimator.early_stopping

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
