import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function for converting scores to probabilities
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import typer

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim, load_processed
from descriptions.modeling.mlflow_utils import (
    calculate_file_hash,
    log_data_info,
    log_metrics_dict,
    setup_experiment,
)
from descriptions.modeling.model import get_model_name, load_model
from descriptions.modeling.preprocess import load_preprocessors
from descriptions.modeling.train import prepare_features_and_labels, split_data

app = typer.Typer()

# Known preprocessor files that should be excluded from model search
PREPROCESSOR_FILES = {
    "tfidf_vectorizer.joblib",
    "genre_binarizer.joblib",
    "normalizer.joblib",
    "feature_selector.joblib",
}


# ---- PRIVATE HELPER FUNCTIONS ----
def _find_model_files() -> List[Path]:
    """
    Find all scikit-learn model files in the models directory.

    Excludes known preprocessor files (tfidf_vectorizer, genre_binarizer).

    Returns:
        List of Path objects pointing to model files
    """
    if not MODELS_DIR.exists():
        return []

    model_files = []
    for file_path in MODELS_DIR.glob("*.joblib"):
        if file_path.name not in PREPROCESSOR_FILES:
            model_files.append(file_path)

    return sorted(model_files)


def _find_default_model() -> Optional[Path]:
    """
    Find the default model file to use for evaluation.

    Priority:
    1. If 'model.joblib' exists, use it
    2. Otherwise, use the first scikit-learn model file found

    Returns:
        Path to model file, or None if no model found
    """
    # First check for the default model.joblib
    default_model = MODELS_DIR / "model.joblib"
    if default_model.exists():
        return default_model

    # Otherwise, find any scikit-learn model file
    model_files = _find_model_files()
    if model_files:
        return model_files[0]

    return None


def load_per_label_thresholds(
    thresholds_path: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """
    Load per-label thresholds from JSON file.

    Args:
        thresholds_path: Path to per-label thresholds JSON file.
                        Defaults to MODELS_DIR / "per_label_thresholds.json"

    Returns:
        Dictionary mapping genre names to thresholds, or None if file doesn't exist
    """
    if thresholds_path is None:
        thresholds_path = MODELS_DIR / "per_label_thresholds.json"

    if not thresholds_path.exists():
        logger.debug(f"Per-label thresholds file not found at {thresholds_path}")
        return None

    try:
        with open(thresholds_path, "r") as f:
            data = json.load(f)
            thresholds = data.get("per_label_thresholds", {})
            logger.debug(f"Loaded per-label thresholds for {len(thresholds)} labels")
            return thresholds
    except Exception as e:
        logger.warning(f"Error loading per-label thresholds from {thresholds_path}: {e}")
        return None


def display_metrics(metrics: dict) -> None:
    """
    Display evaluation metrics in a formatted, visually appealing way.

    Args:
        metrics: Dictionary of metric names and values
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 MODEL EVALUATION METRICS")
    logger.info("=" * 70)

    # Format metric names for better readability
    metric_labels = {
        "hamming_loss": "Hamming Loss",
        "f1": "F1 Score (micro)",
        "precision": "Precision (micro)",
        "recall": "Recall (micro)",
        "jaccard": "Jaccard Score (micro)",
    }

    for metric, value in metrics.items():
        label = metric_labels.get(metric, metric.replace("_", " ").title())
        logger.info(f"  {label:25s}: {value:.4f} ({value * 100:.2f}%)")

    logger.info("=" * 70)
    logger.info("")


# ---- PUBLIC API FUNCTIONS ----
def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: Union[float, Dict[str, float], None] = None,
    mlb: Optional[Any] = None,
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model object (OneVsRestClassifier)
        X: Feature array (numpy array or DataFrame)
        y: True labels (numpy array, binary multi-label format)
        threshold: Probability threshold(s) for predictions.
                   - If float: Global threshold for all labels (default: 0.55)
                   - If Dict[str, float]: Per-label thresholds (genre name -> threshold)
                   - If None: Uses per-label thresholds if available, else 0.55
        mlb: MultiLabelBinarizer (required if using per-label thresholds).
             If None and per-label thresholds provided, will attempt to load from preprocessors.

    Returns:
        Dictionary containing evaluation metrics (micro-averaged for multi-label)
    """
    # Handle threshold parameter
    per_label_thresholds: Optional[Dict[str, float]] = None
    global_threshold: float = 0.55

    if threshold is None:
        # Try to load per-label thresholds
        per_label_thresholds = load_per_label_thresholds()
        if per_label_thresholds:
            logger.info("Using per-label thresholds from saved file")
        else:
            logger.info("Using default global threshold: 0.55")
            global_threshold = 0.55
    elif isinstance(threshold, dict):
        per_label_thresholds = threshold
        logger.info(f"Using provided per-label thresholds for {len(per_label_thresholds)} labels")
    else:
        global_threshold = threshold
        logger.info(f"Using global threshold: {global_threshold}")

    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
        logger.debug("Converted DataFrame to numpy array for evaluation")

    logger.debug(
        f"Evaluating model: X shape {X.shape if hasattr(X, 'shape') else 'unknown'}, y shape {y.shape}"
    )

    # LinearSVC doesn't have predict_proba, so we use decision_function
    # and convert scores to probabilities using sigmoid function
    logger.debug("Generating prediction probabilities...")
    y_scores = model.decision_function(X)
    logger.debug(f"Decision scores generated: shape {y_scores.shape}")

    # Convert scores to probabilities using sigmoid function
    y_proba = expit(y_scores)
    logger.debug(f"Probabilities generated: shape {y_proba.shape}")

    # Apply threshold(s) to get binary predictions
    if per_label_thresholds:
        # Use per-label thresholds
        if mlb is None:
            # Try to load mlb from preprocessors
            try:
                _, mlb, _, _ = load_preprocessors()
            except Exception as e:
                logger.error(f"MultiLabelBinarizer required for per-label thresholds: {e}")
                raise ValueError(
                    "MultiLabelBinarizer (mlb) required when using per-label thresholds"
                )

        y_pred = np.zeros_like(y_proba, dtype=int)
        for label_idx, label_name in enumerate(mlb.classes_):
            if label_name in per_label_thresholds:
                label_threshold = per_label_thresholds[label_name]
                y_pred[:, label_idx] = (y_proba[:, label_idx] >= label_threshold).astype(int)
            else:
                logger.warning(
                    f"Threshold not found for label '{label_name}', using global threshold {global_threshold}"
                )
                y_pred[:, label_idx] = (y_proba[:, label_idx] >= global_threshold).astype(int)
        logger.debug("Binary predictions generated using per-label thresholds")
    else:
        # Use global threshold
        y_pred = (y_proba >= global_threshold).astype(int)
        logger.debug(f"Binary predictions generated using global threshold {global_threshold}")

    logger.debug(f"Binary predictions shape: {y_pred.shape}")

    # For multi-label classification, use average='micro' which calculates metrics globally
    logger.debug("Calculating evaluation metrics (micro-averaged)...")
    hamming = hamming_loss(y, y_pred)
    f1 = f1_score(y, y_pred, average="micro", zero_division=0)
    recall = recall_score(y, y_pred, average="micro", zero_division=0)
    precision = precision_score(y, y_pred, average="micro", zero_division=0)
    jaccard = jaccard_score(y, y_pred, average="micro", zero_division=0)

    metrics = {
        "hamming_loss": float(hamming),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }

    logger.debug("Evaluation metrics calculated successfully")
    return metrics


def save_metrics(metrics: dict, metrics_path: Path) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of metrics to save
        metrics_path: Path where metrics will be saved
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


@app.command()
def main(
    model_path: Optional[str] = typer.Option(
        None,
        help="Path to the saved model file. If not provided, automatically finds a scikit-learn model.",
    ),
    data_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    metrics_path: Optional[Path] = None,
    experiment_name: str = "movie-genre-classification",
    run_id: Optional[str] = None,
    use_processed: bool = typer.Option(
        False,
        "--use-processed",
        help="If True, expects processed data (with TF-IDF features). If False, uses interim data and transforms with saved preprocessors.",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Global probability threshold for predictions (default: None, uses per-label thresholds if available, else 0.55).",
    ),
    use_global_threshold: bool = typer.Option(
        False,
        "--use-global-threshold",
        help="Force use of global threshold instead of per-label thresholds.",
    ),
):
    """
    Evaluate a trained model on test data and save metrics.

    This function can work with either:
    1. Interim data (default): Loads raw data and transforms using saved preprocessors
    2. Processed data (--use-processed): Uses already-processed data with TF-IDF features

    For training, use train.py which properly handles train/test splits before preprocessing.

    Args:
        model_path: Path to the saved model file (as string). If None, automatically finds a scikit-learn model.
        data_path: Path to the data CSV file (interim or processed, depending on use_processed flag)
        metrics_path: Path where metrics will be saved. If None, uses metrics_{model_name}.json
        experiment_name: MLflow experiment name (default: "movie-genre-classification")
        run_id: Optional MLflow run ID to log metrics to. If None and no active run, starts new run.
        use_processed: If True, expects processed data. If False, uses interim data and transforms it.
    """
    logger.info("=" * 70)
    logger.info("Setting up MLflow experiment tracking")
    logger.info("=" * 70)

    # Set up MLflow if run_id provided or if no active run exists
    if run_id:
        mlflow.start_run(run_id=run_id)
        logger.info(f"✓ Continuing existing MLflow run: {run_id}")
    elif mlflow.active_run() is None:
        # Set experiment and start new run using utility function
        setup_experiment(experiment_name, create_if_not_exists=True)
        mlflow.start_run(run_name="evaluation")
        logger.info("✓ Started new MLflow run for evaluation")

    try:
        # Convert string to Path if provided
        model_path_obj: Optional[Path] = None
        if model_path:
            model_path_obj = Path(model_path)

        # Auto-detect model if not provided
        if model_path_obj is None:
            model_path_obj = _find_default_model()
            if model_path_obj is None:
                available_models = _find_model_files()
                if available_models:
                    logger.error(
                        f"No model file specified and no default model found. "
                        f"Available models: {[m.name for m in available_models]}"
                    )
                else:
                    logger.error(
                        "No model files found in models directory. "
                        "Please train a model first or specify a model_path."
                    )
                raise FileNotFoundError("No model file found for evaluation")
            logger.info(f"Auto-detected model: {model_path_obj.name}")
        elif not model_path_obj.exists():
            # If provided path doesn't exist, try to find it in MODELS_DIR
            if not model_path_obj.is_absolute():
                model_path_obj = MODELS_DIR / model_path_obj
            if not model_path_obj.exists():
                # Try to find any model file
                found_model = _find_default_model()
                if found_model:
                    logger.warning(
                        f"Model file {model_path_obj} not found. Using auto-detected model: {found_model.name}"
                    )
                    model_path_obj = found_model
                else:
                    raise FileNotFoundError(f"Model file not found at {model_path_obj}")

        logger.info("=" * 70)
        logger.info("Loading model and data")
        logger.info("=" * 70)
        logger.info(f"Loading model from {model_path_obj}...")
        model = load_model(model_path_obj)
        logger.success(f"✓ Model loaded successfully: {model_path_obj.name}")

        # Log model path to MLflow
        if mlflow.active_run():
            mlflow.log_param("evaluation_model_path", str(model_path_obj))
            logger.debug("Model path logged to MLflow")

        # Load data (either processed or interim)
        if use_processed:
            logger.info(f"Loading processed data from {data_path}...")
            data = load_processed(data_path)
            logger.success(f"✓ Processed data loaded successfully: {len(data)} samples")
            logger.info("Splitting processed data into features (X) and labels (y)...")
            X, y, mlb = split_data(data)
        else:
            logger.info(f"Loading interim data from {data_path}...")
            data = load_interim(data_path)
            logger.success(f"✓ Interim data loaded successfully: {len(data)} samples")
            logger.info("Loading saved preprocessors...")
            vectorizer, mlb, normalizer, feature_selector = load_preprocessors()
            logger.success(
                "✓ Preprocessors loaded successfully (including normalizer and feature selector)"
            )

            # Prepare features and labels on ALL data first (matching notebook approach)
            # This filters the data consistently before splitting
            logger.info("Preparing features and labels on all data (filtering invalid samples)...")
            X_all, y_all, _, _, _, _ = prepare_features_and_labels(
                data,
                vectorizer=vectorizer,
                mlb=mlb,
                normalizer=normalizer,
                feature_selector=feature_selector,
            )
            logger.success(
                f"✓ All data transformed: {X_all.shape[0]} samples, {X_all.shape[1]} features, {y_all.shape[1]} labels"
            )

            # Now split the filtered data into train/test sets (same split as training)
            logger.info(
                "Splitting filtered data into train and test sets (matching training split)..."
            )
            X_train, X, y_train, y = train_test_split(
                X_all,
                y_all,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            )
            logger.success(
                f"✓ Test set: {len(X)} samples ({len(X) / len(X_all) * 100:.1f}% of filtered data)"
            )
            logger.info(
                f"✓ Train set: {len(X_train)} samples ({len(X_train) / len(X_all) * 100:.1f}% of filtered data)"
            )

        logger.success(
            f"✓ Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} labels"
        )

        # Log data info to MLflow using utility function
        if mlflow.active_run():
            if data_path.exists():
                data_hash = calculate_file_hash(data_path)
            else:
                data_hash = None
            log_data_info(data_path, data_hash=data_hash)
            mlflow.log_param("evaluation_data_type", "processed" if use_processed else "interim")
            mlflow.log_param("evaluation_samples", len(data))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_classes", y.shape[1])
            logger.debug("Data information logged to MLflow")

        logger.info("=" * 70)
        logger.info("Evaluating model performance")
        logger.info("=" * 70)

        # Determine which threshold strategy to use
        threshold_param: Union[float, Dict[str, float], None] = None
        per_label_thresholds_loaded: Optional[Dict[str, float]] = None

        if use_global_threshold:
            # Force use of global threshold
            threshold_param = threshold if threshold is not None else 0.55
            logger.info(f"Using global threshold: {threshold_param}")
        else:
            # Try to use per-label thresholds by default
            per_label_thresholds_loaded = load_per_label_thresholds()
            if per_label_thresholds_loaded:
                threshold_param = per_label_thresholds_loaded
                logger.info(
                    f"Using per-label thresholds for {len(per_label_thresholds_loaded)} labels"
                )
            else:
                # Fall back to global threshold
                threshold_param = threshold if threshold is not None else 0.55
                logger.info(
                    f"Per-label thresholds not found, using global threshold: {threshold_param}"
                )

        logger.info(f"Evaluating model on {len(X)} samples...")
        metrics = evaluate_model(model, X, y, threshold=threshold_param, mlb=mlb)
        logger.success("✓ Model evaluation completed successfully")

        # Log metrics to MLflow using utility function
        if mlflow.active_run():
            logger.info("Logging evaluation metrics to MLflow...")
            if per_label_thresholds_loaded and not use_global_threshold:
                mlflow.log_param("evaluation_threshold_type", "per_label")
                mlflow.log_param("evaluation_threshold_count", len(per_label_thresholds_loaded))
                # Log threshold statistics
                threshold_values = list(per_label_thresholds_loaded.values())
                mlflow.log_param("evaluation_threshold_mean", float(np.mean(threshold_values)))
                mlflow.log_param("evaluation_threshold_min", float(np.min(threshold_values)))
                mlflow.log_param("evaluation_threshold_max", float(np.max(threshold_values)))
                # Log per-label thresholds as a JSON artifact
                mlflow.log_dict(
                    per_label_thresholds_loaded, "evaluation_per_label_thresholds.json"
                )
            else:
                mlflow.log_param("evaluation_threshold_type", "global")
                mlflow.log_param("evaluation_threshold", threshold_param)
            log_metrics_dict(metrics, prefix="eval_")
            logger.success("✓ Metrics logged to MLflow")

        # Generate metrics path from model name if not provided
        if metrics_path is None:
            model_name = get_model_name(model)
            metrics_path = MODELS_DIR / f"metrics_{model_name}.json"

        logger.info(f"Saving metrics to {metrics_path}...")
        save_metrics(metrics, metrics_path)
        logger.success(f"✓ Metrics saved successfully to {metrics_path}")

        # Display metrics in a formatted, visually appealing way
        display_metrics(metrics)

        logger.info("=" * 70)
        logger.success("🎉 Evaluation pipeline completed successfully!")
        logger.info("=" * 70)

    finally:
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("MLflow run ended")


if __name__ == "__main__":
    app()
