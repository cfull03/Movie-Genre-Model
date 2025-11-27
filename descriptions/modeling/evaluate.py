import json
from pathlib import Path
from typing import Any, List, Optional

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score
import typer

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim, load_processed
from descriptions.modeling.model import get_model_name, load_model
from descriptions.modeling.preprocess import load_preprocessors
from descriptions.modeling.train import prepare_features_and_labels, split_data

app = typer.Typer()

# Known preprocessor files that should be excluded from model search
PREPROCESSOR_FILES = {"tfidf_vectorizer.joblib", "genre_binarizer.joblib"}


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
def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model object (OneVsRestClassifier)
        X: Feature array (numpy array or DataFrame)
        y: True labels (numpy array, binary multi-label format)

    Returns:
        Dictionary containing evaluation metrics (micro-averaged for multi-label)
    """
    logger.debug(
        f"Evaluating model: X shape {X.shape if hasattr(X, 'shape') else 'unknown'}, y shape {y.shape}"
    )

    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
        logger.debug("Converted DataFrame to numpy array for evaluation")

    logger.debug("Generating predictions from model...")
    y_pred = model.predict(X)
    logger.debug(f"Predictions generated: shape {y_pred.shape}")

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
        # Set experiment and start new run
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"✓ Using existing MLflow experiment: '{experiment_name}'")
        except Exception:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"✓ Created new MLflow experiment: '{experiment_name}'")
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
            vectorizer, mlb = load_preprocessors()
            logger.success("✓ Preprocessors loaded successfully")
            logger.info("Transforming data using saved preprocessors...")
            X, y, _, _ = prepare_features_and_labels(data, vectorizer=vectorizer, mlb=mlb)
            logger.success(
                f"✓ Data transformed: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} labels"
            )

        logger.success(
            f"✓ Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} labels"
        )

        # Log data info to MLflow
        if mlflow.active_run():
            mlflow.log_param("evaluation_data_path", str(data_path))
            mlflow.log_param("evaluation_data_type", "processed" if use_processed else "interim")
            mlflow.log_param("evaluation_samples", len(data))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_classes", y.shape[1])
            logger.debug("Data information logged to MLflow")

        logger.info("=" * 70)
        logger.info("Evaluating model performance")
        logger.info("=" * 70)
        logger.info(f"Evaluating model on {len(X)} samples...")
        metrics = evaluate_model(model, X, y)
        logger.success("✓ Model evaluation completed successfully")

        # Log metrics to MLflow
        if mlflow.active_run():
            logger.info("Logging evaluation metrics to MLflow...")
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", value)
                logger.debug(f"  eval_{metric_name} = {value:.4f}")
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
