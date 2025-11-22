from pathlib import Path
from typing import Any, Optional, List

import json
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
import mlflow

from descriptions.config import MODELS_DIR, PROCESSED_DATA_DIR
from descriptions.dataset import load_processed
from descriptions.modeling.model import load_model, get_model_name
from descriptions.modeling.train import split_data

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
    logger.info("=" * 60)
    logger.info("📊 MODEL EVALUATION METRICS")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")
    logger.info("=" * 60)


# ---- PUBLIC API FUNCTIONS ----
def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model object
        X: Feature array (numpy array or DataFrame)
        y: True labels (numpy array)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    y_pred = model.predict(X)
    
    # For multi-label classification, use average='micro' or 'macro'
    # Using 'micro' which calculates metrics globally
    hamming = hamming_loss(y, y_pred)
    f1 = f1_score(y, y_pred, average='micro', zero_division=0)
    jaccard = jaccard_score(y, y_pred, average='micro', zero_division=0)
    
    return {
        "hamming_loss": float(hamming),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


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
    model_path: Optional[str] = typer.Option(None, help="Path to the saved model file. If not provided, automatically finds a scikit-learn model."),
    data_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    metrics_path: Optional[Path] = None,
    experiment_name: str = "movie-genre-classification",
    run_id: Optional[str] = None,
):
    """
    Evaluate a trained model on test data and save metrics.
    
    Args:
        model_path: Path to the saved model file (as string). If None, automatically finds a scikit-learn model.
        data_path: Path to the processed data CSV file
        metrics_path: Path where metrics will be saved. If None, uses metrics_{model_name}.json
        experiment_name: MLflow experiment name (default: "movie-genre-classification")
        run_id: Optional MLflow run ID to log metrics to. If None and no active run, starts new run.
    """
    # Set up MLflow if run_id provided or if no active run exists
    if run_id:
        mlflow.start_run(run_id=run_id)
        logger.info(f"Continuing MLflow run: {run_id}")
    elif mlflow.active_run() is None:
        # Set experiment and start new run
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name="evaluation")
        logger.info("Started new MLflow run for evaluation")
    
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
        
        logger.info(f"Loading model from {model_path_obj}...")
        model = load_model(model_path_obj)
        logger.success("Model loaded successfully.")
        
        # Log model path to MLflow
        if mlflow.active_run():
            mlflow.log_param("evaluation_model_path", str(model_path_obj))

        logger.info("Loading data...")
        data = load_processed(data_path)
        logger.success("Data loaded successfully.")
        
        # Log data info to MLflow
        if mlflow.active_run():
            mlflow.log_param("evaluation_data_path", str(data_path))
            mlflow.log_param("evaluation_samples", len(data))

        logger.info("Splitting data into features and labels...")
        X, y, _ = split_data(data)
        logger.success("Data split successfully.")

        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X, y)
        logger.success("Model evaluated successfully.")
        
        # Log metrics to MLflow
        if mlflow.active_run():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"eval_{metric_name}", value)
            logger.info("Metrics logged to MLflow")

        # Generate metrics path from model name if not provided
        if metrics_path is None:
            model_name = get_model_name(model)
            metrics_path = MODELS_DIR / f"metrics_{model_name}.json"
        
        logger.info(f"Saving metrics to {metrics_path}...")
        save_metrics(metrics, metrics_path)
        logger.success("Metrics saved successfully.")

        # Display metrics in a formatted, visually appealing way
        display_metrics(metrics)
        
    finally:
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("MLflow run ended")

if __name__ == "__main__":
    app()