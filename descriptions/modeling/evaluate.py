from pathlib import Path
from typing import Any, Optional

import json
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from descriptions.config import MODELS_DIR, PROCESSED_DATA_DIR
from descriptions.dataset import load_processed
from descriptions.modeling.model import load_model, get_model_name
from descriptions.modeling.train import split_data

app = typer.Typer()


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
    precision = precision_score(y, y_pred, average='micro', zero_division=0)
    recall = recall_score(y, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y, y_pred, average='micro', zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
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
    model_path: Path = MODELS_DIR / "model.joblib",
    data_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    metrics_path: Optional[Path] = None,
):
    """
    Evaluate a trained model on test data and save metrics.
    
    Args:
        model_path: Path to the saved model file
        data_path: Path to the processed data CSV file
        metrics_path: Path where metrics will be saved. If None, uses metrics_{model_name}.json
    """
    logger.info("Loading model...")
    model = load_model(model_path)
    logger.success("Model loaded successfully.")

    logger.info("Loading data...")
    data = load_processed(data_path)
    logger.success("Data loaded successfully.")

    logger.info("Splitting data into features and labels...")
    X, y, _ = split_data(data)
    logger.success("Data split successfully.")

    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X, y)
    logger.success("Model evaluated successfully.")

    # Generate metrics path from model name if not provided
    if metrics_path is None:
        model_name = get_model_name(model)
        metrics_path = MODELS_DIR / f"metrics_{model_name}.json"
    
    logger.info(f"Saving metrics to {metrics_path}...")
    save_metrics(metrics, metrics_path)
    logger.success("Metrics saved successfully.")


if __name__ == "__main__":
    app()