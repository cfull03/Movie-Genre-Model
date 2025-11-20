from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
import typer
from sklearn.model_selection import train_test_split

from descriptions.config import MODELS_DIR, PROCESSED_DATA_DIR
from descriptions.dataset import load_processed
from descriptions.modeling.model import build_model, save_model, get_model_name
from descriptions.modeling.preprocess import load_preprocessors

app = typer.Typer()


def split_data(
    data: pd.DataFrame,
    mlb: object = None,
) -> Tuple[pd.DataFrame, np.ndarray, object]:
    """
    Split processed data into features (TF-IDF) and labels (genres).
    
    Labels are transformed through the MultiLabelBinarizer to ensure proper formatting.
    
    Args:
        data: Processed DataFrame with TF-IDF features and genre columns
        mlb: Optional MultiLabelBinarizer. If None, will load from saved model.
    
    Returns:
        Tuple of (features_df, labels_array, mlb)
    """
    # Identify TF-IDF feature columns (columns starting with "tfidf_")
    feature_columns = [col for col in data.columns if col.startswith("tfidf_")]
    
    if not feature_columns:
        raise ValueError(
            "No TF-IDF feature columns found. Expected columns starting with 'tfidf_'"
        )
    
    # Load MultiLabelBinarizer
    if mlb is None:
        try:
            _, mlb = load_preprocessors()
            logger.info("Loaded MultiLabelBinarizer from saved model")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not load MultiLabelBinarizer. "
                "Please run preprocessing first to generate the binarizer."
            )
    
    # Extract genre columns in the order defined by MLB
    genre_columns = [col for col in mlb.classes_ if col in data.columns]
    
    if not genre_columns:
        raise ValueError("No genre label columns found in the processed data")
    
    # Check if all MLB classes are present in data
    missing_genres = [col for col in mlb.classes_ if col not in data.columns]
    if missing_genres:
        logger.warning(
            f"Some genre classes from MLB not found in data: {missing_genres[:5]}..."
            if len(missing_genres) > 5 else f"Missing genres: {missing_genres}"
        )
    
    # Extract features
    X = data[feature_columns]
    y_df = data[genre_columns]
    
    y_list = []
    for i in range(len(y_df)):
        row = y_df.iloc[i]
        genres = [col for col in genre_columns if row[col] == 1]
        y_list.append(genres)
    
    # Transform through MLB to ensure proper binary encoding
    y = mlb.transform(y_list)
    
    logger.info(
        f"Split data: {len(feature_columns)} feature columns, "
        f"{len(mlb.classes_)} label classes, {len(data)} samples"
    )
    logger.info(f"Labels shape after MLB transform: {y.shape}")
    
    return X, y, mlb


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> object:
    """
    Train the movie genre classification model.
    
    Args:
        X_train: Training features (TF-IDF features as DataFrame)
        y_train: Training labels (genre binary labels as numpy array from MLB)
    
    Returns:
        Trained model
    """
    logger.info(f"Training model with {len(X_train)} samples...")
    
    with tqdm(total=3, desc="Training model", unit="step") as pbar:
        # Build model (no vectorizer needed since features are already TF-IDF transformed)
        pbar.set_description("Building model")
        model = build_model()
        pbar.update(1)
        
        pbar.set_description("Converting to numpy array")
        # Convert DataFrame to numpy array for sklearn
        X_train_array = X_train.values
        pbar.update(1)
        
        pbar.set_description("Fitting model")
        # y_train is already a numpy array from MLB transform
        model.fit(X_train_array, y_train)
        pbar.update(1)
    
    logger.success("Model training complete!")
    
    return model


@app.command()
def main(
    processed_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Train a movie genre classification model.
    
    This script loads processed movie data (with TF-IDF features and genre labels),
    trains a multi-label classification model, and saves it to disk.
    
    Args:
        processed_path: Path to processed movies CSV file
        model_path: Path where the trained model will be saved
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    try:
        # Load processed data
        logger.info(f"Loading processed data from {processed_path}...")
        data = load_processed(processed_path)
        logger.success(f"Loaded {len(data)} samples from processed data")
        
        # Split features and labels (labels will be transformed through MLB)
        logger.info("Splitting data into features and labels...")
        X, y, mlb = split_data(data)
        logger.success("Data split successfully (labels transformed through MultiLabelBinarizer)")
        
        # Split into train and test sets
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        logger.success(
            f"Data split: {len(X_train)} training samples, {len(X_test)} test samples"
        )
        
        # Train model
        logger.info("Starting model training...")
        model = train_model(X_train, y_train)
        logger.success("Model training complete!")
        
        # Generate model name based on base classifier if using default path
        default_model_path = MODELS_DIR / "model.joblib"
        if model_path.resolve() == default_model_path.resolve() or model_path.name == "model.joblib":
            model_name = get_model_name(model)
            model_path = MODELS_DIR / f"{model_name}.joblib"
            logger.info(f"Generated model name: {model_name}")
        
        # Save model
        logger.info(f"Saving trained model to {model_path}...")
        save_model(model, model_path)
        logger.success(f"Model saved successfully to {model_path}")
        
        logger.success("Training pipeline completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise


if __name__ == "__main__":
    app()
