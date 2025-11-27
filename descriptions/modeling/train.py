from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim
from descriptions.modeling.model import build_model, get_model_name, save_model
from descriptions.modeling.preprocess import (
    _generate_descriptions,
    _generate_targets,
    save_preprocessors,
)

app = typer.Typer()


def split_data(
    data: pd.DataFrame,
    mlb: object = None,
) -> Tuple[pd.DataFrame, np.ndarray, object]:
    """
    Split processed data into features (TF-IDF) and labels (genres).

    This function is for use with already-processed data (e.g., during evaluation).
    For training, use prepare_features_and_labels() instead to avoid data leakage.

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
            from descriptions.modeling.preprocess import load_preprocessors

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
            if len(missing_genres) > 5
            else f"Missing genres: {missing_genres}"
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


def prepare_features_and_labels(
    data: pd.DataFrame,
    vectorizer=None,
    mlb=None,
) -> Tuple[pd.DataFrame, np.ndarray, object, object]:
    """
    Generate TF-IDF features and multi-label targets from raw data.

    If vectorizer and mlb are None, they will be fitted on the data.
    If provided, they will be used to transform the data.

    Args:
        data: DataFrame with 'description' and 'genre' columns
        vectorizer: Optional pre-fitted TfidfVectorizer. If None, creates and fits a new one.
        mlb: Optional pre-fitted MultiLabelBinarizer. If None, creates and fits a new one.

    Returns:
        Tuple of (features_df, labels_array, vectorizer, mlb)
    """
    logger.info("Generating TF-IDF features from descriptions...")
    X_sparse, vectorizer = _generate_descriptions(data, vectorizer=vectorizer)

    logger.info("Generating multi-label genre targets...")
    y, mlb = _generate_targets(data, mlb=mlb)

    # Convert sparse matrix to dense DataFrame
    logger.debug("Converting sparse TF-IDF matrix to dense DataFrame...")
    X_df = pd.DataFrame(
        X_sparse.toarray(),
        index=data.index,
        columns=[f"tfidf_{i}" for i in range(X_sparse.shape[1])],
    )

    logger.info(
        f"Features and labels prepared: {X_df.shape[0]} samples, "
        f"{X_df.shape[1]} features, {y.shape[1]} labels"
    )

    return X_df, y, vectorizer, mlb


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_params: Optional[dict] = None,
) -> object:
    """
    Train the movie genre classification model.

    Args:
        X_train: Training features (TF-IDF features as DataFrame)
        y_train: Training labels (genre binary labels as numpy array from MLB)
        model_params: Optional dictionary of model hyperparameters (C, penalty, solver, max_iter)

    Returns:
        Trained OneVsRestClassifier model
    """
    n_samples, n_features = X_train.shape[0], X_train.shape[1]
    n_labels = y_train.shape[1]
    logger.info(f"Training model: {n_samples} samples, {n_features} features, {n_labels} labels")

    with tqdm(total=3, desc="Training model", unit="step") as pbar:
        # Build model (no vectorizer needed since features are already TF-IDF transformed)
        pbar.set_description("Building OneVsRestClassifier with LogisticRegression")
        if model_params is None:
            model_params = {}
        model = build_model(**model_params)
        logger.debug(f"Model built with parameters: {model_params}")
        pbar.update(1)

        pbar.set_description("Converting features to numpy array")
        # Convert DataFrame to numpy array for sklearn
        X_train_array = X_train.values
        logger.debug(f"Features converted: shape {X_train_array.shape}")
        pbar.update(1)

        pbar.set_description("Fitting model (this may take a while)")
        # y_train is already a numpy array from MLB transform
        model.fit(X_train_array, y_train)
        logger.debug("Model fitting completed")
        pbar.update(1)

    logger.success(f"Model training complete! Trained on {n_samples} samples")

    return model


def train_test_split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Split features and labels into training and testing sets.

    Args:
        X: Feature DataFrame (TF-IDF features)
        y: Label array (genre binary labels from MultiLabelBinarizer)
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle data before splitting (default: True)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    logger.info(
        f"Train/test split: {1 - test_size:.0%} train, {test_size:.0%} test "
        f"(random_state={random_state}, shuffle={shuffle})"
    )
    logger.info(f"Split complete: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


@app.command()
def main(
    interim_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    force: bool = False,
    experiment_name: str = "movie-genre-classification",
    run_name: Optional[str] = None,
    # Model hyperparameters
    C: float = 50.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 2000,
) -> None:
    """
    Train a movie genre classification model.

    This script loads interim movie data (with description and genre columns),
    splits into train/test sets, fits TF-IDF and MLB on training data only,
    transforms both sets, and trains a multi-label classification model.

    Args:
        interim_path: Path to interim movies CSV file (with description and genre columns)
        model_path: Path where the trained model will be saved
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        force: If True, retrain even if model already exists (default: False)
        experiment_name: MLflow experiment name (default: "movie-genre-classification")
        run_name: Optional name for MLflow run. If None, auto-generated from hyperparameters
        C: Inverse of regularization strength (default: 50.0). Smaller values = stronger regularization.
        penalty: Type of regularization penalty - 'l1' or 'l2' (default: 'l2')
        solver: Algorithm to use for optimization (default: 'lbfgs', works with L2 penalty)
        max_iter: Maximum number of iterations for convergence (default: 2000)
    """
    # Set up MLflow experiment
    logger.info("=" * 70)
    logger.info("Setting up MLflow experiment tracking")
    logger.info("=" * 70)
    try:
        mlflow.create_experiment(experiment_name)
        logger.info(f"✓ Created new MLflow experiment: '{experiment_name}'")
    except Exception:
        mlflow.set_experiment(experiment_name)
        logger.info(f"✓ Using existing MLflow experiment: '{experiment_name}'")

    # Generate run name from hyperparameters if not provided
    if run_name is None:
        run_name = f"C{C}_penalty-{penalty}_solver-{solver}_maxiter-{max_iter}"
    logger.info(f"MLflow run name: '{run_name}'")

    # Start MLflow run - each unique parameter combination creates a new run
    with mlflow.start_run(run_name=run_name):
        try:
            # Determine final model path early to check if it exists
            default_model_path = MODELS_DIR / "model.joblib"
            is_default_path = (
                model_path.resolve() == default_model_path.resolve()
                or model_path.name == "model.joblib"
            )

            # Build model params dict from command line arguments
            model_params_dict = {
                "C": C,
                "penalty": penalty,
                "solver": solver,
                "max_iter": max_iter,
            }

            # Build model to get its name (needed for default path)
            # This is fast since we're just creating an untrained model
            temp_model = build_model(**model_params_dict)
            if is_default_path:
                model_name = get_model_name(temp_model)
                final_model_path = MODELS_DIR / f"{model_name}.joblib"
                logger.info(f"Generated model name: '{model_name}'")
            else:
                final_model_path = model_path

            # Log model parameters to MLflow
            logger.info("Logging model hyperparameters to MLflow...")
            for key, value in model_params_dict.items():
                mlflow.log_param(f"model_{key}", value)
                logger.debug(f"  model_{key} = {value}")
            mlflow.set_tag("model_type", "LogisticRegression-OneVsRest")
            mlflow.set_tag("task", "multi-label-classification")
            logger.success("Model parameters logged to MLflow")

            # Check if model already exists (before loading data)
            if final_model_path.exists() and not force:
                logger.warning(f"Model already exists at {final_model_path}")
                logger.info("Skipping training. Use --force to retrain.")
                return

            logger.info("=" * 70)
            logger.info("Loading raw data")
            logger.info("=" * 70)
            logger.info(f"Loading interim data from {interim_path}...")
            data = load_interim(interim_path)
            logger.success(f"✓ Loaded {len(data)} samples from interim data")

            # Log data info to MLflow
            logger.debug("Logging data information to MLflow...")
            mlflow.log_param("data_path", str(interim_path))
            mlflow.log_param("total_samples", len(data))

            # Split into train and test sets FIRST (before preprocessing to avoid data leakage)
            logger.info("=" * 70)
            logger.info("Splitting data into train and test sets")
            logger.info("=" * 70)
            logger.info(
                f"Train/test split: {1 - test_size:.0%} train, {test_size:.0%} test (random_state={random_state})"
            )
            data_train, data_test = train_test_split(
                data,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )
            logger.success(
                f"✓ Data split complete: {len(data_train)} training samples, {len(data_test)} test samples"
            )

            # Fit TF-IDF and MLB on TRAINING data only
            logger.info("=" * 70)
            logger.info("Fitting preprocessors on training data")
            logger.info("=" * 70)
            logger.info("Generating TF-IDF features and labels from training data...")
            X_train, y_train, vectorizer, mlb = prepare_features_and_labels(
                data_train, vectorizer=None, mlb=None
            )
            logger.success(
                f"✓ Preprocessors fitted on training data: {X_train.shape[1]} features, {y_train.shape[1]} labels"
            )

            # Transform test data using fitted preprocessors
            logger.info("=" * 70)
            logger.info("Transforming test data using fitted preprocessors")
            logger.info("=" * 70)
            X_test, y_test, _, _ = prepare_features_and_labels(
                data_test, vectorizer=vectorizer, mlb=mlb
            )
            logger.success(
                f"✓ Test data transformed: {X_test.shape[0]} samples, {X_test.shape[1]} features, {y_test.shape[1]} labels"
            )

            # Save preprocessors for later use (e.g., prediction)
            logger.info("Saving fitted preprocessors...")
            save_preprocessors(vectorizer, mlb)
            logger.success("✓ Preprocessors saved")

            # Log preprocessing parameters to MLflow
            logger.debug("Logging preprocessing parameters to MLflow...")
            mlflow.log_param("preprocessing_max_features", vectorizer.max_features)
            mlflow.log_param("preprocessing_ngram_range", str(vectorizer.ngram_range))
            mlflow.log_param("preprocessing_stop_words", vectorizer.stop_words)
            mlflow.log_param("preprocessing_sublinear_tf", vectorizer.sublinear_tf)

            # Log training info to MLflow
            logger.debug("Logging training configuration to MLflow...")
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(mlb.classes_))
            mlflow.log_param("test_size_ratio", test_size)
            mlflow.log_param("random_state", random_state)

            # Train model
            logger.info("=" * 70)
            logger.info("Training multi-label classification model")
            logger.info("=" * 70)
            logger.info(
                f"Training model with hyperparameters: C={C}, penalty={penalty}, solver={solver}, max_iter={max_iter}"
            )
            model = train_model(X_train, y_train, model_params=model_params_dict)
            logger.success("✓ Model training completed successfully!")

            logger.info("=" * 70)
            logger.info("Saving trained model")
            logger.info("=" * 70)
            logger.info(f"Saving trained model to {final_model_path}...")
            save_model(model, final_model_path)
            logger.success(f"✓ Model saved successfully to {final_model_path}")

            # Save model to MLflow
            logger.info("Logging model artifact to MLflow...")
            mlflow.sklearn.log_model(model, "model")
            run_id = mlflow.active_run().info.run_id
            logger.success(f"✓ Model logged to MLflow run: {run_id}")

            logger.info("=" * 70)
            logger.success("🎉 Training pipeline completed successfully!")
            logger.info(f"Model saved: {final_model_path}")
            logger.info(f"MLflow run ID: {run_id}")
            logger.info("=" * 70)

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
