import json
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function for converting scores to probabilities
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import typer

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim
from descriptions.modeling.mlflow_utils import (
    calculate_file_hash,
    enable_sklearn_autolog,
    log_data_info,
    log_dataset,
    log_environment_info,
    log_git_info,
    log_pipeline_model,
    log_preprocessors_as_artifacts,
    log_training_summary,
    register_model,
    set_run_context_tags,
    setup_experiment,
)
from descriptions.modeling.model import build_model, build_pipeline, get_model_name, save_model
from descriptions.modeling.preprocess import (
    _generate_descriptions,
    _generate_targets,
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
            from descriptions.modeling.model import load_model

            mlb = load_model("genre_binarizer")
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
    normalizer=None,
    feature_selector=None,
    k_features: int = 4500,
) -> Tuple[pd.DataFrame, np.ndarray, object, object, object, object]:
    """
    Generate TF-IDF features and multi-label targets from raw data.

    If vectorizer and mlb are None, they will be fitted on the data.
    If provided, they will be used to transform the data.

    Args:
        data: DataFrame with 'description' and 'genre' columns
        vectorizer: Optional pre-fitted TfidfVectorizer. If None, creates and fits a new one.
        mlb: Optional pre-fitted MultiLabelBinarizer. If None, creates and fits a new one.
        normalizer: Optional pre-fitted Normalizer. If None, creates and fits a new one.
        feature_selector: Optional pre-fitted SelectKBest. If None, creates and fits a new one.
        k_features: Number of features to select (default: 4500)

    Returns:
        Tuple of (features_df, labels_array, vectorizer, mlb, normalizer, feature_selector)
    """
    logger.info("Generating multi-label genre targets...")
    y, mlb, data_filtered = _generate_targets(data, mlb=mlb)

    logger.info("Generating TF-IDF features from descriptions...")
    X_sparse, vectorizer = _generate_descriptions(data_filtered, vectorizer=vectorizer)

    # Apply L2 normalization
    if normalizer is None:
        logger.info("Applying L2 normalization to feature vectors...")
        normalizer = Normalizer(norm="l2")
        X_sparse = normalizer.fit_transform(X_sparse)
        logger.success("Normalization complete: features normalized to unit length (L2 norm)")
    else:
        logger.info("Using pre-fitted normalizer for transformation")
        X_sparse = normalizer.transform(X_sparse)
        logger.debug("Features normalized")

    # Apply feature selection
    if feature_selector is None:
        logger.info(f"Applying feature selection with SelectKBest (k={k_features})...")
        feature_selector = SelectKBest(score_func=chi2, k=k_features)
        X_sparse = feature_selector.fit_transform(X_sparse, y)
        logger.success(
            f"Feature selection complete: {X_sparse.shape[1]} features selected "
            f"(from {vectorizer.max_features} original features)"
        )
    else:
        logger.info("Using pre-fitted feature selector for transformation")
        X_sparse = feature_selector.transform(X_sparse)
        logger.info(f"Features transformed: {X_sparse.shape[1]} features")

    # Convert sparse matrix to dense DataFrame
    logger.debug("Converting sparse TF-IDF matrix to dense DataFrame...")
    X_df = pd.DataFrame(
        X_sparse.toarray(),
        index=data_filtered.index,
        columns=[f"tfidf_{i}" for i in range(X_sparse.shape[1])],
    )

    logger.info(
        f"Features and labels prepared: {X_df.shape[0]} samples, "
        f"{X_df.shape[1]} features, {y.shape[1]} labels"
    )

    return X_df, y, vectorizer, mlb, normalizer, feature_selector


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
        model_params: Optional dictionary of model hyperparameters (C, penalty, loss, max_iter, tol, class_weight, dual)

    Returns:
        Trained OneVsRestClassifier model
    """
    n_samples, n_features = X_train.shape[0], X_train.shape[1]
    n_labels = y_train.shape[1]
    logger.info(f"Training model: {n_samples} samples, {n_features} features, {n_labels} labels")

    with tqdm(total=3, desc="Training model", unit="step") as pbar:
        # Build model (no vectorizer needed since features are already TF-IDF transformed)
        pbar.set_description("Building OneVsRestClassifier with LinearSVC")
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


def save_parameters(
    model_params_dict: dict,
    model_name: str,
    params_path: Path = MODELS_DIR / "model_parameters.json",
) -> None:
    """
    Save model parameters to a JSON file.
    """
    with open(MODELS_DIR / f"{model_name}_parameters.json", "w") as f:
        json.dump(model_params_dict, f)
    logger.success(
        f"✓ Parameters saved successfully to {MODELS_DIR / f'{model_name}_parameters.json'}"
    )


@app.command()
def main(
    interim_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    force: bool = False,
    experiment_name: str = "movie-genre-classification",
    run_name: Optional[str] = None,
    # Model hyperparameters (best parameters from cross-validation)
    C: float = typer.Option(
        0.1,
        "--C",
        help="Regularization strength for LinearSVC (larger = less regularization)",
    ),
    penalty: str = typer.Option(
        "l2", "--penalty", help="Penalty type for LinearSVC ('l1' or 'l2')"
    ),
    loss: str = typer.Option(
        "squared_hinge", "--loss", help="Loss function for LinearSVC ('hinge' or 'squared_hinge')"
    ),
    max_iter: int = typer.Option(1000, "--max-iter", help="Max iterations for LinearSVC"),
    tol: float = typer.Option(1e-3, "--tol", help="Tolerance for LinearSVC"),
    class_weight: str = typer.Option("balanced", "--class-weight", help="Class weight strategy"),
    dual: bool = typer.Option(
        False, "--dual/--no-dual", help="Solve dual or primal optimization problem"
    ),
    k_features: int = typer.Option(
        4500, "--k-features", help="Number of features to select with SelectKBest"
    ),
    register_to_registry: bool = typer.Option(
        False,
        "--register-model",
        help="Register the trained model in MLflow Model Registry.",
    ),
    registered_model_name: Optional[str] = typer.Option(
        None,
        "--registered-model-name",
        help="Name for the registered model (default: movie-genre-classifier).",
    ),
    model_stage: str = typer.Option(
        "None",
        "--model-stage",
        help="Stage to assign when registering (None, Staging, Production, Archived).",
    ),
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
        C: Regularization strength for LinearSVC (default: 0.1). Larger values = less regularization.
        penalty: Type of regularization penalty - 'l1' or 'l2' (default: 'l2')
        loss: Loss function - 'hinge' or 'squared_hinge' (default: 'squared_hinge')
        max_iter: Maximum number of iterations for convergence (default: 1000)
        tol: Tolerance for stopping criteria (default: 1e-3)
        class_weight: Class weight strategy - 'balanced' or None (default: 'balanced')
        dual: Whether to solve dual or primal optimization problem (default: False)
        k_features: Number of features to select with SelectKBest (default: 4500)
        register_to_registry: If True, register the model in MLflow Model Registry
        registered_model_name: Name for the registered model (default: movie-genre-classifier)
        model_stage: Stage when registering (None, Staging, Production, Archived)
    """
    # Set up MLflow experiment
    logger.info("=" * 70)
    logger.info("Setting up MLflow experiment tracking")
    logger.info("=" * 70)
    setup_experiment(experiment_name, create_if_not_exists=True)
    enable_sklearn_autolog(log_models=False, log_datasets=True)

    # Generate run name from hyperparameters if not provided
    if run_name is None:
        run_name = f"LinearSVC_C{C}_penalty-{penalty}_loss-{loss}_k{k_features}"
    logger.info(f"MLflow run name: '{run_name}'")

    # Start MLflow run - each unique parameter combination creates a new run
    with mlflow.start_run(run_name=run_name):
        try:
            set_run_context_tags(source="training")
            # Log git and environment info
            logger.info("Logging git and environment information...")
            log_git_info()
            log_environment_info()
            logger.success("✓ Git and environment info logged")
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
                "loss": loss,
                "max_iter": max_iter,
                "tol": tol,
                "class_weight": class_weight,
                "dual": dual,
                "random_state": random_state,
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
                save_parameters(model_params_dict, model_name)
                logger.debug(f"  model_{key} = {value}")
            mlflow.set_tag("model_type", "LinearSVC-OneVsRest")
            mlflow.set_tag("task", "multi-label-classification")
            mlflow.log_param("k_features", k_features)
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

            # Log data info to MLflow with hash for versioning
            logger.debug("Logging data information to MLflow...")
            data_hash = calculate_file_hash(interim_path) if interim_path.exists() else None
            log_data_info(interim_path, data_hash=data_hash)
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
            log_dataset(
                data_train,
                context="training",
                source=str(interim_path),
                name="train",
            )
            log_dataset(
                data_test,
                context="validation",
                source=str(interim_path),
                name="validation",
            )

            # Fit MLB on TRAINING data only (for label encoding/decoding)
            logger.info("=" * 70)
            logger.info("Fitting MultiLabelBinarizer on training data")
            logger.info("=" * 70)
            # Use _generate_targets to get mlb and filtered data (same row order as y_train)
            from descriptions.modeling.preprocess import _generate_targets
            y_train, mlb, data_train_filtered = _generate_targets(data_train, mlb=None)
            X_train_text = data_train_filtered["description"].fillna("").astype(str).tolist()
            logger.success(
                f"✓ MultiLabelBinarizer fitted: {y_train.shape[1]} labels, {len(X_train_text)} training samples"
            )

            # Build pipeline with all preprocessing steps + model
            logger.info("=" * 70)
            logger.info("Building prediction pipeline")
            logger.info("=" * 70)
            logger.info(
                f"Building pipeline with hyperparameters: "
                f"C={C}, penalty={penalty}, loss={loss}, k_features={k_features}"
            )
            pipeline = build_pipeline(
                model_params=model_params_dict,
                k_features=k_features,
            )
            logger.success("✓ Pipeline built successfully")

            # Fit pipeline on training data (text descriptions → binary labels)
            logger.info("=" * 70)
            logger.info("Training pipeline on training data")
            logger.info("=" * 70)
            logger.info("Fitting pipeline: TF-IDF → Normalizer → Feature Selection → Classifier...")
            pipeline.fit(X_train_text, y_train)
            logger.success("✓ Pipeline training completed successfully!")

            # Prepare test data for evaluation
            logger.info("=" * 70)
            logger.info("Preparing test data for evaluation")
            logger.info("=" * 70)
            X_test_text = data_test["description"].fillna("").astype(str).tolist()
            # Use mlb to transform test labels
            from descriptions.modeling.preprocess import _generate_targets
            y_test, _, _ = _generate_targets(data_test, mlb=mlb)
            logger.success(
                f"✓ Test data prepared: {len(X_test_text)} samples, {y_test.shape[1]} labels"
            )

            # Extract transformed features for metrics calculation (for backward compatibility)
            X_train_transformed = pipeline.named_steps['feature_selector'].transform(
                pipeline.named_steps['normalizer'].transform(
                    pipeline.named_steps['tfidf'].transform(X_train_text)
                )
            )
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()

            # Save pipeline and mlb only (pipeline contains tfidf, normalizer, feature_selector, classifier)
            logger.info("=" * 70)
            logger.info("Saving trained pipeline and label encoder")
            logger.info("=" * 70)
            save_model(pipeline, final_model_path)
            logger.success(f"✓ Pipeline saved to {final_model_path}")

            save_model(mlb, "genre_binarizer")
            logger.success("✓ MultiLabelBinarizer saved (for label encoding/decoding)")

            # Log preprocessors as MLflow artifacts (for tracking only; extracted from pipeline)
            logger.info("Logging preprocessors as MLflow artifacts...")
            vectorizer = pipeline.named_steps["tfidf"]
            normalizer = pipeline.named_steps["normalizer"]
            feature_selector = pipeline.named_steps["feature_selector"]
            log_preprocessors_as_artifacts(
                vectorizer, mlb, normalizer, feature_selector, artifact_dir="preprocessors"
            )
            mlflow.log_param("preprocessing_feature_selector_score_func", "chi2")

            # Calculate training metrics using pipeline
            logger.info("Calculating training metrics...")

            try:
                model = pipeline.named_steps['classifier']
                y_train_scores = model.decision_function(X_train_transformed)
                y_train_proba = expit(y_train_scores)
                y_train_pred = (y_train_proba >= 0.55).astype(int)

                train_metrics = {
                    "hamming_loss": float(hamming_loss(y_train, y_train_pred)),
                    "f1": float(f1_score(y_train, y_train_pred, average="micro", zero_division=0)),
                    "precision": float(
                        precision_score(y_train, y_train_pred, average="micro", zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_train, y_train_pred, average="micro", zero_division=0)
                    ),
                    "jaccard": float(
                        jaccard_score(y_train, y_train_pred, average="micro", zero_division=0)
                    ),
                }
                logger.success("✓ Training metrics calculated")

                # Save metrics to models directory
                metrics_path = MODELS_DIR / f"metrics_{model_name}.json"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump(train_metrics, f, indent=2)
                logger.success(f"✓ Metrics saved to {metrics_path}")
            except Exception as e:
                logger.warning(f"Could not calculate training metrics: {e}")
                train_metrics = None

            # Log training summary with metrics
            logger.debug("Logging training summary to MLflow...")
            log_training_summary(
                model_name=model_name,
                train_size=len(X_train_text),
                test_size=len(X_test_text),
                n_features=X_train_transformed.shape[1],
                n_classes=len(mlb.classes_),
                metrics=train_metrics,
            )
            mlflow.log_param("test_size_ratio", test_size)
            mlflow.log_param("random_state", random_state)

            # Model is already saved as part of pipeline above
            # Just log to MLflow here

            logger.info("=" * 70)
            logger.info("Saving Parameters")
            logger.info("=" * 70)
            params_file = MODELS_DIR / f"{model_name}_parameters.json"
            logger.info(f"Saved Parameters to {params_file}...")
            save_parameters(model_params_dict, model_name)
            logger.success(f"✓ Parameters saved successfully to {params_file}")

            # Save pipeline to MLflow with signature and input example
            logger.info("Logging pipeline artifact to MLflow (with signature and input example)...")
            log_pipeline_model(
                pipeline,
                artifact_path="model",
                input_example=X_train_text[:5],
                metadata={"model_type": "LinearSVC-OneVsRest", "task": "multi-label-classification"},
            )
            run_id = mlflow.active_run().info.run_id
            logger.success(f"✓ Pipeline logged to MLflow run: {run_id}")

            if register_to_registry:
                reg_name = registered_model_name or "movie-genre-classifier"
                register_model(
                    model_path="model",
                    model_name=reg_name,
                    stage=model_stage,
                    description=f"LinearSVC C={C}, penalty={penalty}, loss={loss}, k_features={k_features}",
                )

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
