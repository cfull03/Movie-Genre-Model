"""MLflow utilities for improved experiment tracking.

This module provides helper functions for:
- Git and environment tracking
- Artifact logging (preprocessors, plots)
- Model registry integration
- Run linking and organization
"""

from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

from loguru import logger
import mlflow
import mlflow.sklearn

from descriptions.config import PROJ_ROOT


def log_git_info() -> None:
    """Log git repository information to MLflow.

    Logs git commit hash, branch, and repository URL if available.
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping git info logging.")
        return

    try:
        # Get git commit hash
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )

        # Get git branch
        try:
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=PROJ_ROOT,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            branch = "unknown"

        # Get repository URL (if available)
        try:
            remote_url = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=PROJ_ROOT,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            remote_url = None

        # Log to MLflow
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.set_tag("git_branch", branch)
        if remote_url:
            mlflow.set_tag("git_repo", remote_url)

        logger.debug(f"Logged git info: commit={commit_hash[:8]}, branch={branch}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Could not retrieve git information: {e}")
        mlflow.set_tag("git_commit", "unknown")


def log_environment_info() -> None:
    """Log Python and package environment information to MLflow.

    Logs Python version and key package versions.
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping environment info logging.")
        return

    try:
        # Python version
        python_version = sys.version.split()[0]
        mlflow.log_param("python_version", python_version)

        # Key package versions
        key_packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "mlflow",
            "joblib",
        ]

        package_versions = {}
        for package in key_packages:
            try:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                package_versions[package] = version
                mlflow.log_param(f"pkg_{package}", version)
            except ImportError:
                mlflow.log_param(f"pkg_{package}", "not_installed")

        logger.debug(f"Logged environment info: Python {python_version}")
        logger.debug(f"Logged package versions: {list(package_versions.keys())}")

    except Exception as e:
        logger.warning(f"Could not log environment information: {e}")


def log_preprocessors_as_artifacts(
    vectorizer: Any,
    mlb: Any,
    normalizer: Any = None,
    feature_selector: Any = None,
    artifact_dir: str = "preprocessors",
) -> None:
    """Log preprocessors as MLflow artifacts.

    This allows preprocessors to be tracked with the model run,
    making it easier to reproduce predictions.

    Args:
        vectorizer: TfidfVectorizer to log
        mlb: MultiLabelBinarizer to log
        normalizer: Optional Normalizer to log
        feature_selector: Optional SelectKBest to log
        artifact_dir: Directory name for artifacts in MLflow
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping preprocessor logging.")
        return

    import tempfile

    import joblib

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifacts_dir = tmp_path / artifact_dir
            artifacts_dir.mkdir(exist_ok=True)

            # Save preprocessors to temporary directory
            logger.debug("Saving preprocessors to temporary directory...")
            joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")
            joblib.dump(mlb, artifacts_dir / "genre_binarizer.joblib")

            if normalizer is not None:
                joblib.dump(normalizer, artifacts_dir / "normalizer.joblib")

            if feature_selector is not None:
                joblib.dump(feature_selector, artifacts_dir / "feature_selector.joblib")

            # Log preprocessor metadata
            mlflow.log_param("preprocessing_max_features", vectorizer.max_features)
            mlflow.log_param("preprocessing_ngram_range", str(vectorizer.ngram_range))
            mlflow.log_param("preprocessing_stop_words", str(vectorizer.stop_words))
            mlflow.log_param("preprocessing_sublinear_tf", vectorizer.sublinear_tf)
            if feature_selector is not None:
                mlflow.log_param("preprocessing_feature_selector_k", feature_selector.k)

            # Log artifacts directory
            mlflow.log_artifacts(str(artifacts_dir), artifact_dir)
            logger.success(f"✓ Logged preprocessors as artifacts to '{artifact_dir}/'")

    except Exception as e:
        logger.error(f"Error logging preprocessors: {e}")
        raise


def log_figure(figure: Any, artifact_path: str, clear_figure: bool = True) -> None:
    """Log a matplotlib figure as an MLflow artifact.

    Args:
        figure: Matplotlib figure object
        artifact_path: Path where the figure will be saved in MLflow
        clear_figure: If True, clears the figure after logging to free memory
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping figure logging.")
        return

    try:
        mlflow.log_figure(figure, artifact_path)
        logger.debug(f"Logged figure to {artifact_path}")
        if clear_figure:
            import matplotlib.pyplot as plt

            plt.close(figure)
    except Exception as e:
        logger.warning(f"Could not log figure: {e}")


def log_metrics_dict(metrics: Dict[str, float], prefix: str = "") -> None:
    """Log a dictionary of metrics to MLflow.

    Args:
        metrics: Dictionary of metric names to values
        prefix: Optional prefix to add to metric names (e.g., "train_", "eval_")
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping metrics logging.")
        return

    for metric_name, value in metrics.items():
        full_name = f"{prefix}{metric_name}" if prefix else metric_name
        mlflow.log_metric(full_name, value)


def setup_experiment(experiment_name: str, create_if_not_exists: bool = True) -> str:
    """Set up an MLflow experiment and return experiment ID.

    Args:
        experiment_name: Name of the experiment
        create_if_not_exists: If True, creates experiment if it doesn't exist

    Returns:
        Experiment ID
    """
    try:
        if create_if_not_exists:
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"✓ Created new MLflow experiment: '{experiment_name}'")
                return experiment_id
            except Exception:
                # Experiment already exists
                pass

        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            logger.info(f"✓ Using existing MLflow experiment: '{experiment_name}'")
            return experiment.experiment_id
        else:
            # Fallback: create experiment
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✓ Created MLflow experiment: '{experiment_name}'")
            return experiment_id

    except Exception as e:
        logger.warning(f"Error setting up experiment: {e}")
        # Fallback to default experiment
        return "0"


def register_model(
    model_path: str,
    model_name: str,
    stage: str = "None",
    description: Optional[str] = None,
) -> str:
    """Register a model in MLflow Model Registry.

    Args:
        model_path: Path to the model artifact in MLflow (e.g., "model")
        model_name: Name for the registered model
        stage: Stage for the model (None, Staging, Production, Archived)
        description: Optional description for the model version

    Returns:
        Registered model version URI
    """
    try:
        # Register model
        model_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{model_path}",
            model_name,
        )

        if description:
            # Update description
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description,
            )

        # Transition to stage if specified
        if stage and stage != "None":
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage,
            )
            logger.success(
                f"✓ Registered model '{model_name}' version {model_version.version} to stage '{stage}'"
            )
        else:
            logger.success(f"✓ Registered model '{model_name}' version {model_version.version}")

        return model_version.version

    except Exception as e:
        logger.warning(f"Could not register model in Model Registry: {e}")
        logger.debug("This is optional - model is still logged to MLflow")
        return None


def log_data_info(data_path: Path, data_hash: Optional[str] = None) -> None:
    """Log dataset information to MLflow.

    Args:
        data_path: Path to the dataset file
        data_hash: Optional hash/checksum of the dataset for versioning
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping data info logging.")
        return

    mlflow.log_param("data_path", str(data_path))
    if data_path.exists():
        file_size = data_path.stat().st_size
        mlflow.log_param("data_size_bytes", file_size)
        mlflow.log_param("data_size_mb", round(file_size / (1024 * 1024), 2))

    if data_hash:
        mlflow.log_param("data_hash", data_hash)
        mlflow.set_tag("data_version", data_hash[:8])


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file for versioning.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hash as hex string
    """
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def log_training_summary(
    model_name: str,
    train_size: int,
    test_size: int,
    n_features: int,
    n_classes: int,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log a training summary with key information.

    Args:
        model_name: Name of the model
        train_size: Number of training samples
        test_size: Number of test samples
        n_features: Number of features
        n_classes: Number of classes/labels
        metrics: Optional dictionary of metrics to log
    """
    if not mlflow.active_run():
        return

    mlflow.set_tag("model_name", model_name)
    mlflow.log_param("train_size", train_size)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("n_classes", n_classes)

    if metrics:
        log_metrics_dict(metrics, prefix="train_")
