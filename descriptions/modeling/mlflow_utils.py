"""MLflow utilities for improved experiment tracking.

This module provides helper functions for:
- Git and environment tracking
- Artifact logging (preprocessors, plots)
- Model registry integration
- Run linking and organization
- Sklearn autolog (optional)
- Dataset logging (training/validation lineage)
- Model signature and input example
- Run context tags (user, source)
"""

from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd

from descriptions.config import PROJ_ROOT

# Optional: mlflow.data for dataset tracking (MLflow 2.x)
try:
    import mlflow.data

    _HAS_MLFLOW_DATA = True
except ImportError:
    _HAS_MLFLOW_DATA = False


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
    alias: Optional[str] = None,
) -> Optional[str]:
    """Register a model in MLflow Model Registry.

    Args:
        model_path: Path to the model artifact in MLflow (e.g., "model")
        model_name: Name for the registered model
        stage: Stage for the model (None, Staging, Production, Archived)
        description: Optional description for the model version
        alias: Optional alias to set (e.g. "champion", "production")

    Returns:
        Registered model version number, or None on error.
    """
    try:
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_path}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description,
            )

        if stage and stage != "None":
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

        if alias:
            try:
                client.set_registered_model_alias(model_name, alias, model_version.version)
                logger.success(f"✓ Set alias '{alias}' -> version {model_version.version}")
            except Exception as e:
                logger.warning(f"Could not set alias: {e}")

        return str(model_version.version)

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


def enable_sklearn_autolog(
    log_models: bool = False,
    log_datasets: bool = True,
    log_model_signatures: bool = True,
    log_post_training_metrics: bool = True,
    exclusive: bool = False,
    silent: bool = False,
) -> None:
    """Enable MLflow autologging for scikit-learn.

    Use log_models=False when you log the pipeline manually (e.g. with
    custom signature/input_example) to avoid duplicate model artifacts.
    Autolog will still capture estimator params and post-training metrics
    when you call sklearn.metrics.* or model.score().

    Args:
        log_models: If True, autolog logs the fitted estimator. Set False
            when logging the full pipeline manually.
        log_datasets: If True, log train/validation dataset info when applicable.
        log_model_signatures: If True, infer and log model signature (when log_models=True).
        log_post_training_metrics: If True, capture metrics from sklearn.metrics.* and model.score().
        exclusive: If True, autologged content is not logged to user-created fluent runs.
        silent: If True, suppress MLflow autolog warnings.
    """
    try:
        mlflow.sklearn.autolog(
            log_models=log_models,
            log_datasets=log_datasets,
            log_model_signatures=log_model_signatures,
            log_post_training_metrics=log_post_training_metrics,
            exclusive=exclusive,
            silent=silent,
        )
        logger.debug(
            "Sklearn autolog enabled (log_models=%s, log_datasets=%s)",
            log_models,
            log_datasets,
        )
    except Exception as e:
        logger.warning(f"Could not enable sklearn autolog: {e}")


def set_run_context_tags(
    source: str = "training",
    user: Optional[str] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> None:
    """Set standard run context tags for traceability.

    Args:
        source: Run type, e.g. 'training', 'evaluation', 'tuning'.
        user: Username (defaults to getpass.getuser() if available).
        extra_tags: Optional additional tags to set.
    """
    if not mlflow.active_run():
        return
    mlflow.set_tag("run_source", source)
    if user is None:
        try:
            import getpass

            user = getpass.getuser()
        except Exception:
            user = "unknown"
    if user:
        mlflow.set_tag("user", user)
    if extra_tags:
        for k, v in extra_tags.items():
            mlflow.set_tag(k, str(v))


def log_dataset(
    df: pd.DataFrame,
    context: str = "training",
    name: Optional[str] = None,
    source: Optional[Union[str, Path]] = None,
) -> None:
    """Log a dataset to the current run for lineage (MLflow 2.x data API).

    Args:
        df: DataFrame to log (training or validation data).
        context: How the dataset is used, e.g. 'training', 'validation', 'evaluation'.
        name: Optional dataset name.
        source: Optional path or URL describing dataset origin.
    """
    if not mlflow.active_run():
        return
    if not _HAS_MLFLOW_DATA:
        logger.debug("mlflow.data not available; skipping dataset logging")
        return
    try:
        source_str = str(source) if source else None
        dataset = mlflow.data.from_pandas(
            df,
            source=source_str,
            name=name or f"dataset_{context}",
        )
        mlflow.log_input(dataset, context=context)
        logger.debug(f"Logged dataset to MLflow (context={context}, rows={len(df)})")
    except Exception as e:
        logger.warning(f"Could not log dataset to MLflow: {e}")


def log_pipeline_model(
    pipeline: Any,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    input_example: Optional[Union[List[str], pd.DataFrame]] = None,
    signature: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    await_registration_for: int = 300,
) -> Optional[Any]:
    """Log a scikit-learn pipeline to MLflow with optional signature and registry.

    If input_example is provided and signature is None, the signature is inferred
    from the example and pipeline.predict(input_example).

    Args:
        pipeline: Fitted sklearn Pipeline to log.
        artifact_path: Artifact path for the model (default 'model').
        registered_model_name: If set, register the model under this name.
        input_example: Small sample of model input (e.g. list of strings for text).
        signature: MLflow ModelSignature; inferred from input_example if not provided.
        metadata: Optional dict attached to the model.
        await_registration_for: Seconds to wait for model version READY (0 to skip).

    Returns:
        ModelInfo from log_model, or None on error.
    """
    if not mlflow.active_run():
        logger.warning("No active MLflow run. Skipping model logging.")
        return None

    input_example_for_log = None
    if input_example is not None:
        if isinstance(input_example, pd.DataFrame):
            input_example_for_log = input_example.head(5)
        elif isinstance(input_example, list) and input_example:
            input_example_for_log = pd.DataFrame({"description": input_example[:5]})

    if signature is None and input_example_for_log is not None:
        try:
            from mlflow.models import infer_signature

            text_list = (
                input_example_for_log.iloc[:, 0].tolist()
                if isinstance(input_example_for_log, pd.DataFrame)
                else list(input_example_for_log)[:5]
            )
            preds = pipeline.predict(text_list)
            signature = infer_signature(input_example_for_log, preds)
            logger.debug("Inferred model signature from input example")
        except Exception as e:
            logger.warning(f"Could not infer model signature: {e}")
            signature = None

    log_kw: Dict[str, Any] = {
        "signature": signature,
        "input_example": input_example_for_log,
    }
    if metadata:
        log_kw["metadata"] = metadata
    if registered_model_name:
        log_kw["registered_model_name"] = registered_model_name
        log_kw["await_registration_for"] = await_registration_for

    try:
        return mlflow.sklearn.log_model(pipeline, artifact_path, **log_kw)
    except TypeError:
        log_kw.pop("await_registration_for", None)
        return mlflow.sklearn.log_model(pipeline, artifact_path, **log_kw)


def search_runs(
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[List[str]] = None,
    max_results: int = 100,
) -> List[Any]:
    """Search MLflow runs for an experiment.

    Args:
        experiment_name: Experiment name (used if experiment_id not set).
        experiment_id: Experiment ID.
        filter_string: MLflow filter string (e.g. "metrics.f1 > 0.7").
        order_by: List of order clauses (e.g. ["metrics.eval_f1 DESC"]).
        max_results: Maximum number of runs to return.

    Returns:
        List of Run objects.
    """
    client = mlflow.tracking.MlflowClient()
    if experiment_id is None and experiment_name:
        exp = client.get_experiment_by_name(experiment_name)
        if exp:
            experiment_id = exp.experiment_id
    if experiment_id is None:
        return []
    return client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=order_by or [],
        max_results=max_results,
    )
