from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import typer

from descriptions.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR
from descriptions.dataset import load_interim, load_processed
from descriptions.modeling.model import get_model_name, load_model
from descriptions.modeling.preprocess import load_preprocessors
from descriptions.modeling.train import prepare_features_and_labels, split_data

app = typer.Typer()


def plot_confusion_matrix(
    y_true: Union[npt.NDArray[np.int64], pd.Series, List[int]],
    y_pred: Union[npt.NDArray[np.int64], pd.Series, List[int]],
    labels: Union[List[str], npt.NDArray[np.str_], pd.Index],
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        labels: List of label names for the confusion matrix
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Confusion matrix saved to {output_path}")


def plot_recall_precision(
    y_true: Union[npt.NDArray[np.int64], pd.Series, List[int]],
    y_pred: Union[npt.NDArray[np.int64], pd.Series, List[int]],
    labels: Union[List[str], npt.NDArray[np.str_], pd.Index],
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Plot recall and precision scores as a bar chart.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        labels: List of label names (used for context, not directly plotted)
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    plt.figure(figsize=figsize)
    metrics = ["Recall", "Precision", "F1 Score"]
    scores = [recall, precision, f1]
    sns.barplot(x=metrics, y=scores, palette="viridis")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Recall and Precision Scores")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Recall and precision plot saved to {output_path}")


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Union[str, Path],
    current_threshold: Optional[float] = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot threshold sensitivity curve showing how metrics change with threshold.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_proba: Prediction probabilities (n_samples, n_labels)
        output_path: Path where the figure will be saved
        current_threshold: Optional current threshold to mark on plot
        figsize: Figure size tuple (width, height) in inches
    """
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    logger.debug(f"Calculating metrics for {len(thresholds)} threshold values...")
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision_scores.append(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall_scores.append(recall_score(y_true, y_pred, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        thresholds, f1_scores, "o-", label="F1 Score", linewidth=2, markersize=6, color="#1f77b4"
    )
    ax.plot(
        thresholds,
        precision_scores,
        "s-",
        label="Precision",
        linewidth=2,
        markersize=6,
        color="#ff7f0e",
    )
    ax.plot(
        thresholds, recall_scores, "^-", label="Recall", linewidth=2, markersize=6, color="#2ca02c"
    )

    # Mark current threshold if provided
    if current_threshold is not None:
        ax.axvline(
            x=current_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Current Threshold ({current_threshold})",
            alpha=0.7,
        )
        # Add annotation
        ax.annotate(
            f"Threshold = {current_threshold}",
            xy=(current_threshold, 0.95),
            xytext=(current_threshold + 0.1, 0.95),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    ax.axvline(
        x=optimal_threshold,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Optimal F1 Threshold ({optimal_threshold:.2f})",
        alpha=0.7,
    )

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Sensitivity Curve", fontsize=14, fontweight="bold")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Threshold sensitivity plot saved to {output_path}")
    logger.info(f"  Optimal threshold (max F1): {optimal_threshold:.2f} (F1={optimal_f1:.3f})")


def plot_genre_cooccurrence(
    y_true: np.ndarray,
    mlb: object,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """
    Plot genre co-occurrence heatmap showing which genres appear together.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        mlb: MultiLabelBinarizer with genre classes
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    n_genres = len(mlb.classes_)
    cooccurrence = np.zeros((n_genres, n_genres))

    # Calculate co-occurrence matrix
    for i in range(n_genres):
        for j in range(n_genres):
            if i == j:
                # Diagonal: count of movies with this genre
                cooccurrence[i, j] = y_true[:, i].sum()
            else:
                # Off-diagonal: count of movies with both genres
                cooccurrence[i, j] = ((y_true[:, i] == 1) & (y_true[:, j] == 1)).sum()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=mlb.classes_,
        yticklabels=mlb.classes_,
        cbar_kws={"label": "Co-occurrence Count"},
        square=True,
    )
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Genre", fontsize=12)
    ax.set_title("Genre Co-occurrence Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Genre co-occurrence plot saved to {output_path}")


def plot_genre_frequency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mlb: object,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Plot genre frequency distribution comparing true vs predicted counts.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        mlb: MultiLabelBinarizer with genre classes
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(mlb.classes_))
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, true_counts, width, label="True", alpha=0.8, color="#2ca02c")
    bars2 = ax.bar(
        x_pos + width / 2, pred_counts, width, label="Predicted", alpha=0.8, color="#1f77b4"
    )

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Genre Frequency Distribution (True vs Predicted)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mlb.classes_, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Genre frequency plot saved to {output_path}")


def plot_prediction_count_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of number of genres per movie (true vs predicted).

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    true_counts = y_true.sum(axis=1)
    pred_counts = y_pred.sum(axis=1)

    max_count = max(true_counts.max(), pred_counts.max())
    bins = np.arange(0, max_count + 2) - 0.5

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(true_counts, bins=bins, alpha=0.7, label="True", color="#2ca02c", edgecolor="black")
    ax.hist(
        pred_counts, bins=bins, alpha=0.7, label="Predicted", color="#1f77b4", edgecolor="black"
    )

    ax.set_xlabel("Number of Genres per Movie", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Genre Counts per Movie", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(int(max_count) + 1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Prediction count distribution plot saved to {output_path}")


def plot_prediction_confidence(
    y_proba: np.ndarray,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of prediction confidence (probabilities).

    Args:
        y_proba: Prediction probabilities (n_samples, n_labels)
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    # Flatten all probabilities
    all_probas = y_proba.flatten()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(all_probas, bins=50, alpha=0.7, color="#9467bd", edgecolor="black")

    ax.set_xlabel("Prediction Probability", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Prediction Confidence", fontsize=14, fontweight="bold")
    ax.axvline(
        x=0.5, color="red", linestyle="--", linewidth=2, label="Default Threshold (0.5)", alpha=0.7
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add statistics text
    mean_prob = all_probas.mean()
    median_prob = np.median(all_probas)
    ax.text(
        0.05,
        0.95,
        f"Mean: {mean_prob:.3f}\nMedian: {median_prob:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Prediction confidence plot saved to {output_path}")


def plot_hamming_loss_per_genre(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mlb: object,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """
    Plot Hamming loss (error rate) per genre.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        mlb: MultiLabelBinarizer with genre classes
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    # Calculate per-genre Hamming loss (proportion of errors)
    hamming_per_genre = []
    for i in range(len(mlb.classes_)):
        errors = (y_true[:, i] != y_pred[:, i]).sum()
        total = len(y_true)
        hamming_per_genre.append(errors / total)

    hamming_per_genre = np.array(hamming_per_genre)

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(mlb.classes_))
    bars = ax.bar(x_pos, hamming_per_genre, alpha=0.8, color="#d62728")

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Hamming Loss (Error Rate)", fontsize=12)
    ax.set_title("Hamming Loss per Genre", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mlb.classes_, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Hamming loss per genre plot saved to {output_path}")


def plot_genre_confusion_patterns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mlb: object,
    output_path: Union[str, Path],
    figsize: tuple[int, int] = (14, 12),
) -> None:
    """
    Plot confusion patterns showing false positives and false negatives between genres.

    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        mlb: MultiLabelBinarizer with genre classes
        output_path: Path where the figure will be saved
        figsize: Figure size tuple (width, height) in inches
    """
    n_genres = len(mlb.classes_)
    confusion_matrix = np.zeros((n_genres, n_genres))

    # Calculate confusion: false positives (predicted but not true)
    # and false negatives (true but not predicted)
    for i in range(n_genres):
        for j in range(n_genres):
            if i == j:
                # Diagonal: correct predictions
                confusion_matrix[i, j] = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
            else:
                # Off-diagonal: predicted j when true is i (false positives for j)
                confusion_matrix[i, j] = (
                    (y_true[:, i] == 1) & (y_pred[:, j] == 1) & (y_pred[:, i] == 0)
                ).sum()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=mlb.classes_,
        yticklabels=mlb.classes_,
        cbar_kws={"label": "Count"},
        square=True,
    )
    ax.set_xlabel("Predicted Genre", fontsize=12)
    ax.set_ylabel("True Genre", fontsize=12)
    ax.set_title("Genre Confusion Patterns (False Positives)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Genre confusion patterns plot saved to {output_path}")


@app.command()
def main(
    model_path: Optional[str] = typer.Option(
        None,
        help="Path to the saved model file. If not provided, automatically finds a scikit-learn model.",
    ),
    data_path: Path = INTERIM_DATA_DIR / "merged_movies.csv",
    output_dir: Path = FIGURES_DIR,
    use_processed: bool = typer.Option(
        False,
        "--use-processed",
        help="If True, expects processed data (with TF-IDF features). If False, uses interim data and transforms with saved preprocessors.",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Probability threshold for predictions (default: 0.5). Lower values predict more labels.",
    ),
) -> None:
    """
    Generate visualization plots comparing model predictions vs true labels.

    This function loads a trained model, makes predictions on data, and generates
    visualization plots including confusion matrices and precision/recall metrics.

    The default data source is merged_movies.csv, which combines top_movies.csv
    with wiki_movie_plots_deduped.csv for enriched descriptions and genres.

    Args:
        model_path: Path to the saved model file. If None, automatically finds a scikit-learn model.
        data_path: Path to the data CSV file (interim or processed, depending on use_processed flag).
                  Defaults to merged_movies.csv (enriched with wiki data).
        output_dir: Directory where plots will be saved
        use_processed: If True, expects processed data. If False, uses interim data and transforms it.
        threshold: Probability threshold for predictions (default: 0.5)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Loading model and data")
    logger.info("=" * 70)

    # Load model
    if model_path is None:
        logger.debug("No model path provided, searching for default model...")
        model_files = list(MODELS_DIR.glob("*.joblib"))
        model_files = [
            f
            for f in model_files
            if f.name not in {"tfidf_vectorizer.joblib", "genre_binarizer.joblib"}
        ]
        if not model_files:
            raise FileNotFoundError(f"No model found in {MODELS_DIR}. Please train a model first.")
        model_path_obj = model_files[0]
        logger.info(f"Using default model: {model_path_obj.name}")
    else:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = MODELS_DIR / model_path_obj
        logger.info(f"Loading model from {model_path_obj}...")

    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found at {model_path_obj}")

    model = load_model(model_path_obj)
    logger.success(f"✓ Model loaded successfully: {model_path_obj.name}")

    # Load data (either processed or interim)
    if use_processed:
        logger.info(f"Loading processed data from {data_path}...")
        data = load_processed(data_path)
        logger.success(f"✓ Processed data loaded successfully: {len(data)} samples")
        logger.info("Splitting processed data into features (X) and labels (y)...")
        X, y_true, mlb = split_data(data)
    else:
        logger.info(f"Loading merged interim data from {data_path}...")
        data = load_interim(data_path)
        logger.success(f"✓ Merged interim data loaded successfully: {len(data)} samples")
        logger.info("Loading saved preprocessors...")
        vectorizer, mlb = load_preprocessors()
        logger.success("✓ Preprocessors loaded successfully")
        logger.info("Transforming data using saved preprocessors...")
        X, y_true, _, _ = prepare_features_and_labels(data, vectorizer=vectorizer, mlb=mlb)
        logger.success(
            f"✓ Data transformed: {X.shape[0]} samples, {X.shape[1]} features, {y_true.shape[1]} labels"
        )

    # Make predictions
    logger.info("=" * 70)
    logger.info("Making predictions")
    logger.info("=" * 70)
    logger.info("Generating prediction probabilities...")
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_proba = model.predict_proba(X_array)
    logger.info(f"Applying threshold={threshold} to generate binary predictions...")
    y_pred_binary = (y_proba >= threshold).astype(int)
    logger.success(f"✓ Predictions generated: {len(y_pred_binary)} samples")

    # Convert binary predictions to labels for plotting
    # For multi-label, we'll plot per-label metrics
    logger.info("=" * 70)
    logger.info("Generating plots")
    logger.info("=" * 70)

    model_name = get_model_name(model)

    # Generate per-label precision and recall plot
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None, zero_division=0
    )

    # Create precision/recall/F1 bar plot per genre
    logger.info("Creating precision/recall/F1 plot per genre...")
    fig, ax = plt.subplots(figsize=(14, 8))
    x_pos = np.arange(len(mlb.classes_))
    width = 0.25

    ax.bar(x_pos - width, precision, width, label="Precision", alpha=0.8, color="#ff7f0e")
    ax.bar(x_pos, recall, width, label="Recall", alpha=0.8, color="#2ca02c")
    ax.bar(x_pos + width, f1, width, label="F1 Score", alpha=0.8, color="#1f77b4")

    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Precision, Recall, and F1 Score by Genre - {model_name}", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mlb.classes_, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    precision_recall_path = output_dir / f"precision_recall_{model_name}.png"
    plt.savefig(precision_recall_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✓ Precision/recall/F1 plot saved to {precision_recall_path}")

    # Generate overall metrics plot
    logger.info("Creating overall metrics plot...")
    overall_f1 = f1_score(y_true, y_pred_binary, average="macro")
    overall_precision = precision_score(y_true, y_pred_binary, average="macro", zero_division=0)
    overall_recall = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)
    jaccard = jaccard_score(y_true, y_pred_binary, average="macro", zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 6))  # pyright: ignore[reportUnusedVariable]
    metrics = ["F1 Score", "Precision", "Recall", "Jaccard Score"]
    scores = [overall_f1, overall_precision, overall_recall, jaccard]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars = ax.bar(metrics, scores, color=colors, alpha=0.8)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Overall Model Performance Metrics - {model_name}", fontsize=14, fontweight="bold"
    )
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    metrics_path = output_dir / f"metrics_{model_name}.png"
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"✓ Metrics plot saved to {metrics_path}")

    # Generate threshold sensitivity curve
    logger.info("Creating threshold sensitivity curve...")
    threshold_sensitivity_path = output_dir / f"threshold_sensitivity_{model_name}.png"
    plot_threshold_sensitivity(
        y_true, y_proba, threshold_sensitivity_path, current_threshold=threshold
    )
    logger.success(f"✓ Threshold sensitivity plot saved to {threshold_sensitivity_path}")

    # Generate genre co-occurrence heatmap
    logger.info("Creating genre co-occurrence heatmap...")
    cooccurrence_path = output_dir / f"genre_cooccurrence_{model_name}.png"
    plot_genre_cooccurrence(y_true, mlb, cooccurrence_path)
    logger.success(f"✓ Genre co-occurrence plot saved to {cooccurrence_path}")

    # Generate genre frequency distribution
    logger.info("Creating genre frequency distribution...")
    frequency_path = output_dir / f"genre_frequency_{model_name}.png"
    plot_genre_frequency(y_true, y_pred_binary, mlb, frequency_path)
    logger.success(f"✓ Genre frequency plot saved to {frequency_path}")

    # Generate prediction count distribution
    logger.info("Creating prediction count distribution...")
    count_dist_path = output_dir / f"prediction_count_dist_{model_name}.png"
    plot_prediction_count_distribution(y_true, y_pred_binary, count_dist_path)
    logger.success(f"✓ Prediction count distribution plot saved to {count_dist_path}")

    # Generate prediction confidence distribution
    logger.info("Creating prediction confidence distribution...")
    confidence_path = output_dir / f"prediction_confidence_{model_name}.png"
    plot_prediction_confidence(y_proba, confidence_path)
    logger.success(f"✓ Prediction confidence plot saved to {confidence_path}")

    # Generate Hamming loss per genre
    logger.info("Creating Hamming loss per genre...")
    hamming_path = output_dir / f"hamming_loss_{model_name}.png"
    plot_hamming_loss_per_genre(y_true, y_pred_binary, mlb, hamming_path)
    logger.success(f"✓ Hamming loss plot saved to {hamming_path}")

    # Generate genre confusion patterns
    logger.info("Creating genre confusion patterns...")
    confusion_path = output_dir / f"genre_confusion_{model_name}.png"
    plot_genre_confusion_patterns(y_true, y_pred_binary, mlb, confusion_path)
    logger.success(f"✓ Genre confusion patterns plot saved to {confusion_path}")

    logger.info("=" * 70)
    logger.success("🎉 Plot generation completed successfully!")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info(f"  - Precision/Recall/F1: {precision_recall_path.name}")
    logger.info(f"  - Overall Metrics: {metrics_path.name}")
    logger.info(f"  - Threshold Sensitivity: {threshold_sensitivity_path.name}")
    logger.info(f"  - Genre Co-occurrence: {cooccurrence_path.name}")
    logger.info(f"  - Genre Frequency: {frequency_path.name}")
    logger.info(f"  - Prediction Count Distribution: {count_dist_path.name}")
    logger.info(f"  - Prediction Confidence: {confidence_path.name}")
    logger.info(f"  - Hamming Loss per Genre: {hamming_path.name}")
    logger.info(f"  - Genre Confusion Patterns: {confusion_path.name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
