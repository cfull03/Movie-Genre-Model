from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function for converting scores to probabilities
import typer

from descriptions.config import MODELS_DIR
from descriptions.modeling.model import load_model
from descriptions.modeling.preprocess import load_preprocessors

app = typer.Typer()


def predict_genres(
    descriptions: List[str],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.55,
    top_k: int = 3,
) -> List[List[str]]:
    """
    Predict genres for a list of movie descriptions.

    Args:
        descriptions: List of movie description strings
        model_path: Path to trained model. If None, uses default model.
        threshold: Probability threshold for predictions (default: 0.55).
                   Only genres above this threshold will be included.
        top_k: Maximum number of top genres to select (default: 3).
               The top k genres by probability will be selected, but only
               those above the threshold will be returned.

    Returns:
        List of lists of predicted genres (one list per description)
    """
    logger.info(f"Preparing to predict genres for {len(descriptions)} description(s)...")

    # Load model
    if model_path is None:
        logger.debug("No model path provided, searching for default model...")
        model_files = list(MODELS_DIR.glob("*.joblib"))
        model_files = [
            f
            for f in model_files
            if f.name
            not in {
                "tfidf_vectorizer.joblib",
                "genre_binarizer.joblib",
                "normalizer.joblib",
                "feature_selector.joblib",
            }
        ]
        if not model_files:
            raise FileNotFoundError(f"No model found in {MODELS_DIR}. Please train a model first.")
        model_path = model_files[0]
        logger.info(f"Using default model: {model_path.name}")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = MODELS_DIR / model_path
        logger.info(f"Loading model from {model_path}...")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = load_model(model_path)
    logger.success(f"✓ Model loaded successfully: {model_path.name}")

    # Load preprocessors
    logger.info(
        "Loading preprocessors (TfidfVectorizer, MultiLabelBinarizer, Normalizer, and Feature Selector)..."
    )
    vectorizer, mlb, normalizer, feature_selector = load_preprocessors()
    logger.success("✓ Preprocessors loaded successfully")

    # Transform descriptions to TF-IDF features
    logger.info(f"Transforming {len(descriptions)} descriptions to TF-IDF features...")
    X = vectorizer.transform(descriptions)
    logger.debug(f"TF-IDF features generated: shape {X.shape}")

    # Apply L2 normalization (same as training)
    logger.info("Applying L2 normalization...")
    X = normalizer.transform(X)
    logger.debug(f"Features normalized: shape {X.shape}")

    # Apply feature selection (same as training)
    logger.info("Applying feature selection...")
    X = feature_selector.transform(X)
    logger.debug(f"Features after selection: shape {X.shape}")

    # Get prediction probabilities
    # LinearSVC doesn't have predict_proba, so we use decision_function
    # and convert scores to probabilities using sigmoid function
    logger.info("Generating prediction scores...")
    y_scores = model.decision_function(X)
    logger.debug(f"Decision scores generated: shape {y_scores.shape}")

    # Convert scores to probabilities using sigmoid function
    # sigmoid(x) = 1 / (1 + exp(-x))
    logger.info("Converting scores to probabilities...")
    y_proba = expit(y_scores)
    logger.debug(f"Probabilities generated: shape {y_proba.shape}")

    # Select top-k genres per sample, but only include those above threshold
    logger.info(f"Selecting top {top_k} genres with probability >= {threshold}...")
    y_pred_binary = np.zeros_like(y_proba, dtype=int)

    for i in range(y_proba.shape[0]):
        # Get top-k indices for this sample (sorted by probability descending)
        top_k_indices = np.argsort(y_proba[i])[-top_k:][::-1]

        # Only include genres that are above threshold
        for idx in top_k_indices:
            if y_proba[i, idx] >= threshold:
                y_pred_binary[i, idx] = 1

    logger.debug(f"Binary predictions generated: shape {y_pred_binary.shape}")
    logger.debug(f"Average genres per sample: {y_pred_binary.sum(axis=1).mean():.2f}")

    # Decode predictions back to genre labels
    logger.info("Decoding predictions to genre labels...")
    predicted_genres = mlb.inverse_transform(y_pred_binary)

    # Log prediction statistics
    n_predictions = sum(len(genres) for genres in predicted_genres)
    avg_genres = n_predictions / len(predicted_genres) if predicted_genres else 0
    logger.success(
        f"✓ Predictions complete: {n_predictions} total genre predictions "
        f"({avg_genres:.2f} genres per movie on average)"
    )

    return predicted_genres


@app.command()
def main(
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Single movie description to predict genres for",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file",
        "-i",
        help="Path to CSV file with movie descriptions. Must have 'description' column.",
    ),
    model_path: Optional[Union[str, Path]] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to trained model file. If not provided, uses default from models/",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        "-o",
        help="Path to save predictions CSV. If not provided, prints to console.",
    ),
    description_column: str = typer.Option(
        "description",
        "--description-column",
        "-c",
        help="Column name in input file containing descriptions (default: 'description')",
    ),
    threshold: float = typer.Option(
        0.55,
        "--threshold",
        "-t",
        help="Probability threshold for predictions (default: 0.55). Only genres above this threshold will be included.",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Maximum number of top genres to select (default: 3). The top k genres by probability will be selected, but only those above the threshold will be returned.",
    ),
) -> None:
    """
    Predict movie genres from descriptions.

    Can accept either:
    - A single description via --description flag
    - A CSV file with descriptions via --input-file flag

    Examples:
        # Predict from single description
        python -m descriptions.modeling.predict --description "A thrilling action movie"

        # Predict from CSV file
        python -m descriptions.modeling.predict --input-file data/test_movies.csv

        # Save predictions to file
        python -m descriptions.modeling.predict --input-file data/test_movies.csv --output-file predictions.csv
    """
    logger.info("=" * 70)
    logger.info("Movie Genre Prediction")
    logger.info("=" * 70)

    descriptions: List[str] = []
    input_data: Optional[pd.DataFrame] = None

    # Get descriptions from either command line or file
    if description:
        descriptions = [description]
        logger.info("Using description from command line argument")
    elif input_file:
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            raise typer.Exit(1)

        logger.info(f"Loading descriptions from {input_file}...")
        try:
            input_data = pd.read_csv(input_file)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise typer.Exit(1)

        if description_column not in input_data.columns:
            logger.error(
                f"Column '{description_column}' not found in input file. "
                f"Available columns: {list(input_data.columns)}"
            )
            raise typer.Exit(1)

        descriptions = input_data[description_column].fillna("").astype(str).tolist()
        logger.success(f"✓ Loaded {len(descriptions)} description(s) from file")
    else:
        logger.error("Please provide either --description or --input-file")
        raise typer.Exit(1)

    if not descriptions:
        logger.error("No descriptions provided")
        raise typer.Exit(1)

    # Filter out empty descriptions
    descriptions = [desc.strip() for desc in descriptions if desc.strip()]
    if not descriptions:
        logger.error("No valid (non-empty) descriptions found")
        raise typer.Exit(1)

    logger.info(f"Processing {len(descriptions)} valid description(s)...")

    # Make predictions
    try:
        logger.info("=" * 70)
        logger.info("Generating predictions")
        logger.info("=" * 70)
        predicted_genres = predict_genres(
            descriptions, model_path=model_path, threshold=threshold, top_k=top_k
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise typer.Exit(1)

    # Format results
    logger.info("=" * 70)
    logger.info("Formatting results")
    logger.info("=" * 70)
    results = []
    for desc, genres in zip(descriptions, predicted_genres):
        genre_str = ", ".join(genres) if genres else "None"
        results.append(
            {
                "description": desc[:100] + "..." if len(desc) > 100 else desc,
                "predicted_genres": genre_str,
                "genre_count": len(genres),
            }
        )

    # Output results
    results_df = pd.DataFrame(results)

    if output_file:
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.success(f"✓ Predictions saved to {output_file}")

        # If input data was provided, create merged output
        if input_data is not None:
            merged_df = input_data.copy()
            merged_df["predicted_genres"] = [
                ", ".join(g) if g else "None" for g in predicted_genres
            ]
            merged_df["genre_count"] = [len(g) for g in predicted_genres]

            merged_output = output_file.parent / f"{output_file.stem}_merged{output_file.suffix}"
            merged_df.to_csv(merged_output, index=False)
            logger.success(f"✓ Merged predictions with input data saved to {merged_output}")
    else:
        # Print to console
        logger.info("")
        logger.info("=" * 70)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 70)
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Description: {result['description']}")
            logger.info(f"   Predicted Genres: {result['predicted_genres']}")
            logger.info(f"   Number of Genres: {result['genre_count']}")
        logger.info("")
        logger.info("=" * 70)

    logger.success("🎉 Prediction pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
