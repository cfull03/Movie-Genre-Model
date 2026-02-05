from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function for converting scores to probabilities
import typer

from descriptions.config import MODELS_DIR
from descriptions.modeling.evaluate import load_per_label_thresholds
from descriptions.modeling.model import load_model
from sklearn.pipeline import Pipeline

app = typer.Typer()


def predict_genres(
    descriptions: List[str],
    model_path: Optional[Union[str, Path]] = None,
    threshold: Optional[Union[float, Dict[str, float]]] = None,
    top_k: int = 3,
    use_global_threshold: bool = False,
) -> List[List[str]]:
    """
    Predict genres for a list of movie descriptions.

    Args:
        descriptions: List of movie description strings
        model_path: Path to trained model. If None, uses default model.
        threshold: Probability threshold for predictions. Can be:
                   - None: Uses per-label thresholds if available, else 0.55
                   - float: Global threshold value (overrides per-label thresholds)
                   - Dict[str, float]: Per-label thresholds (genre name -> threshold)
        top_k: Maximum number of top genres to select (default: 3).
               The top k genres by probability will be selected, but only
               those above their threshold will be returned.
        use_global_threshold: If True, forces use of global threshold even if
                             per-label thresholds are available.

    Returns:
        List of lists of predicted genres (one list per description)
    """
    logger.info(f"Preparing to predict genres for {len(descriptions)} description(s)...")

    # Load model
    if model_path is None:
        logger.debug("No model path provided, searching for default model...")
        model_files = list(MODELS_DIR.glob("*.joblib"))
        model_files = [
            f for f in model_files if f.name != "genre_binarizer.joblib"
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

    # Try to load pipeline (new approach) or fall back to individual components (backward compatibility)
    loaded_model = load_model(model_path)
    logger.success(f"✓ Model/pipeline loaded successfully: {model_path.name}")

    # Loaded model must be a Pipeline (tfidf → normalizer → feature_selector → classifier)
    if isinstance(loaded_model, Pipeline) or (
        hasattr(loaded_model, "named_steps")
        and "classifier" in getattr(loaded_model, "named_steps", {})
    ):
        logger.info("Using Pipeline model")
        pipeline = loaded_model
        logger.info("Loading MultiLabelBinarizer for label encoding/decoding...")
        mlb = load_model("genre_binarizer")
        logger.success("✓ MultiLabelBinarizer loaded successfully")
        
        # Use pipeline to get decision scores (handles all preprocessing internally)
        logger.info(f"Transforming {len(descriptions)} descriptions using pipeline...")
        model = pipeline.named_steps['classifier']
        # Get preprocessed features for decision_function
        X_preprocessed = pipeline.named_steps['feature_selector'].transform(
            pipeline.named_steps['normalizer'].transform(
                pipeline.named_steps['tfidf'].transform(descriptions)
            )
        )
        if hasattr(X_preprocessed, 'toarray'):
            X_preprocessed = X_preprocessed.toarray()
        logger.debug(f"Features preprocessed: shape {X_preprocessed.shape}")
        
        logger.info("Generating prediction scores...")
        y_scores = model.decision_function(X_preprocessed)
        logger.debug(f"Decision scores generated: shape {y_scores.shape}")
    else:
        raise ValueError(
            "Loaded model must be a Pipeline (tfidf → normalizer → feature_selector → classifier). "
            "Please retrain the model to generate the correct format."
        )

    # Convert scores to probabilities using sigmoid function
    # sigmoid(x) = 1 / (1 + exp(-x))
    logger.info("Converting scores to probabilities...")
    y_proba = expit(y_scores)
    logger.debug(f"Probabilities generated: shape {y_proba.shape}")

    # Determine which thresholds to use
    per_label_thresholds: Optional[Dict[str, float]] = None
    global_threshold = 0.55

    if use_global_threshold:
        # Force use of global threshold
        if threshold is None or isinstance(threshold, dict):
            global_threshold = 0.55 if threshold is None else 0.55
            logger.info(f"Using global threshold: {global_threshold}")
        else:
            global_threshold = threshold
            logger.info(f"Using global threshold: {global_threshold}")
    elif isinstance(threshold, dict):
        # Explicit per-label thresholds provided
        per_label_thresholds = threshold
        logger.info(f"Using provided per-label thresholds for {len(per_label_thresholds)} labels")
    elif threshold is not None:
        # Explicit global threshold provided
        global_threshold = threshold
        logger.info(f"Using global threshold: {global_threshold}")
    else:
        # Try to load per-label thresholds by default
        per_label_thresholds = load_per_label_thresholds()
        if per_label_thresholds:
            logger.info(f"Using per-label thresholds for {len(per_label_thresholds)} labels")
        else:
            global_threshold = 0.55
            logger.info(
                f"Per-label thresholds not found, using global threshold: {global_threshold}"
            )

    # Apply thresholds and select top-k genres per sample
    y_pred_binary = np.zeros_like(y_proba, dtype=int)

    if per_label_thresholds:
        # Use per-label thresholds
        logger.info("Applying per-label thresholds...")
        for label_idx, label_name in enumerate(mlb.classes_):
            if label_name in per_label_thresholds:
                label_threshold = per_label_thresholds[label_name]
                y_pred_binary[:, label_idx] = (y_proba[:, label_idx] >= label_threshold).astype(
                    int
                )
            else:
                logger.debug(
                    f"Threshold not found for label '{label_name}', using global threshold {global_threshold}"
                )
                y_pred_binary[:, label_idx] = (y_proba[:, label_idx] >= global_threshold).astype(
                    int
                )
        logger.debug("Binary predictions generated using per-label thresholds")
    else:
        # Use global threshold
        logger.info(f"Applying global threshold: {global_threshold}")
        y_pred_binary = (y_proba >= global_threshold).astype(int)
        logger.debug(f"Binary predictions generated using global threshold {global_threshold}")

    # Apply top-k selection: for each sample, select top-k from genres that passed threshold
    logger.info(f"Selecting top {top_k} genres from thresholded predictions...")
    for i in range(y_proba.shape[0]):
        # Get indices of genres that passed threshold
        passed_indices = np.where(y_pred_binary[i] == 1)[0]

        if len(passed_indices) > top_k:
            # Select top-k by probability from those that passed
            passed_proba = y_proba[i, passed_indices]
            top_k_passed_indices = passed_indices[np.argsort(passed_proba)[-top_k:][::-1]]

            # Reset all predictions for this sample
            y_pred_binary[i, :] = 0
            # Set only the top-k
            y_pred_binary[i, top_k_passed_indices] = 1

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
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Global probability threshold (default: None, uses per-label thresholds if available, else 0.55). Only genres above threshold will be included.",
    ),
    use_global_threshold: bool = typer.Option(
        False,
        "--use-global-threshold",
        help="Force use of global threshold even if per-label thresholds are available.",
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
            descriptions,
            model_path=model_path,
            threshold=threshold,
            top_k=top_k,
            use_global_threshold=use_global_threshold,
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
