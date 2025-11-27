from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import typer

from descriptions.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from descriptions.dataset import load_interim, to_processed

# Handle both relative (when imported as module) and absolute (when run as script) imports
try:
    from .model import load_model, save_model
except ImportError:
    from descriptions.modeling.model import load_model, save_model

app = typer.Typer()

__all__ = [
    "build_preprocessor",
    "load_preprocessors",
    "save_preprocessors",
]


# ----- PRIVATE HELPER FUNCTIONS -----
def _preprocess_genres(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and split the genres column into lists of genre strings.

    Args:
        data: DataFrame with a 'genre' column containing genre strings

    Returns:
        DataFrame with 'genre' column converted to lists of unique, sorted genre strings
    """
    logger.info("Starting genre preprocessing: cleaning and splitting genre strings")
    df = data.copy()

    logger.debug("Filling missing genres with empty strings")
    s = df["genre"].fillna("").astype(str).str.strip()
    n_filled = s.isin([""]).sum()
    if n_filled > 0:
        logger.warning(f"Found {n_filled} samples with missing genres (filled with empty string)")

    logger.debug("Splitting genre strings by comma and cleaning")
    df["genre"] = s.str.split(r"\s*,\s*").apply(
        lambda genres: sorted({g.strip() for g in genres if g.strip()})
    )
    total_genres = sum(len(genres) for genres in df["genre"])
    avg_genres = total_genres / len(df) if len(df) > 0 else 0
    logger.success(
        f"Genre preprocessing complete: {len(df)} samples processed, "
        f"average {avg_genres:.2f} genres per sample"
    )
    return df


def _generate_targets(
    data: pd.DataFrame,
    mlb: Optional[MultiLabelBinarizer] = None,
) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    """
    Generate multi-label targets from the genre column.

    Args:
        data: DataFrame with 'genre' column containing genre lists
        mlb: Optional pre-fitted MultiLabelBinarizer. If None, creates and fits a new one.

    Returns:
        Tuple of (binary label array, MultiLabelBinarizer)
    """
    logger.info(f"Generating multi-label targets from {len(data)} samples...")

    df = _preprocess_genres(data)
    genres_list = df["genre"]

    if mlb is None:
        logger.debug("Creating and fitting new MultiLabelBinarizer")
        _, mlb = build_preprocessor()
        y = mlb.fit_transform(genres_list)
        n_labels = len(mlb.classes_)
        logger.info(f"MultiLabelBinarizer fitted: {n_labels} unique genre labels identified")
    else:
        logger.debug("Using pre-fitted MultiLabelBinarizer for transformation")
        y = mlb.transform(genres_list)
        logger.info(
            f"Transformed labels using existing MultiLabelBinarizer ({len(mlb.classes_)} labels)"
        )

    logger.success(f"Targets generated: shape {y.shape} (samples × labels)")
    return y, mlb


def _generate_descriptions(
    data: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Generate TF-IDF features from the description column.

    Args:
        data: DataFrame with 'description' column containing text descriptions
        vectorizer: Optional pre-fitted TfidfVectorizer. If None, creates and fits a new one.

    Returns:
        Tuple of (sparse TF-IDF feature matrix, TfidfVectorizer)
    """
    logger.info(f"Generating TF-IDF features from {len(data)} movie descriptions...")

    texts = data["description"].fillna("").astype(str)
    n_empty = texts.str.len().eq(0).sum()
    if n_empty > 0:
        logger.warning(f"Found {n_empty} samples with empty descriptions")

    if vectorizer is None:
        logger.debug("Creating and fitting new TfidfVectorizer")
        vectorizer, _ = build_preprocessor()
        X_desc = vectorizer.fit_transform(texts)
        logger.info(
            f"TfidfVectorizer fitted: {X_desc.shape[1]} features extracted "
            f"(max_features={vectorizer.max_features})"
        )
    else:
        logger.debug("Using pre-fitted TfidfVectorizer for transformation")
        X_desc = vectorizer.transform(texts)
        logger.info(
            f"Transformed descriptions using existing TfidfVectorizer ({X_desc.shape[1]} features)"
        )

    logger.success(f"Description features generated: sparse matrix shape {X_desc.shape}")
    return X_desc, vectorizer


# ----- PUBLIC API -----
def build_preprocessor() -> Tuple[TfidfVectorizer, MultiLabelBinarizer]:
    """
    Build and return the preprocessing components (TfidfVectorizer and MultiLabelBinarizer).

    Returns:
        Tuple of (TfidfVectorizer, MultiLabelBinarizer) ready for fitting
    """
    logger.debug("Building preprocessing components: TfidfVectorizer and MultiLabelBinarizer")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    mlb = MultiLabelBinarizer()
    logger.debug(
        "TfidfVectorizer configured: max_features=20000, ngram_range=(1,2), sublinear_tf=True"
    )
    return vectorizer, mlb


def load_preprocessors(
    vectorizer_path: Optional[Path] = None,
    mlb_path: Optional[Path] = None,
) -> Tuple[TfidfVectorizer, MultiLabelBinarizer]:
    """
    Load fitted preprocessors from saved model files.

    Args:
        vectorizer_path: Path to saved TfidfVectorizer. Defaults to MODELS_DIR / "tfidf_vectorizer.joblib"
        mlb_path: Path to saved MultiLabelBinarizer. Defaults to MODELS_DIR / "genre_binarizer.joblib"

    Returns:
        Tuple of (fitted_vectorizer, fitted_mlb)
    """
    if vectorizer_path is None:
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if mlb_path is None:
        mlb_path = MODELS_DIR / "genre_binarizer.joblib"

    logger.info(f"Loading TfidfVectorizer from {vectorizer_path}...")
    vectorizer = load_model("tfidf_vectorizer")
    logger.debug(
        f"TfidfVectorizer loaded: max_features={vectorizer.max_features}, ngram_range={vectorizer.ngram_range}"
    )

    logger.info(f"Loading MultiLabelBinarizer from {mlb_path}...")
    mlb = load_model("genre_binarizer")
    logger.debug(f"MultiLabelBinarizer loaded: {len(mlb.classes_)} genre classes")

    logger.success(
        f"Preprocessors loaded successfully: TfidfVectorizer ({vectorizer.max_features} features), "
        f"MultiLabelBinarizer ({len(mlb.classes_)} labels)"
    )
    return vectorizer, mlb


def save_preprocessors(
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
) -> None:
    """
    Save the fitted preprocessors to the models directory.

    Args:
        vectorizer: Fitted TfidfVectorizer to save
        mlb: Fitted MultiLabelBinarizer to save
    """
    logger.info("Saving fitted preprocessors to models directory...")
    save_model(vectorizer, "tfidf_vectorizer")
    logger.debug(f"TfidfVectorizer saved: {vectorizer.max_features} features")
    save_model(mlb, "genre_binarizer")
    logger.debug(f"MultiLabelBinarizer saved: {len(mlb.classes_)} genre classes")
    logger.success("Fitted preprocessors saved successfully to models directory")


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    force: bool = False,
) -> None:
    """
    Preprocess movie data: generate features and targets, save processed data and fitted preprocessors.

    WARNING: This function fits TF-IDF and MLB on the ENTIRE dataset. This causes data leakage
    if the processed data is later used for training with train/test splits. For training,
    use the training pipeline in train.py which properly splits data before fitting preprocessors.

    This function is intended for:
    - Exploratory data analysis
    - Creating baseline datasets for evaluation
    - Visualization purposes

    Args:
        input_path: Path to cleaned interim data CSV file
        output_path: Path where processed data will be saved
        force: If True, reprocess even if outputs already exist (default: False)
    """
    # Check if outputs already exist
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    mlb_path = MODELS_DIR / "genre_binarizer.joblib"

    outputs_exist = output_path.exists() and vectorizer_path.exists() and mlb_path.exists()

    if outputs_exist and not force:
        logger.info(f"Processed data already exists at {output_path}")
        logger.info(f"Preprocessors already exist at {vectorizer_path.name} and {mlb_path.name}")
        logger.info("Skipping preprocessing. Use --force to reprocess.")
        return

    logger.warning(
        "⚠️  WARNING: This preprocessing fits TF-IDF and MLB on the ENTIRE dataset. "
        "This causes data leakage if used for training. For training, use train.py instead."
    )
    logger.info(f"Loading interim data from {input_path}...")
    data = load_interim(input_path)
    logger.success(f"Loaded {len(data)} samples from interim data")

    logger.info("=" * 70)
    logger.info("Step 1/4: Generating TF-IDF features from descriptions")
    logger.info("=" * 70)
    X, vect = _generate_descriptions(data)
    logger.success(f"TF-IDF features generated: {X.shape[0]} samples × {X.shape[1]} features")

    logger.info("=" * 70)
    logger.info("Step 2/4: Generating multi-label genre targets")
    logger.info("=" * 70)
    y, mlb = _generate_targets(data)
    logger.success(f"Genre targets generated: {y.shape[0]} samples × {y.shape[1]} labels")

    logger.info("=" * 70)
    logger.info("Step 3/4: Converting sparse matrices to DataFrames")
    logger.info("=" * 70)
    logger.info(f"Converting sparse TF-IDF matrix to dense DataFrame (shape: {X.shape})...")
    logger.warning("This may take a while for large matrices...")

    # Use data index if it exists and is not a default RangeIndex, otherwise create one
    if hasattr(data, "index") and not isinstance(data.index, pd.RangeIndex):
        index = data.index
    else:
        index = pd.RangeIndex(len(data))

    # Convert sparse matrix to dense array (this can be slow for large matrices)
    with tqdm(total=2, desc="Converting to dense format", unit="step") as pbar:
        pbar.set_description("Converting sparse matrix to dense array")
        X_dense = X.toarray()
        pbar.update(1)

        pbar.set_description("Creating feature DataFrame")
        X_df = pd.DataFrame(
            X_dense, index=index, columns=[f"tfidf_{i}" for i in range(X.shape[1])]
        )
        pbar.update(1)

    logger.success(
        f"Feature DataFrame created: {X_df.shape[0]} samples × {X_df.shape[1]} features"
    )

    # Convert y (numpy array) to DataFrame with genre names
    logger.debug("Converting label array to DataFrame with genre column names")
    y_df = pd.DataFrame(y, index=index, columns=mlb.classes_)
    logger.debug(f"Label DataFrame created: {y_df.shape[0]} samples × {y_df.shape[1]} labels")

    # Combine X and y into one DataFrame
    logger.info("Combining features and labels into single processed DataFrame...")
    processed_df = pd.concat([X_df, y_df], axis=1)
    logger.success(
        f"Processed DataFrame created: {processed_df.shape[0]} samples, "
        f"{X_df.shape[1]} features + {y_df.shape[1]} labels = {processed_df.shape[1]} total columns"
    )

    logger.info("=" * 70)
    logger.info("Step 4/4: Saving processed data and preprocessors")
    logger.info("=" * 70)
    logger.info(f"Saving processed data to {output_path}...")
    logger.warning("This may take a while for large datasets...")
    with tqdm(total=1, desc="Saving processed data", unit="file") as pbar:
        to_processed(processed_df, output_path)
        pbar.update(1)
    logger.success(f"Processed data saved successfully to {output_path}")

    save_preprocessors(vect, mlb)
    logger.info("=" * 70)
    logger.success("Preprocessing pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
