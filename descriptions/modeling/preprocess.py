from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
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
    "_generate_descriptions",
    "_generate_targets",
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
    min_genre_percentage: float = 5.0,
) -> Tuple[np.ndarray, MultiLabelBinarizer, pd.DataFrame]:
    """
    Generate multi-label targets from the genre column, filtering out rare genres.

    Args:
        data: DataFrame with 'genre' column containing genre lists
        mlb: Optional pre-fitted MultiLabelBinarizer. If None, creates and fits a new one.
        min_genre_percentage: Minimum percentage of samples a genre must appear in to be kept (default: 5.0)

    Returns:
        Tuple of (binary label array, MultiLabelBinarizer, filtered DataFrame with same row order as y)
    """
    logger.info(f"Generating multi-label targets from {len(data)} samples...")

    df = _preprocess_genres(data)
    genres_list = df["genre"]

    if mlb is None:
        logger.debug("Creating and fitting new MultiLabelBinarizer")
        # Step 1: Analyze genre frequencies to identify rare genres
        logger.info(f"Analyzing genre frequencies (removing genres < {min_genre_percentage}%)...")
        mlb_temp = MultiLabelBinarizer()
        y_temp = mlb_temp.fit_transform(genres_list)

        # Calculate genre frequencies and percentages
        genre_counts = y_temp.sum(axis=0)
        genre_percentages = (genre_counts / len(y_temp)) * 100

        # Identify genres to remove
        genres_to_remove = set()
        for i, (genre, percentage) in enumerate(zip(mlb_temp.classes_, genre_percentages)):
            if percentage < min_genre_percentage:
                genres_to_remove.add(genre)
                logger.debug(
                    f"  Removing '{genre}': {percentage:.2f}% ({int(genre_counts[i])} samples)"
                )

        if genres_to_remove:
            logger.info(
                f"Identified {len(genres_to_remove)} genres to remove: {sorted(genres_to_remove)}"
            )

        # Step 2: Filter out rare genres from genre lists
        genres_list_filtered = genres_list.apply(
            lambda genres: sorted({g for g in genres if g not in genres_to_remove})
        )

        # Count how many samples lost all genres
        samples_lost = (genres_list_filtered.apply(len) == 0).sum()
        if samples_lost > 0:
            logger.warning(
                f"{samples_lost} samples lost all genres after filtering and will be removed"
            )
            keep_mask = genres_list_filtered.apply(len) > 0
            genres_list_filtered = genres_list_filtered[keep_mask]
            data = data[keep_mask]
            logger.info(f"Removed {samples_lost} samples with no genres")

        # Step 3: Fit MultiLabelBinarizer on filtered training data
        logger.debug("Fitting MultiLabelBinarizer on filtered genre lists")
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(genres_list_filtered)
        n_labels = len(mlb.classes_)
        logger.info(f"MultiLabelBinarizer fitted: {n_labels} unique genre labels identified")
        logger.info(f"Genres kept: {sorted(mlb.classes_)}")
    else:
        logger.debug("Using pre-fitted MultiLabelBinarizer for transformation")
        # Filter genres to only those in mlb.classes_
        genres_list_filtered = genres_list.apply(
            lambda genres: sorted({g for g in genres if g in mlb.classes_})
        )
        # Remove samples that lost all genres
        keep_mask = genres_list_filtered.apply(len) > 0
        genres_list_filtered = genres_list_filtered[keep_mask]
        data = data[keep_mask]
        y = mlb.transform(genres_list_filtered)
        logger.info(
            f"Transformed labels using existing MultiLabelBinarizer ({len(mlb.classes_)} labels)"
        )

    logger.success(f"Targets generated: shape {y.shape} (samples × labels)")
    return y, mlb, data


def _generate_descriptions(
    data: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Generate TF-IDF features from the description column.

    Args:
        data: DataFrame with 'description' column containing text descriptions
        vectorizer: Optional TfidfVectorizer. If None, creates and fits a new one with defaults.
                   If provided but not fitted (no vocabulary_ attribute), fits it on the data.
                   If provided and already fitted, uses it to transform the data.

    Returns:
        Tuple of (sparse TF-IDF feature matrix, TfidfVectorizer)
    """
    logger.info(f"Generating TF-IDF features from {len(data)} movie descriptions...")

    texts = data["description"].fillna("").astype(str)
    n_empty = texts.str.len().eq(0).sum()
    if n_empty > 0:
        logger.warning(f"Found {n_empty} samples with empty descriptions")

    if vectorizer is None:
        logger.debug("Creating and fitting new TfidfVectorizer with default parameters")
        vectorizer, _, _, _ = build_preprocessor()
        X_desc = vectorizer.fit_transform(texts)
        logger.info(
            f"TfidfVectorizer fitted: {X_desc.shape[1]} features extracted "
            f"(max_features={vectorizer.max_features})"
        )
    else:
        # Check if vectorizer is already fitted by checking for vocabulary_ attribute
        if hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None:
            logger.debug("Using pre-fitted TfidfVectorizer for transformation")
            X_desc = vectorizer.transform(texts)
            logger.info(
                f"Transformed descriptions using existing TfidfVectorizer ({X_desc.shape[1]} features)"
            )
        else:
            logger.debug("Fitting provided TfidfVectorizer with custom parameters")
            X_desc = vectorizer.fit_transform(texts)
            logger.info(
                f"TfidfVectorizer fitted with custom parameters: {X_desc.shape[1]} features extracted "
                f"(max_features={vectorizer.max_features}, ngram_range={vectorizer.ngram_range})"
            )

    logger.success(f"Description features generated: sparse matrix shape {X_desc.shape}")
    return X_desc, vectorizer


# ----- PUBLIC API -----
def build_preprocessor() -> Tuple[TfidfVectorizer, MultiLabelBinarizer, Normalizer, SelectKBest]:
    """
    Build and return the preprocessing components.

    Returns:
        Tuple of (TfidfVectorizer, MultiLabelBinarizer, Normalizer, SelectKBest) ready for fitting.
        - TfidfVectorizer: Converts text to TF-IDF features
        - MultiLabelBinarizer: Converts genre lists to binary multi-label format
        - Normalizer: L2 normalization for feature vectors (helps chi2 feature selection)
        - SelectKBest: Feature selector using chi2 test (k=4500 features)
    """
    logger.debug(
        "Building preprocessing components: TfidfVectorizer, MultiLabelBinarizer, Normalizer, and SelectKBest"
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_df=0.7,
        min_df=3,
        use_idf=True,
    )
    logger.debug(
        "TfidfVectorizer configured: max_features=10000, ngram_range=(1,3), "
        "sublinear_tf=True, max_df=0.7, min_df=3"
    )

    mlb = MultiLabelBinarizer()

    normalizer = Normalizer(norm="l2")
    logger.debug("Normalizer configured: L2 norm (normalizes each sample to unit length)")

    kbest = SelectKBest(score_func=chi2, k=4500)
    logger.debug("SelectKBest configured: chi2 test, k=4500 features")

    return vectorizer, mlb, normalizer, kbest


def load_preprocessors(
    vectorizer_path: Optional[Path] = None,
    mlb_path: Optional[Path] = None,
    normalizer_path: Optional[Path] = None,
    kbest_path: Optional[Path] = None,
) -> Tuple[TfidfVectorizer, MultiLabelBinarizer, Normalizer, SelectKBest]:
    """
    Load fitted preprocessors from saved model files.

    Args:
        vectorizer_path: Path to saved TfidfVectorizer. Defaults to MODELS_DIR / "tfidf_vectorizer.joblib"
        mlb_path: Path to saved MultiLabelBinarizer. Defaults to MODELS_DIR / "genre_binarizer.joblib"
        normalizer_path: Path to saved Normalizer. Defaults to MODELS_DIR / "normalizer.joblib"
        kbest_path: Path to saved SelectKBest feature selector. Defaults to MODELS_DIR / "feature_selector.joblib"

    Returns:
        Tuple of (fitted_vectorizer, fitted_mlb, fitted_normalizer, fitted_kbest)
    """
    if vectorizer_path is None:
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    if mlb_path is None:
        mlb_path = MODELS_DIR / "genre_binarizer.joblib"
    if normalizer_path is None:
        normalizer_path = MODELS_DIR / "normalizer.joblib"
    if kbest_path is None:
        kbest_path = MODELS_DIR / "feature_selector.joblib"

    logger.info(f"Loading TfidfVectorizer from {vectorizer_path}...")
    vectorizer = load_model("tfidf_vectorizer")
    logger.debug(
        f"TfidfVectorizer loaded: max_features={vectorizer.max_features}, ngram_range={vectorizer.ngram_range}"
    )

    logger.info(f"Loading MultiLabelBinarizer from {mlb_path}...")
    mlb = load_model("genre_binarizer")
    logger.debug(f"MultiLabelBinarizer loaded: {len(mlb.classes_)} genre classes")

    logger.info(f"Loading Normalizer from {normalizer_path}...")
    try:
        normalizer = load_model("normalizer")
        logger.debug(f"Normalizer loaded: norm={normalizer.norm}")
    except FileNotFoundError:
        # For backward compatibility, create a default normalizer if not found
        logger.warning("Normalizer not found, creating default L2 normalizer")
        normalizer = Normalizer(norm="l2")

    logger.info(f"Loading Feature Selector from {kbest_path}...")
    kbest = load_model("feature_selector")
    logger.debug(f"Feature Selector loaded: {kbest.k} features selected")

    logger.success(
        f"Preprocessors loaded successfully: TfidfVectorizer ({vectorizer.max_features} features), "
        f"MultiLabelBinarizer ({len(mlb.classes_)} labels), "
        f"Normalizer (L2 norm), "
        f"SelectKBest ({kbest.k} features selected)"
    )
    return vectorizer, mlb, normalizer, kbest


def save_preprocessors(
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
    normalizer: Normalizer,
    kbest: SelectKBest,
) -> None:
    """
    Save the fitted preprocessors to the models directory.

    Args:
        vectorizer: Fitted TfidfVectorizer to save
        mlb: Fitted MultiLabelBinarizer to save
        normalizer: Fitted Normalizer to save
        kbest: Fitted SelectKBest feature selector to save
    """
    logger.info("Saving fitted preprocessors to models directory...")
    save_model(vectorizer, "tfidf_vectorizer")
    logger.debug(f"TfidfVectorizer saved: {vectorizer.max_features} features")
    save_model(mlb, "genre_binarizer")
    logger.debug(f"MultiLabelBinarizer saved: {len(mlb.classes_)} genre classes")
    save_model(normalizer, "normalizer")
    logger.debug(f"Normalizer saved: norm={normalizer.norm}")
    save_model(kbest, "feature_selector")
    logger.debug(f"Feature Selector saved: {kbest.k} features selected")
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
    normalizer_path = MODELS_DIR / "normalizer.joblib"
    kbest_path = MODELS_DIR / "feature_selector.joblib"

    outputs_exist = (
        output_path.exists()
        and vectorizer_path.exists()
        and mlb_path.exists()
        and normalizer_path.exists()
        and kbest_path.exists()
    )

    if outputs_exist and not force:
        logger.info(f"Processed data already exists at {output_path}")
        logger.info(
            f"Preprocessors already exist: {vectorizer_path.name}, {mlb_path.name}, {normalizer_path.name}, {kbest_path.name}"
        )
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
    logger.info("Step 1/5: Generating TF-IDF features from descriptions")
    logger.info("=" * 70)
    logger.info("Step 1/5: Generating multi-label genre targets (filter data)")
    logger.info("=" * 70)
    y, mlb, data = _generate_targets(data)
    logger.success(f"Genre targets generated: {y.shape[0]} samples × {y.shape[1]} labels")

    logger.info("=" * 70)
    logger.info("Step 2/5: Generating TF-IDF features from filtered data")
    logger.info("=" * 70)
    X, vect = _generate_descriptions(data)
    logger.success(f"TF-IDF features generated: {X.shape[0]} samples × {X.shape[1]} features")

    logger.info("=" * 70)
    logger.info("Step 3/5: Applying L2 normalization")
    logger.info("=" * 70)
    logger.info("Applying L2 normalization to feature vectors...")
    normalizer = Normalizer(norm="l2")
    X = normalizer.fit_transform(X)
    logger.success("Normalization complete: features normalized to unit length (L2 norm)")

    logger.info("=" * 70)
    logger.info("Step 4/5: Applying feature selection with SelectKBest")
    logger.info("=" * 70)
    logger.info("Fitting SelectKBest feature selector (chi2 test, k=4500)...")
    kbest = SelectKBest(score_func=chi2, k=4500)
    X_selected = kbest.fit_transform(X, y)
    logger.success(
        f"Feature selection complete: {X_selected.shape[1]} features selected "
        f"(from {X.shape[1]} original features)"
    )
    X = X_selected  # Update X to use selected features

    logger.info("=" * 70)
    logger.info("Step 5/5: Converting sparse matrices to DataFrames")
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
    logger.info("Step 6/6: Saving processed data and preprocessors")
    logger.info("=" * 70)
    logger.info(f"Saving processed data to {output_path}...")
    logger.warning("This may take a while for large datasets...")
    with tqdm(total=1, desc="Saving processed data", unit="file") as pbar:
        to_processed(processed_df, output_path)
        pbar.update(1)
    logger.success(f"Processed data saved successfully to {output_path}")

    save_preprocessors(vect, mlb, normalizer, kbest)
    logger.info("=" * 70)
    logger.success("Preprocessing pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
