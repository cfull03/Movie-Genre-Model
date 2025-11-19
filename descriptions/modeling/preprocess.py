from pathlib import Path
from typing import Optional, Tuple
from scipy.sparse import csr_matrix
import numpy as np

from loguru import logger
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from descriptions.dataset import load_interim, to_processed
from descriptions.modeling.model import save_model, load_model
from descriptions.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()

__all__ = [
    "build_preprocessor",
    "load_preprocessors",
    "save_preprocessors",
]

# ----- PRIVATE HELPER FUNCTIONS -----
def _preprocess_genres(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and split the genres column into lists of genre strings."""
    logger.info("Preprocessing genres...")
    df = data.copy()

    logger.info("Filling missing genres...")
    s = df["genre"].fillna("").astype(str).str.strip()
    logger.success("Missing genres filled.")

    logger.info("Splitting genres into lists...")
    df["genre"] = (
        s.str.split(r"\s*,\s*")
         .apply(lambda genres: sorted({g.strip() for g in genres if g.strip()}))
    )
    logger.success("Genres preprocessing complete.")
    return df


def _generate_targets(
    data: pd.DataFrame,
    mlb: Optional[MultiLabelBinarizer] = None,
) -> Tuple[np.ndarray, MultiLabelBinarizer]:
    """Generate multi-label targets from the genre column."""
    logger.info("Generating targets from dataset...")

    df = _preprocess_genres(data)
    genres_list = df["genre"]

    if mlb is None:
        _, mlb = build_preprocessor()
        y = mlb.fit_transform(genres_list)
    else:
        y = mlb.transform(genres_list)

    logger.success("Targets generated.")
    return y, mlb


def _generate_descriptions(
    data: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Generate TF-IDF features from the description column."""
    logger.info("Generating description features...")

    texts = data["description"].fillna("").astype(str)

    if vectorizer is None:
        vectorizer, _ = build_preprocessor()
        X_desc = vectorizer.fit_transform(texts)
    else:
        X_desc = vectorizer.transform(texts)

    logger.success("Description features generated.")
    return X_desc, vectorizer


# ----- PUBLIC API -----
def build_preprocessor() -> Tuple[TfidfVectorizer, MultiLabelBinarizer]:
    """Build and return the preprocessing components (TfidfVectorizer and MultiLabelBinarizer)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    mlb = MultiLabelBinarizer()
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
    
    logger.info(f"Loading fitted TfidfVectorizer from {vectorizer_path}...")
    vectorizer = load_model("tfidf_vectorizer")
    
    logger.info(f"Loading fitted MultiLabelBinarizer from {mlb_path}...")
    mlb = load_model("genre_binarizer")
    
    logger.success("Fitted preprocessors loaded successfully.")
    return vectorizer, mlb


def save_preprocessors(
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
) -> None:
    """Save the fitted preprocessors to the models directory."""
    save_model(vectorizer, "tfidf_vectorizer")
    save_model(mlb, "genre_binarizer")
    logger.success("Fitted preprocessors saved successfully.")



@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
) -> None:
    """Preprocess movie data: generate features and targets, save processed data and fitted preprocessors."""
    data = load_interim(input_path)

    logger.info("Preparing to generate Descriptions feature (X)")
    X, vect = _generate_descriptions(data)
    logger.success("Successfully generated Descriptions and fitted and transformed the descriptions")

    logger.info("Preparing to generate Genres feature (y)")
    y, mlb = _generate_targets(data)
    logger.success("Successfully generated Genres and fitted and transformed the Genres")

    # Convert X (sparse matrix) to DataFrame with feature names
    logger.info("Converting features to DataFrame...")
    # Use data index if it exists and is not a default RangeIndex, otherwise create one
    if hasattr(data, 'index') and not isinstance(data.index, pd.RangeIndex):
        index = data.index
    else:
        index = pd.RangeIndex(len(data))
    
    X_df = pd.DataFrame(
        X.toarray(),
        index=index,
        columns=[f"tfidf_{i}" for i in range(X.shape[1])]
    )
    
    # Convert y (numpy array) to DataFrame with genre names
    y_df = pd.DataFrame(
        y,
        index=index,
        columns=mlb.classes_
    )
    
    # Combine X and y into one DataFrame
    processed_df = pd.concat([X_df, y_df], axis=1)
    logger.success("Features and targets combined into DataFrame")
    
    # Save to processed_movies.csv
    logger.info(f"Saving processed data to {output_path}...")
    to_processed(processed_df, output_path)
    logger.success(f"Processed data saved to {output_path}")

    logger.info("Saving fitted preprocessors...")
    save_preprocessors(vect, mlb)
    logger.success("Saved the: TF-IDF Vectorizer and MultiLabelBinarizer")


if __name__ == "__main__":
    app()
