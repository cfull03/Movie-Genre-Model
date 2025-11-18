from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from tqdm import tqdm
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

from descriptions.dataset import load_interim, to_processed, save_model
from descriptions.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR

app = typer.Typer()

__all__ = [
    "generate_descriptions",
    "generate_targets",
]

# ----- PRIVATE HELPER FUNCTIONS -----
def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english")


def _build_multilabel_binarizer() -> MultiLabelBinarizer:
    return MultiLabelBinarizer()


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


# ----- PUBLIC API -----
def generate_descriptions(
    data: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple:
    """Generate TF-IDF features from the description column."""
    logger.info("Generating description features...")

    texts = data["description"].fillna("").astype(str)

    if vectorizer is None:
        vectorizer = _build_vectorizer()
        X_desc = vectorizer.fit_transform(texts)
    else:
        X_desc = vectorizer.transform(texts)

    logger.success("Description features generated.")
    # ✅ return both features and fitted vectorizer
    return X_desc, vectorizer


def generate_targets(
    data: pd.DataFrame,
    mlb: Optional[MultiLabelBinarizer] = None,
) -> Tuple:
    """Generate multi-label targets from the genre column."""
    logger.info("Generating targets from dataset...")

    df = _preprocess_genres(data)
    genres_list = df["genre"]

    if mlb is None:
        mlb = _build_multilabel_binarizer()
        y = mlb.fit_transform(genres_list)
    else:
        y = mlb.transform(genres_list)

    logger.success("Targets generated.")
    # ✅ return both targets and fitted binarizer
    return y, mlb



@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    mlb_path: Path = MODELS_DIR / "genre_binarizer.joblib",
    tdidf_path: Path = MODELS_DIR / "tdidf_vectorizer.joblib",
) -> None:

    data = load_interim()

    logger.info("Preparing to generate Descriptions feature (X)")
    X, vect = generate_descriptions(data)
    logger.success("Successfully generated Descriptions and fitted and transformed the descriptions")

    logger.info("Preparing to generate Genres feature (y)")
    y, mlb = generate_targets(data)
    logger.success("Successfully generated Genres and fitted and transformed the Genres")

    # Convert X (sparse matrix) to DataFrame with feature names
    logger.info("Converting features to DataFrame...")
    X_df = pd.DataFrame(
        X.toarray(),
        index=data.index,
        columns=[f"tfidf_{i}" for i in range(X.shape[1])]
    )
    
    # Convert y (numpy array) to DataFrame with genre names
    y_df = pd.DataFrame(
        y,
        index=data.index,
        columns=mlb.classes_
    )
    
    # Combine X and y into one DataFrame
    processed_df = pd.concat([X_df, y_df], axis=1)
    logger.success("Features and targets combined into DataFrame")
    
    # Save to processed_movies.csv
    logger.info(f"Saving processed data to {output_path}...")
    to_processed(processed_df, output_path)
    logger.success(f"Processed data saved to {output_path}")

    save_model(vect, 'tdidf_vectorizer')
    logger.success("Saved the: TDIDF_Vectorizer")
    save_model(mlb, 'genre_binarizer')
    logger.success("Saved the: MultiLabelBinarizer")


if __name__ == "__main__":
    app()
