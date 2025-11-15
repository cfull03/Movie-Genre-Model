from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from descriptions.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

__all__ = ["generate_features", "get_vectorizer"]

# ----- PRIVATE HELPER FUNCTIONS -----
def _build_vectorizer(data: pd.DataFrame) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit(data["description"])

def _preprocess_genres(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and split the genres column into lists of genre strings."""
    df = data.copy()

    s = df['genre'].fillna("").astype(str).str.strip()
    df['genre'] = (
        s.str.split(r"\s*,\s*")
         .apply(lambda genres: sorted({g.strip() for g in genres if g.strip()}))
    )

    return df


# ----- PUBLIC API -----
def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate features from the dataset."""
    logger.info("Generating features from dataset...")
    data = _preprocess_genres(data)
    vectorizer = _build_vectorizer(data)
    data['description_vector'] = vectorizer.transform(data["description"])
    return data

def get_vectorizer() -> TfidfVectorizer:
    return _build_vectorizer(data)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
