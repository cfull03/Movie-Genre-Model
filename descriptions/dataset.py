from __future__ import annotations
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib

from descriptions.config import MODELS_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

__all__ = ["load_data", "to_interim", "to_processed", "to_model"]

# ----- PRIVATE HELPER FUNCTIONS -----
def _basic_cleaning(df: pd.DataFrame, col: str = "movie_name") -> pd.DataFrame:
    logger.info("Performing basic cleaning on dataset...")
    data = df.copy()

    s = data[col].dropna().drop_duplicates()
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[ -]+", "_", regex=True)
    s = s.str.replace(r"[^a-z0-9_]", "", regex=True)

    data = data.loc[s.index].copy()
    data[col] = s

    logger.success("Basic cleaning complete.")
    return data


def _set_index(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Setting index on dataset...")
    out = df.set_index("movie_name")
    logger.success("Index set complete.")
    return out



# ----- PUBLIC API -----
def load_data(input_path: Path = RAW_DATA_DIR / "top_movies.csv") -> pd.DataFrame:
    """Load the raw movies CSV into a DataFrame."""
    return pd.read_csv(input_path)

def load_interim(input_path: Path = INTERIM_DATA_DIR / "top_movies.csv") -> pd.DataFrame:
    """Load the interim movies CSV into a DataFrame."""
    return pd.read_csv(input_path)

def load_processed(input_path: Path = PROCESSED_DATA_DIR / "top_movies.csv") -> pd.DataFrame:
    """Load the processed movies CSV into a DataFrame."""
    return pd.read_csv(input_path)

def load_model(input_path: Path = MODELS_DIR):
    "Load the intended model as a *.joblib"
    return joblib.load(input_path)


def to_interim(
    data: pd.DataFrame,
    output_path: Path = INTERIM_DATA_DIR / "top_movies.csv",
) -> None:
    """Save data to the interim data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def to_processed(
    data: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "top_movies.csv",
) -> None:
    """Save data to the processed data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

def save_model(
    model, 
    model_name: str
) -> None:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ensure it has an extension
    if not model_name.endswith(".joblib"):
        model_name = f"{model_name}.joblib"

    output_path = MODELS_DIR / model_name
    joblib.dump(model, output_path)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "top_movies.csv",
    output_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    # ----------------------------------------------
):
    # ---- Temporary CLI LOGIC FOR PREPPING DATA ----
    logger.info("Loading data from {input_path}...")
    df = load_data(input_path)
    logger.info("Basic cleaning data...")
    df = _basic_cleaning(df)
    logger.info("Setting index on data...")
    df = _set_index(df)
    logger.info("Saving data to {output_path}...")
    to_interim(df, output_path)
    logger.success("Data saved to {output_path}.")
    # ----------------------------------------------

if __name__ == "__main__":
    app()
