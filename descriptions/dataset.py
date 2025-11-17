from __future__ import annotations
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from descriptions.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

__all__ = ["load_data", "to_interim", "to_processed"]

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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
