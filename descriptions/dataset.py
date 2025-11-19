from __future__ import annotations
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib

from descriptions.config import MODELS_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

__all__ = ["load_data", "load_interim", "load_processed",
            "to_interim", "to_processed"]

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
    try:
        logger.info(f"Loading data from {input_path}...")
        data = pd.read_csv(input_path)
        logger.success("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
    return data


def load_interim(input_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv") -> pd.DataFrame:
    """
    Load the interim movies CSV into a DataFrame.
    
    If the CSV has an index column (from previous processing), it will be loaded as the index.
    """
    try:
        logger.info(f"Loading data from {input_path}...")
        # Try to load with index first, fallback to no index
        try:
            data = pd.read_csv(input_path, index_col=0)
        except (ValueError, IndexError):
            # If index_col=0 fails, load without index
            data = pd.read_csv(input_path)
        logger.success("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
    return data

def load_processed(input_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv") -> pd.DataFrame:
    """
    Load the processed movies CSV into a DataFrame.
    
    If the CSV has an index column (from previous processing), it will be loaded as the index.
    """
    try:
        logger.info(f"Loading data from {input_path}...")
        # Try to load with index first, fallback to no index
        try:
            data = pd.read_csv(input_path, index_col=0)
        except (ValueError, IndexError):
            # If index_col=0 fails, load without index
            data = pd.read_csv(input_path)
        logger.success("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
    return data



def to_interim(
    data: pd.DataFrame,
    output_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
) -> None:
    """
    Save data to the interim data directory.
    
    If the DataFrame has a named index, it will be saved as a column.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with index if it's not the default RangeIndex
    has_named_index = data.index.name is not None or not isinstance(data.index, pd.RangeIndex)
    data.to_csv(output_path, index=has_named_index)


def to_processed(
    data: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
) -> None:
    """
    Save data to the processed data directory.
    
    If the DataFrame has a named index, it will be saved as a column.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save with index if it's not the default RangeIndex
    has_named_index = data.index.name is not None or not isinstance(data.index, pd.RangeIndex)
    data.to_csv(output_path, index=has_named_index)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "top_movies.csv",
    output_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    # ----------------------------------------------
):
    # ---- Temporary CLI LOGIC FOR PREPPING DATA ----
    logger.info(f"Loading data from {input_path}...")
    df = load_data(input_path)
    
    logger.info("Basic cleaning data...")
    df = _basic_cleaning(df)

    logger.info("Setting index on data...")
    df = _set_index(df)

    logger.info(f"Saving data to {output_path}...")
    to_interim(df, output_path)
    logger.success(f"Data saved to {output_path}.")
    # ----------------------------------------------

if __name__ == "__main__":
    app()
