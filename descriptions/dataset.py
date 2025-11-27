from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from descriptions.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

__all__ = ["load_data", "load_interim", "load_processed", "to_interim", "to_processed"]


# ----- PRIVATE HELPER FUNCTIONS -----
def _basic_cleaning(df: pd.DataFrame, col: str = "movie_name") -> pd.DataFrame:
    logger.info("Performing basic cleaning on dataset...")
    data = df.copy()

    with tqdm(total=4, desc="Cleaning data", unit="step") as pbar:
        pbar.set_description("Dropping duplicates and nulls")
        s = data[col].dropna().drop_duplicates()
        pbar.update(1)

        pbar.set_description("Converting to string and stripping")
        s = s.astype(str).str.strip()
        pbar.update(1)

        pbar.set_description("Replacing spaces and hyphens")
        s = s.str.replace(r"[ -]+", "_", regex=True)
        pbar.update(1)

        pbar.set_description("Removing special characters")
        s = s.str.replace(r"[^a-z0-9_]", "", regex=True)
        pbar.update(1)

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
    """
    Load the raw movies CSV into a DataFrame.

    Args:
        input_path: Path to the raw data CSV file

    Returns:
        DataFrame containing the raw movie data

    Raises:
        FileNotFoundError: If the input file does not exist
    """
    try:
        logger.info(f"Loading raw data from {input_path}...")
        data = pd.read_csv(input_path)
        logger.success(
            f"✓ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns"
        )
        logger.debug(f"Columns: {list(data.columns)}")
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        raise e
    return data


def load_interim(input_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv") -> pd.DataFrame:
    """
    Load the interim movies CSV into a DataFrame.

    If the CSV has an index column (from previous processing), it will be loaded as the index.

    Args:
        input_path: Path to the interim data CSV file

    Returns:
        DataFrame containing the cleaned interim data

    Raises:
        FileNotFoundError: If the input file does not exist
    """
    try:
        logger.info(f"Loading interim data from {input_path}...")
        # Try to load with index first, fallback to no index
        try:
            data = pd.read_csv(input_path, index_col=0)
            logger.debug("Loaded with index column")
        except (ValueError, IndexError):
            # If index_col=0 fails, load without index
            data = pd.read_csv(input_path)
            logger.debug("Loaded without index column")
        logger.success(
            f"✓ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns"
        )
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        raise e
    return data


def load_processed(input_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv") -> pd.DataFrame:
    """
    Load the processed movies CSV into a DataFrame.

    If the CSV has an index column (from previous processing), it will be loaded as the index.

    Args:
        input_path: Path to the processed data CSV file

    Returns:
        DataFrame containing the processed data (features + labels)

    Raises:
        FileNotFoundError: If the input file does not exist
    """
    try:
        logger.info(f"Loading processed data from {input_path}...")
        # Try to load with index first, fallback to no index
        try:
            data = pd.read_csv(input_path, index_col=0)
            logger.debug("Loaded with index column")
        except (ValueError, IndexError):
            # If index_col=0 fails, load without index
            data = pd.read_csv(input_path)
            logger.debug("Loaded without index column")
        logger.success(
            f"✓ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns"
        )
    except FileNotFoundError:
        logger.error(f"File not found at {input_path}")
        raise FileNotFoundError(f"File not found at {input_path}")
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
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
    force: bool = False,
):
    """
    Process raw data into cleaned interim data.

    Args:
        input_path: Path to raw data CSV file
        output_path: Path where cleaned data will be saved
        force: If True, reprocess even if output already exists (default: False)
    """
    # Check if output already exists
    if output_path.exists() and not force:
        logger.info(f"Cleaned data already exists at {output_path}")
        logger.info("Skipping data processing. Use --force to reprocess.")
        return

    logger.info("=" * 70)
    logger.info("Processing raw data into cleaned interim data")
    logger.info("=" * 70)

    logger.info(f"Step 1/3: Loading raw data from {input_path}...")
    df = load_data(input_path)

    logger.info("Step 2/3: Performing basic cleaning on data...")
    df = _basic_cleaning(df)
    logger.success(f"✓ Cleaning complete: {len(df)} samples remaining")

    logger.info("Step 3/3: Setting index and saving to interim...")
    df = _set_index(df)
    logger.debug(f"Index set to: {df.index.name}")

    logger.info(f"Saving cleaned data to {output_path}...")
    to_interim(df, output_path)
    logger.success(f"✓ Data saved successfully to {output_path}")

    logger.info("=" * 70)
    logger.success("🎉 Data processing pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
