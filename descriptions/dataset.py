from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from descriptions.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

__all__ = [
    "load_data",
    "load_interim",
    "load_processed",
    "to_interim",
    "to_processed",
    "merge_wiki_dataset_left",
]


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


def _normalize_title(title: str) -> str:
    """
    Normalize a movie title for matching.

    Args:
        title: Movie title string

    Returns:
        Normalized title (lowercase, stripped, special chars removed)
    """
    if pd.isna(title):
        return ""
    return (
        str(title)
        .lower()
        .strip()
        .replace(":", "")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
        .replace(",", "")
        .replace("-", " ")
        .replace("  ", " ")
    )


def _fuzzy_match_title(
    left_title: str, right_titles: pd.Series, threshold: float = 0.85
) -> str | None:
    """
    Find the best fuzzy match for a title in a series of titles.

    Args:
        left_title: Title to match
        right_titles: Series of titles to search in
        threshold: Minimum similarity ratio (0-1)

    Returns:
        Best matching title or None if no match above threshold
    """
    normalized_left = _normalize_title(left_title)
    if not normalized_left:
        return None

    best_match = None
    best_ratio = 0.0

    for right_title in right_titles:
        normalized_right = _normalize_title(right_title)
        if not normalized_right:
            continue

        ratio = SequenceMatcher(None, normalized_left, normalized_right).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = right_title

    if best_ratio >= threshold:
        return best_match
    return None


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


def clean(
    input_path: Path = RAW_DATA_DIR / "top_movies.csv",
    output_path: Path = INTERIM_DATA_DIR / "cleaned_movies.csv",
    force: bool = False,
) -> None:
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


def merge_wiki_dataset_left(
    left_path: Path = RAW_DATA_DIR / "top_movies.csv",
    right_path: Path | None = None,
    output_path: Path = INTERIM_DATA_DIR / "merged_movies.csv",
    fuzzy_threshold: float = 0.85,
    use_fuzzy: bool = True,
) -> pd.DataFrame:
    """
    Perform a left merge of top_movies.csv (left) with wiki_movie_plots_deduped.csv (right).

    This keeps all movies from top_movies.csv and enriches them with data from
    wiki_movie_plots_deduped.csv where matches are found (exact or fuzzy).

    Args:
        left_path: Path to top_movies.csv (left dataset)
        right_path: Path to wiki_movie_plots_deduped.csv (right dataset)
        output_path: Path where merged data will be saved
        fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0-1)
        use_fuzzy: If True, use fuzzy matching for titles that don't match exactly

    Returns:
        Merged DataFrame with all left movies and enriched data from right
    """
    logger.info("=" * 70)
    logger.info("Left merge: top_movies.csv ← wiki_movie_plots_deduped.csv")
    logger.info("=" * 70)

    # Load left dataset (top_movies.csv)
    logger.info(f"Loading left dataset from {left_path}...")
    left_df = load_data(left_path)
    logger.success(f"✓ Loaded {len(left_df)} movies from top_movies.csv")

    # Load right dataset (wiki_movie_plots_deduped.csv)
    if right_path is None:
        # Try external directory first, then raw directory
        right_path = EXTERNAL_DATA_DIR / "wiki_movie_plots_deduped.csv"
        if not right_path.exists():
            right_path = RAW_DATA_DIR / "wiki_movie_plots_deduped.csv"

    logger.info(f"Loading right dataset from {right_path}...")
    try:
        right_df = pd.read_csv(right_path)
        logger.success(f"✓ Loaded {len(right_df)} movies from wiki dataset")
    except FileNotFoundError:
        logger.error(f"Wiki dataset not found at {right_path}")
        raise FileNotFoundError(
            f"Wiki dataset not found. Please ensure wiki_movie_plots_deduped.csv "
            f"exists in {EXTERNAL_DATA_DIR} or {RAW_DATA_DIR}"
        )

    # Prepare right dataset: rename columns to match left
    logger.info("Preparing right dataset...")
    right_df_clean = right_df.copy()
    right_df_clean = right_df_clean.rename(
        columns={"Title": "movie_name", "Plot": "description", "Genre": "genre"}
    )

    # Drop rows with missing essential data
    right_df_clean = right_df_clean.dropna(subset=["movie_name", "description"])
    right_df_clean = right_df_clean[["movie_name", "genre", "description"]].copy()

    logger.success(f"✓ Prepared {len(right_df_clean)} wiki movies")

    # Create normalized keys for matching
    logger.info("Creating normalized keys for matching...")
    left_df["_match_key"] = left_df["movie_name"].apply(_normalize_title)
    right_df_clean["_match_key"] = right_df_clean["movie_name"].apply(_normalize_title)

    # Step 1: Exact matches
    logger.info("Finding exact matches...")
    exact_matches = left_df["_match_key"].isin(right_df_clean["_match_key"])
    exact_count = exact_matches.sum()
    logger.success(f"✓ Found {exact_count} exact matches")

    # Step 2: Fuzzy matches for non-exact matches (if enabled)
    fuzzy_matches = {}
    if use_fuzzy and exact_count < len(left_df):
        logger.info(f"Finding fuzzy matches (threshold={fuzzy_threshold})...")
        unmatched_left = left_df[~exact_matches].copy()
        unmatched_right = right_df_clean[
            ~right_df_clean["_match_key"].isin(left_df["_match_key"])
        ].copy()

        if len(unmatched_left) > 0 and len(unmatched_right) > 0:
            logger.info(f"  Searching {len(unmatched_left)} unmatched movies...")
            with tqdm(total=len(unmatched_left), desc="Fuzzy matching") as pbar:
                for idx, row in unmatched_left.iterrows():
                    match = _fuzzy_match_title(
                        row["movie_name"], unmatched_right["movie_name"], threshold=fuzzy_threshold
                    )
                    if match:
                        fuzzy_matches[row["movie_name"]] = match
                    pbar.update(1)

            logger.success(f"✓ Found {len(fuzzy_matches)} fuzzy matches")

    # Step 3: Create mapping from left movie_name to right movie_name
    logger.info("Creating merge mapping...")

    # Build mapping dictionary: left movie_name -> right movie_name
    merge_mapping = {}

    # Add exact matches
    left_exact = left_df[exact_matches].copy()
    right_exact = right_df_clean[
        right_df_clean["_match_key"].isin(left_exact["_match_key"])
    ].copy()

    # Create mapping for exact matches using normalized keys
    for left_idx, left_row in left_exact.iterrows():
        left_key = left_row["_match_key"]
        right_match = right_exact[right_exact["_match_key"] == left_key]
        if len(right_match) > 0:
            # Take first match if multiple (shouldn't happen, but handle it)
            merge_mapping[left_row["movie_name"]] = right_match.iloc[0]["movie_name"]

    # Add fuzzy matches
    if fuzzy_matches:
        merge_mapping.update(fuzzy_matches)

    logger.info(f"Created mapping for {len(merge_mapping)} movies")

    # Step 4: Create right dataset with merge keys
    right_matched = right_df_clean[
        right_df_clean["movie_name"].isin(merge_mapping.values())
    ].copy()

    # Create reverse mapping: right movie_name -> left movie_name
    reverse_mapping = {v: k for k, v in merge_mapping.items()}
    right_matched["_merge_key"] = right_matched["movie_name"].map(reverse_mapping)

    # Step 5: Perform left merge
    logger.info("Performing left merge...")
    left_df["_merge_key"] = left_df["movie_name"]

    merged_df = pd.merge(
        left_df,
        right_matched[["_merge_key", "genre", "description"]],
        on="_merge_key",
        how="left",
        suffixes=("", "_wiki"),
    )

    # Step 6: Combine columns (prefer wiki data if available, otherwise keep original)
    logger.info("Combining columns...")

    # For description: prefer wiki if available (usually longer/more detailed)
    merged_df["description"] = merged_df.apply(
        lambda row: row["description_wiki"]
        if pd.notna(row["description_wiki"])
        else row["description"],
        axis=1,
    )

    # For genre: prefer wiki if available, otherwise keep original
    merged_df["genre"] = merged_df.apply(
        lambda row: (row["genre_wiki"] if pd.notna(row["genre_wiki"]) else row["genre"]), axis=1
    )

    # Drop helper columns
    merged_df = merged_df.drop(
        columns=["_match_key", "_merge_key", "description_wiki", "genre_wiki"]
    )

    # Ensure we have the right columns
    merged_df = merged_df[["movie_name", "genre", "description"]].copy()

    logger.success(f"✓ Merge complete: {len(merged_df)} movies")
    logger.info(f"   ({exact_count} exact matches + {len(fuzzy_matches)} fuzzy matches)")

    # Set index and save
    logger.info("Setting index and saving...")
    merged_df = _set_index(merged_df)
    to_interim(merged_df, output_path)

    logger.info("=" * 70)
    logger.success("🎉 Left merge completed successfully!")
    logger.info("=" * 70)

    return merged_df


@app.command()
def main(
    left_path: Path = RAW_DATA_DIR / "top_movies.csv",
    right_path: Path | None = None,
    output_path: Path = INTERIM_DATA_DIR / "merged_movies.csv",
    fuzzy_threshold: float = 0.85,
    use_fuzzy: bool = True,
    force: bool = False,
):
    """
    Merge top_movies.csv (left) with wiki_movie_plots_deduped.csv (right) using left merge.

    This keeps all movies from top_movies.csv and enriches them with data from
    wiki_movie_plots_deduped.csv where matches are found (exact or fuzzy).
    Output is saved to interim directory.

    Uses fuzzy matching by default to catch variations like:
    - Roman numerals (II vs 2)
    - Number formats (12 vs Twelve)
    - Article placement (The Movie vs Movie, The)

    Args:
        left_path: Path to top_movies.csv (left dataset)
        right_path: Path to wiki_movie_plots_deduped.csv (right dataset).
                   If None, searches in external/ and raw/ directories.
        output_path: Path where merged data will be saved (default: interim/merged_movies.csv)
        fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0-1, default: 0.85)
        use_fuzzy: Use fuzzy matching for titles that don't match exactly (default: True)
        force: If True, overwrite existing output file
    """
    if output_path.exists() and not force:
        logger.info(f"Merged dataset already exists at {output_path}")
        logger.info("Skipping merge. Use --force to overwrite.")
        return

    merge_wiki_dataset_left(
        left_path=left_path,
        right_path=right_path,
        output_path=output_path,
        fuzzy_threshold=fuzzy_threshold,
        use_fuzzy=use_fuzzy,
    )


if __name__ == "__main__":
    app()
