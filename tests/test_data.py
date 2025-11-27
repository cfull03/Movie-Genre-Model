from __future__ import annotations
from pathlib import Path

import pytest
import pandas as pd
from loguru import logger

from descriptions.dataset import load_data

# Note: Actual data may have 'movie_name' or 'title', and 'genre' or 'genres'
# We'll check for required columns more flexibly


def test_data_columns():
    """Test that loaded data has the required columns."""
    df = load_data()
    # Check for essential columns (description is required, genre/genres may vary)
    assert 'description' in df.columns, "Data must have 'description' column"
    # Check for at least one genre-related column
    genre_cols = [col for col in df.columns if 'genre' in col.lower() or 'title' in col.lower() or 'movie' in col.lower()]
    assert len(genre_cols) > 0, f"Data must have a genre/title column. Found columns: {list(df.columns)}"
    logger.success("Data columns test passed")


def test_data_types():
    """Test that data columns have the correct data types."""
    df = load_data()
    assert df["description"].dtype == "object", f"Expected 'description' to be object, got {df['description'].dtype}"
    # Find genre column (could be 'genre' or 'genres')
    genre_col = [col for col in df.columns if 'genre' in col.lower()]
    if genre_col:
        assert df[genre_col[0]].dtype == "object", f"Expected genre column to be object, got {df[genre_col[0]].dtype}"
    logger.success("Data types test passed")