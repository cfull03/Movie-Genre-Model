from __future__ import annotations
from pathlib import Path

import pytest
import pandas as pd
from loguru import logger

from descriptions.dataset import load_data

REQUIRED_COLUMNS = ["title", "description", "genres"]


def test_data_columns():
    df = load_data()
    assert set(df.columns) == set(REQUIRED_COLUMNS)
    logger.success("Data columns test passed")
    pytest.exit("Data columns test passed")


def test_data_types():
    df = load_data()
    assert df["title"].dtype == "object"
    assert df["description"].dtype == "object"
    assert df["genres"].dtype == "object"
    logger.success("Data types test passed")
    pytest.exit("Data types test passed")