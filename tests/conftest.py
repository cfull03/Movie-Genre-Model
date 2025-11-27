"""Shared pytest fixtures for testing."""
from __future__ import annotations

from pathlib import Path
import tempfile
import shutil

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from descriptions.config import MODELS_DIR


@pytest.fixture
def sample_movie_data():
    """Sample movie data for testing."""
    return pd.DataFrame({
        'movie_name': ['test_movie_1', 'test_movie_2', 'test_movie_3'],
        'description': [
            'A thrilling action movie with explosions',
            'A romantic comedy about two people',
            'A sci-fi adventure in space'
        ],
        'genre': [
            'Action, Thriller',
            'Romance, Comedy',
            'Sci-Fi, Adventure'
        ]
    })


@pytest.fixture
def sample_processed_data():
    """Sample processed data with TF-IDF features and genre labels."""
    n_samples = 10
    n_features = 100
    n_genres = 5
    
    # Create TF-IDF features
    feature_columns = [f'tfidf_{i}' for i in range(n_features)]
    tfidf_features = np.random.rand(n_samples, n_features)
    
    # Create genre labels
    genre_columns = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller']
    genre_labels = np.random.randint(0, 2, size=(n_samples, n_genres))
    
    # Combine into DataFrame
    data = pd.DataFrame(
        np.hstack([tfidf_features, genre_labels]),
        columns=feature_columns + genre_columns,
        index=[f'movie_{i}' for i in range(n_samples)]
    )
    
    return data


@pytest.fixture
def sample_genres_list():
    """Sample list of genre lists for testing."""
    return [
        ['Action', 'Thriller'],
        ['Romance', 'Comedy'],
        ['Sci-Fi', 'Adventure'],
        ['Drama'],
        ['Action', 'Comedy', 'Drama']
    ]


@pytest.fixture
def sample_descriptions():
    """Sample movie descriptions for testing."""
    return [
        'A thrilling action movie with explosions and car chases',
        'A romantic comedy about two people who fall in love',
        'A sci-fi adventure set in outer space',
        'A dramatic story about family relationships',
        'An action-packed comedy with lots of laughs'
    ]


@pytest.fixture
def temp_models_dir(tmp_path):
    """Temporary directory for model files during testing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def sample_vectorizer():
    """Sample fitted TfidfVectorizer for testing."""
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    texts = [
        'action movie thriller',
        'romantic comedy',
        'sci-fi adventure',
        'drama romance',
        'comedy action'
    ]
    vectorizer.fit(texts)
    return vectorizer


@pytest.fixture
def sample_mlb(sample_genres_list):
    """Sample fitted MultiLabelBinarizer for testing."""
    mlb = MultiLabelBinarizer()
    mlb.fit(sample_genres_list)
    return mlb


@pytest.fixture
def sample_model():
    """Sample trained model for testing."""
    base_estimator = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=100,
        random_state=42
    )
    model = OneVsRestClassifier(base_estimator)
    
    # Create dummy training data
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=(10, 3))
    
    model.fit(X, y)
    return model

