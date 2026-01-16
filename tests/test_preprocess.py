"""Tests for preprocessing functions."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest
from scipy.sparse import csr_matrix

from descriptions.modeling.preprocess import (
    build_preprocessor,
    _preprocess_genres,
    _generate_targets,
    _generate_descriptions,
    save_preprocessors,
    load_preprocessors,
)


class TestBuildPreprocessor:
    """Tests for build_preprocessor function."""
    
    def test_returns_correct_types(self):
        """Build_preprocessor returns correct types."""
        vectorizer, mlb, normalizer, kbest = build_preprocessor()
        assert isinstance(vectorizer, TfidfVectorizer)
        assert isinstance(mlb, MultiLabelBinarizer)
        assert isinstance(normalizer, Normalizer)
        assert isinstance(kbest, SelectKBest)
    
    def test_tfidf_settings(self):
        """TF-IDF has correct settings."""
        vectorizer, _, _, _ = build_preprocessor()
        assert vectorizer.max_features == 10000
        assert vectorizer.sublinear_tf is True
        assert vectorizer.ngram_range == (1, 3)
        assert vectorizer.stop_words == 'english'


class TestPreprocessGenres:
    """Tests for _preprocess_genres function."""
    
    def test_splits_correctly(self):
        """Genre strings are split correctly."""
        data = pd.DataFrame({
            'genre': ['Action, Adventure', 'Drama, Romance, Comedy']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Adventure']
        assert len(result['genre'].iloc[1]) == 3
        assert all(genre in result['genre'].iloc[1] for genre in ['Drama', 'Romance', 'Comedy'])
    
    def test_handles_empty_strings(self):
        """Empty genre strings are handled."""
        data = pd.DataFrame({
            'genre': ['Action, Adventure', '', 'Drama']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[1] == []
    
    def test_handles_missing_values(self):
        """Missing values are handled."""
        data = pd.DataFrame({
            'genre': ['Action', None, 'Drama, Comedy']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[1] == []
    
    def test_removes_duplicates(self):
        """Duplicate genres are removed."""
        data = pd.DataFrame({
            'genre': ['Action, Action, Adventure', 'Drama, Drama']
        })
        result = _preprocess_genres(data)
        assert len(result['genre'].iloc[0]) == 2
        assert len(result['genre'].iloc[1]) == 1
    
    def test_sorts_genres(self):
        """Genres are sorted."""
        data = pd.DataFrame({
            'genre': ['Zebra, Action, Drama']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Drama', 'Zebra']
    
    def test_strips_whitespace(self):
        """Whitespace is stripped from genres."""
        data = pd.DataFrame({
            'genre': [' Action , Adventure ', ' Drama ']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Adventure']
        assert result['genre'].iloc[1] == ['Drama']


class TestGenerateTargets:
    """Tests for _generate_targets function."""
    
    def test_creates_binary_matrix(self, sample_movie_data):
        """Targets are created as binary matrix."""
        y, mlb, _ = _generate_targets(sample_movie_data)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2
        assert y.dtype in [np.int64, np.int32, np.int8]
        assert np.all((y == 0) | (y == 1))
    
    def test_with_fitted_mlb(self, sample_movie_data, sample_mlb):
        """Targets can be generated with pre-fitted MLB."""
        y, mlb, _ = _generate_targets(sample_movie_data, mlb=sample_mlb)
        assert y.shape[1] == len(sample_mlb.classes_)
        assert mlb is sample_mlb
    
    def test_shape_matches_data(self, sample_movie_data):
        """Target shape matches data shape."""
        y, mlb, _ = _generate_targets(sample_movie_data)
        assert y.shape[0] == len(sample_movie_data)
        assert y.shape[1] == len(mlb.classes_)


class TestGenerateDescriptions:
    """Tests for _generate_descriptions function."""
    
    def test_creates_sparse_matrix(self, sample_movie_data):
        """Descriptions are converted to sparse matrix."""
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(sample_movie_data, vectorizer=test_vectorizer)
        assert isinstance(X, csr_matrix)
        assert X.shape[0] == len(sample_movie_data)
    
    def test_with_fitted_vectorizer(self, sample_movie_data, sample_vectorizer):
        """Descriptions can be generated with pre-fitted vectorizer."""
        X, vectorizer = _generate_descriptions(sample_movie_data, vectorizer=sample_vectorizer)
        assert vectorizer is sample_vectorizer
        assert X.shape[1] <= sample_vectorizer.max_features
        assert X.shape[1] > 0
    
    def test_handles_empty_descriptions(self):
        """Empty descriptions are handled."""
        data = pd.DataFrame({
            'description': ['A movie', '', 'Another movie']
        })
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(data, vectorizer=test_vectorizer)
        assert X.shape[0] == 3
    
    def test_handles_missing_values(self):
        """Missing descriptions are handled."""
        data = pd.DataFrame({
            'description': ['A movie', None, 'Another movie']
        })
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(data, vectorizer=test_vectorizer)
        assert X.shape[0] == 3


class TestSaveLoadPreprocessors:
    """Tests for saving and loading preprocessors."""
    
    def test_save_creates_files(self, sample_vectorizer, sample_mlb, tmp_path, monkeypatch):
        """Save_preprocessors creates model files."""
        import descriptions.modeling.preprocess as preprocess_module
        import descriptions.modeling.model as model_module
        
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        original_model_models_dir = model_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_module.MODELS_DIR = test_models_dir
        model_module.MODELS_DIR = test_models_dir
        
        try:
            normalizer = Normalizer(norm='l2')
            kbest = SelectKBest(k=100)
            
            save_preprocessors(sample_vectorizer, sample_mlb, normalizer, kbest)
            
            expected_files = [
                'tfidf_vectorizer.joblib',
                'genre_binarizer.joblib',
                'normalizer.joblib',
                'feature_selector.joblib'
            ]
            assert all((test_models_dir / f).exists() for f in expected_files)
        finally:
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
            model_module.MODELS_DIR = original_model_models_dir
    
    def test_load_returns_correct_types(self, sample_vectorizer, sample_mlb, tmp_path, monkeypatch):
        """Load_preprocessors returns correct types."""
        import descriptions.modeling.preprocess as preprocess_module
        import descriptions.modeling.model as model_module
        
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        original_model_models_dir = model_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_module.MODELS_DIR = test_models_dir
        model_module.MODELS_DIR = test_models_dir
        
        try:
            normalizer = Normalizer(norm='l2')
            kbest = SelectKBest(k=100)
            
            save_preprocessors(sample_vectorizer, sample_mlb, normalizer, kbest)
            loaded_vectorizer, loaded_mlb, loaded_normalizer, loaded_kbest = load_preprocessors()
            
            assert isinstance(loaded_vectorizer, TfidfVectorizer)
            assert isinstance(loaded_mlb, MultiLabelBinarizer)
            assert isinstance(loaded_normalizer, Normalizer)
            assert isinstance(loaded_kbest, SelectKBest)
            assert loaded_vectorizer.max_features == sample_vectorizer.max_features
            assert len(loaded_mlb.classes_) == len(sample_mlb.classes_)
        finally:
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
            model_module.MODELS_DIR = original_model_models_dir
