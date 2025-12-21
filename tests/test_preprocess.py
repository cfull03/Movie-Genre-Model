"""Tests for preprocessing functions."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from descriptions.modeling.preprocess import (
    build_preprocessor,
    _preprocess_genres,
    _generate_targets,
    _generate_descriptions,
    save_preprocessors,
    load_preprocessors,
)
from descriptions.config import MODELS_DIR


class TestBuildPreprocessor:
    """Tests for build_preprocessor function."""
    
    def test_build_preprocessor_returns_correct_types(self):
        """Test that build_preprocessor returns correct types."""
        from sklearn.preprocessing import Normalizer
        from sklearn.feature_selection import SelectKBest
        
        vectorizer, mlb, normalizer, kbest = build_preprocessor()
        assert isinstance(vectorizer, TfidfVectorizer)
        assert isinstance(mlb, MultiLabelBinarizer)
        assert isinstance(normalizer, Normalizer)
        assert isinstance(kbest, SelectKBest)
    
    def test_tfidf_max_features_setting(self):
        """Test that TF-IDF has correct max_features."""
        vectorizer, _, _, _ = build_preprocessor()
        assert vectorizer.max_features == 10000
    
    def test_tfidf_sublinear_tf_enabled(self):
        """Test that sublinear_tf is enabled."""
        vectorizer, _, _, _ = build_preprocessor()
        assert vectorizer.sublinear_tf == True
    
    def test_tfidf_ngram_range(self):
        """Test that ngram_range is set correctly."""
        vectorizer, _, _, _ = build_preprocessor()
        assert vectorizer.ngram_range == (1, 3)
    
    def test_tfidf_stop_words(self):
        """Test that stop_words are set correctly."""
        vectorizer, _, _, _ = build_preprocessor()
        assert vectorizer.stop_words == 'english'


class TestPreprocessGenres:
    """Tests for _preprocess_genres function."""
    
    def test_preprocess_genres_splits_correctly(self):
        """Test that genre strings are split correctly."""
        data = pd.DataFrame({
            'genre': ['Action, Adventure', 'Drama, Romance, Comedy']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Adventure']
        assert len(result['genre'].iloc[1]) == 3
        assert 'Drama' in result['genre'].iloc[1]
        assert 'Romance' in result['genre'].iloc[1]
        assert 'Comedy' in result['genre'].iloc[1]
    
    def test_preprocess_genres_handles_empty_strings(self):
        """Test that empty genre strings are handled."""
        data = pd.DataFrame({
            'genre': ['Action, Adventure', '', 'Drama']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[1] == []
    
    def test_preprocess_genres_handles_missing_values(self):
        """Test that missing values are handled."""
        data = pd.DataFrame({
            'genre': ['Action', None, 'Drama, Comedy']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[1] == []
    
    def test_preprocess_genres_removes_duplicates(self):
        """Test that duplicate genres are removed."""
        data = pd.DataFrame({
            'genre': ['Action, Action, Adventure', 'Drama, Drama']
        })
        result = _preprocess_genres(data)
        assert len(result['genre'].iloc[0]) == 2
        assert len(result['genre'].iloc[1]) == 1
    
    def test_preprocess_genres_sorts_genres(self):
        """Test that genres are sorted."""
        data = pd.DataFrame({
            'genre': ['Zebra, Action, Drama']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Drama', 'Zebra']
    
    def test_preprocess_genres_strips_whitespace(self):
        """Test that whitespace is stripped from genres."""
        data = pd.DataFrame({
            'genre': [' Action , Adventure ', ' Drama ']
        })
        result = _preprocess_genres(data)
        assert result['genre'].iloc[0] == ['Action', 'Adventure']
        assert result['genre'].iloc[1] == ['Drama']


class TestGenerateTargets:
    """Tests for _generate_targets function."""
    
    def test_generate_targets_creates_binary_matrix(self, sample_movie_data):
        """Test that targets are created as binary matrix."""
        y, mlb, _ = _generate_targets(sample_movie_data)
        assert isinstance(y, np.ndarray)
        assert y.ndim == 2
        assert y.dtype in [np.int64, np.int32, np.int8]
        assert np.all((y == 0) | (y == 1))
    
    def test_generate_targets_with_fitted_mlb(self, sample_movie_data, sample_mlb):
        """Test that targets can be generated with pre-fitted MLB."""
        y, mlb, _ = _generate_targets(sample_movie_data, mlb=sample_mlb)
        assert y.shape[1] == len(sample_mlb.classes_)
        assert mlb is sample_mlb
    
    def test_generate_targets_shape_matches_data(self, sample_movie_data):
        """Test that target shape matches data shape."""
        y, mlb, _ = _generate_targets(sample_movie_data)
        assert y.shape[0] == len(sample_movie_data)
        assert y.shape[1] == len(mlb.classes_)


class TestGenerateDescriptions:
    """Tests for _generate_descriptions function."""
    
    def test_generate_descriptions_creates_sparse_matrix(self, sample_movie_data):
        """Test that descriptions are converted to sparse matrix."""
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Use a vectorizer with lower min_df for small test data
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(sample_movie_data, vectorizer=test_vectorizer)
        assert isinstance(X, csr_matrix)
        assert X.shape[0] == len(sample_movie_data)
    
    def test_generate_descriptions_with_fitted_vectorizer(self, sample_movie_data, sample_vectorizer):
        """Test that descriptions can be generated with pre-fitted vectorizer."""
        X, vectorizer = _generate_descriptions(sample_movie_data, vectorizer=sample_vectorizer)
        assert vectorizer is sample_vectorizer
        # The actual number of features may be less than max_features if vocabulary is smaller
        assert X.shape[1] <= sample_vectorizer.max_features
        assert X.shape[1] > 0  # Should have at least some features
    
    def test_generate_descriptions_handles_empty_descriptions(self):
        """Test that empty descriptions are handled."""
        # Use a vectorizer with lower min_df for small test data
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        data = pd.DataFrame({
            'description': ['A movie', '', 'Another movie']
        })
        # Create a vectorizer with min_df=1 for small test data
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(data, vectorizer=test_vectorizer)
        assert X.shape[0] == 3
    
    def test_generate_descriptions_handles_missing_values(self):
        """Test that missing descriptions are handled."""
        # Use a vectorizer with lower min_df for small test data
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        data = pd.DataFrame({
            'description': ['A movie', None, 'Another movie']
        })
        # Create a vectorizer with min_df=1 for small test data
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(data, vectorizer=test_vectorizer)
        assert X.shape[0] == 3


class TestSaveLoadPreprocessors:
    """Tests for saving and loading preprocessors."""
    
    def test_save_preprocessors_creates_files(self, sample_vectorizer, sample_mlb, tmp_path, monkeypatch):
        """Test that save_preprocessors creates model files."""
        from sklearn.preprocessing import Normalizer
        from sklearn.feature_selection import SelectKBest
        
        # Temporarily change MODELS_DIR in both modules
        import descriptions.modeling.preprocess as preprocess_module
        import descriptions.modeling.model as model_module
        
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        original_model_models_dir = model_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_module.MODELS_DIR = test_models_dir
        model_module.MODELS_DIR = test_models_dir
        
        try:
            # Create normalizer and feature selector for testing
            normalizer = Normalizer(norm='l2')
            kbest = SelectKBest(k=100)
            
            save_preprocessors(sample_vectorizer, sample_mlb, normalizer, kbest)
            
            assert (test_models_dir / 'tfidf_vectorizer.joblib').exists()
            assert (test_models_dir / 'genre_binarizer.joblib').exists()
            assert (test_models_dir / 'normalizer.joblib').exists()
            assert (test_models_dir / 'feature_selector.joblib').exists()
        finally:
            # Restore original MODELS_DIR
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
            model_module.MODELS_DIR = original_model_models_dir
    
    def test_load_preprocessors_returns_correct_types(self, sample_vectorizer, sample_mlb, tmp_path, monkeypatch):
        """Test that load_preprocessors returns correct types."""
        from sklearn.preprocessing import Normalizer
        from sklearn.feature_selection import SelectKBest
        
        # Temporarily change MODELS_DIR in both modules
        import descriptions.modeling.preprocess as preprocess_module
        import descriptions.modeling.model as model_module
        
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        original_model_models_dir = model_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_module.MODELS_DIR = test_models_dir
        model_module.MODELS_DIR = test_models_dir
        
        try:
            # Create normalizer and feature selector for testing
            normalizer = Normalizer(norm='l2')
            kbest = SelectKBest(k=100)
            
            # Save then load
            save_preprocessors(sample_vectorizer, sample_mlb, normalizer, kbest)
            loaded_vectorizer, loaded_mlb, loaded_normalizer, loaded_kbest = load_preprocessors()
            
            assert isinstance(loaded_vectorizer, TfidfVectorizer)
            assert isinstance(loaded_mlb, MultiLabelBinarizer)
            assert isinstance(loaded_normalizer, Normalizer)
            assert isinstance(loaded_kbest, SelectKBest)
            assert loaded_vectorizer.max_features == sample_vectorizer.max_features
            assert len(loaded_mlb.classes_) == len(sample_mlb.classes_)
        finally:
            # Restore original MODELS_DIR
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
            model_module.MODELS_DIR = original_model_models_dir

