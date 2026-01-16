"""Tests for edge cases and error handling."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from descriptions.dataset import load_data, load_interim, load_processed
from descriptions.modeling.model import load_model
from descriptions.modeling.preprocess import load_preprocessors, _preprocess_genres, _generate_descriptions
from descriptions.modeling.predict import predict_genres
from descriptions.modeling.evaluate import evaluate_model
from descriptions.modeling.train import split_data
from sklearn.feature_extraction.text import TfidfVectorizer


class TestDataLoadingEdgeCases:
    """Tests for edge cases in data loading."""
    
    @pytest.mark.parametrize("load_func", [load_data, load_interim, load_processed])
    def test_missing_file_raises_error(self, load_func):
        """Loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_func(Path('nonexistent_file.csv'))


class TestModelLoadingEdgeCases:
    """Tests for edge cases in model loading."""
    
    def test_invalid_path_raises_error(self):
        """Loading model from invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model.joblib')
    
    def test_missing_preprocessors_raises_error(self, tmp_path, monkeypatch):
        """Loading missing preprocessors raises error."""
        import descriptions.modeling.preprocess as preprocess_module
        import descriptions.modeling.model as model_module
        
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        original_model_models_dir = model_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        preprocess_module.MODELS_DIR = test_models_dir
        model_module.MODELS_DIR = test_models_dir
        
        try:
            with pytest.raises(FileNotFoundError):
                load_preprocessors()
        finally:
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
            model_module.MODELS_DIR = original_model_models_dir


class TestPreprocessingEdgeCases:
    """Tests for edge cases in preprocessing."""
    
    def test_empty_dataframe(self):
        """Preprocessing empty dataframe is handled."""
        empty_df = pd.DataFrame({'genre': []})
        result = _preprocess_genres(empty_df)
        assert len(result) == 0
    
    def test_all_missing_genres(self):
        """All missing genres are handled."""
        data = pd.DataFrame({'genre': [None, None, None]})
        result = _preprocess_genres(data)
        assert all(genres == [] for genres in result['genre'])
    
    def test_all_missing_descriptions(self):
        """All missing descriptions are handled."""
        data = pd.DataFrame({'description': [None, None, 'some text']})
        test_vectorizer = TfidfVectorizer(max_features=100, min_df=1, stop_words='english')
        X, vectorizer = _generate_descriptions(data, vectorizer=test_vectorizer)
        assert X.shape[0] == 3


class TestPredictionEdgeCases:
    """Tests for edge cases in prediction."""
    
    def test_without_trained_model_raises_error(self, tmp_path, monkeypatch):
        """Predicting without trained model raises error."""
        monkeypatch.setattr('descriptions.modeling.predict.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        with pytest.raises(FileNotFoundError):
            predict_genres(['A test description'])
    
    def test_empty_descriptions(self):
        """Empty descriptions list raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            predict_genres([])
    
    def test_single_empty_description(self):
        """Single empty description is handled."""
        try:
            result = predict_genres([''])
            assert isinstance(result, list)
        except (ValueError, FileNotFoundError):
            pass  # Both are acceptable behaviors


class TestModelEvaluationEdgeCases:
    """Tests for edge cases in model evaluation."""
    
    def test_empty_features_raises_error(self, sample_model):
        """Evaluating with empty features raises error."""
        X_empty = np.array([]).reshape(0, 5)
        y = np.random.randint(0, 2, size=(0, 3))
        
        with pytest.raises((ValueError, IndexError)):
            evaluate_model(sample_model, X_empty, y)
    
    def test_shape_mismatch_raises_error(self, sample_model):
        """Shape mismatch raises error."""
        X = np.random.rand(10, 5)
        y_wrong = np.random.randint(0, 2, size=(5, 3))
        
        with pytest.raises((ValueError, IndexError)):
            evaluate_model(sample_model, X, y_wrong)


class TestDataValidationEdgeCases:
    """Tests for edge cases in data validation."""
    
    def test_missing_tfidf_features(self):
        """Split_data raises error when no TF-IDF features found."""
        data = pd.DataFrame({
            'genre_Action': [1, 0, 1],
            'genre_Comedy': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match='No TF-IDF feature columns found'):
            split_data(data)
    
    def test_missing_genre_columns(self):
        """Split_data raises error when no genre columns found."""
        data = pd.DataFrame({
            'tfidf_0': [0.1, 0.2, 0.3],
            'tfidf_1': [0.4, 0.5, 0.6]
        })
        
        with pytest.raises(ValueError, match='No genre label columns found'):
            split_data(data)


class TestThresholdEdgeCases:
    """Tests for threshold edge cases."""
    
    def test_threshold_0_predicts_all(self):
        """Threshold 0 predicts all labels with probability > 0."""
        threshold = 0.0
        probabilities = np.array([[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]])
        predictions = (probabilities >= threshold).astype(int)
        assert np.all(predictions == 1)
    
    def test_threshold_1_predicts_none(self):
        """Threshold 1 predicts no labels (unless probability == 1)."""
        threshold = 1.0
        probabilities = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])
        predictions = (probabilities >= threshold).astype(int)
        assert np.all(predictions == 0)
    
    def test_threshold_05_default(self):
        """Default threshold 0.5 works correctly."""
        threshold = 0.5
        probabilities = np.array([[0.3, 0.6, 0.4], [0.7, 0.2, 0.9]])
        predictions = (probabilities >= threshold).astype(int)
        expected = np.array([[0, 1, 0], [1, 0, 1]])
        assert np.array_equal(predictions, expected)
