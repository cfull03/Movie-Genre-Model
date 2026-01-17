"""Tests for prediction functionality."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch

from descriptions.modeling.predict import predict_genres


@pytest.fixture
def mock_model_and_preprocessors():
    """Create mocked model and preprocessors."""
    mock_model = Mock()
    mock_model.decision_function.return_value = np.array([[0.8, 0.6, 0.4, 0.3, 0.2]])
    
    mock_vectorizer = Mock()
    mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
    
    mock_mlb = Mock()
    mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama')]
    # Set classes_ to have enough elements for all test cases (at least 6 for test_custom_top_k)
    mock_mlb.classes_ = ['Action', 'Thriller', 'Drama', 'Comedy', 'Romance', 'Horror']
    
    mock_normalizer = Mock()
    mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
    
    mock_feature_selector = Mock()
    mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
    
    return mock_model, (mock_vectorizer, mock_mlb, mock_normalizer, mock_feature_selector)


class TestPredictGenresTopK:
    """Tests for top-k selection in predict_genres."""
    
    @patch('descriptions.modeling.predict.load_per_label_thresholds')
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_default_top_k(self, mock_load_preprocessors, mock_load_model, mock_load_thresholds, mock_model_and_preprocessors):
        """Predict_genres uses default top_k=3."""
        mock_model, preprocessors = mock_model_and_preprocessors
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = preprocessors
        mock_load_thresholds.return_value = None
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            result = predict_genres(['A test description'])
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_per_label_thresholds')
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_custom_top_k(self, mock_load_preprocessors, mock_load_model, mock_load_thresholds, mock_model_and_preprocessors):
        """Predict_genres respects custom top_k parameter."""
        mock_model, preprocessors = mock_model_and_preprocessors
        mock_model.decision_function.return_value = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = preprocessors
        mock_load_thresholds.return_value = None
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            result = predict_genres(['A test description'], top_k=5)
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_per_label_thresholds')
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_top_k_with_threshold(self, mock_load_preprocessors, mock_load_model, mock_load_thresholds, mock_model_and_preprocessors):
        """Top_k selection respects threshold."""
        mock_model, preprocessors = mock_model_and_preprocessors
        mock_model.decision_function.return_value = np.array([[1.4, 1.0, -0.4, -0.8, -1.2]])
        preprocessors[1].inverse_transform.return_value = [('Action', 'Thriller')]
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = preprocessors
        mock_load_thresholds.return_value = None
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            result = predict_genres(['A test description'], top_k=3, threshold=0.5)
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_per_label_thresholds')
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_multiple_descriptions(self, mock_load_preprocessors, mock_load_model, mock_load_thresholds, mock_model_and_preprocessors):
        """Predict_genres handles multiple descriptions."""
        mock_model, preprocessors = mock_model_and_preprocessors
        mock_model.decision_function.return_value = np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3]
        ])
        preprocessors[0].transform.return_value = np.array([[1, 2], [3, 4]])
        preprocessors[2].transform.return_value = np.array([[1, 2], [3, 4]])
        preprocessors[3].transform.return_value = np.array([[1, 2], [3, 4]])
        preprocessors[1].inverse_transform.return_value = [
            ('Action', 'Thriller'),
            ('Comedy', 'Romance')
        ]
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = preprocessors
        mock_load_thresholds.return_value = None
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            result = predict_genres(['First description', 'Second description'], top_k=2)
            assert isinstance(result, list)
            assert len(result) == 2


class TestPredictGenresEdgeCases:
    """Tests for edge cases in predict_genres."""
    
    def test_empty_list_raises_error(self):
        """Empty descriptions list raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            predict_genres([])
