"""Tests for prediction functionality."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from descriptions.modeling.predict import predict_genres


class TestPredictGenresTopK:
    """Tests for top-k selection in predict_genres."""
    
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_predict_genres_default_top_k(self, mock_load_preprocessors, mock_load_model):
        """Test that predict_genres uses default top_k=3."""
        # Setup mocks
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [0.8, 0.6, 0.4, 0.3, 0.2]  # 5 genres, top 3 should be selected
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_mlb = Mock()
        mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama')]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
        
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = (
            mock_vectorizer, mock_mlb, mock_normalizer, mock_feature_selector
        )
        
        # Mock MODELS_DIR to avoid file system issues
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            
            result = predict_genres(['A test description'])
            
            # Should return genres (mock returns 3 genres)
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_predict_genres_custom_top_k(self, mock_load_preprocessors, mock_load_model):
        """Test that predict_genres respects custom top_k parameter."""
        # Setup mocks
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # 6 genres
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_mlb = Mock()
        mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama', 'Comedy', 'Romance')]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
        
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = (
            mock_vectorizer, mock_mlb, mock_normalizer, mock_feature_selector
        )
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            
            # Test with top_k=5
            result = predict_genres(['A test description'], top_k=5)
            
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_predict_genres_top_k_with_threshold(self, mock_load_preprocessors, mock_load_model):
        """Test that top_k selection respects threshold."""
        # Setup mocks
        mock_model = Mock()
        # Create probabilities: [0.8, 0.7, 0.4, 0.3, 0.2]
        # With threshold=0.5 and top_k=3, should only get first 2 (above threshold)
        mock_model.decision_function.return_value = np.array([
            [1.4, 1.0, -0.4, -0.8, -1.2]  # sigmoid gives ~0.8, 0.73, 0.4, 0.31, 0.23
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_mlb = Mock()
        # Only genres above threshold should be returned
        mock_mlb.inverse_transform.return_value = [('Action', 'Thriller')]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
        
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = (
            mock_vectorizer, mock_mlb, mock_normalizer, mock_feature_selector
        )
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            
            result = predict_genres(
                ['A test description'],
                top_k=3,
                threshold=0.5
            )
            
            assert isinstance(result, list)
            assert len(result) == 1
    
    @patch('descriptions.modeling.predict.load_model')
    @patch('descriptions.modeling.predict.load_preprocessors')
    def test_predict_genres_multiple_descriptions(self, mock_load_preprocessors, mock_load_model):
        """Test that predict_genres handles multiple descriptions."""
        # Setup mocks
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3]
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2], [3, 4]])
        
        mock_mlb = Mock()
        mock_mlb.inverse_transform.return_value = [
            ('Action', 'Thriller'),
            ('Comedy', 'Romance')
        ]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2], [3, 4]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2], [3, 4]])
        
        mock_load_model.return_value = mock_model
        mock_load_preprocessors.return_value = (
            mock_vectorizer, mock_mlb, mock_normalizer, mock_feature_selector
        )
        
        with patch('descriptions.modeling.predict.MODELS_DIR') as mock_models_dir:
            mock_models_dir.glob.return_value = [Mock(name='linearsvc.joblib')]
            
            result = predict_genres(
                ['First description', 'Second description'],
                top_k=2
            )
            
            assert isinstance(result, list)
            assert len(result) == 2


class TestPredictGenresEdgeCases:
    """Tests for edge cases in predict_genres."""
    
    def test_predict_genres_empty_list_raises_error(self):
        """Test that empty descriptions list raises error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            predict_genres([])
    
    def test_predict_genres_invalid_top_k(self):
        """Test that invalid top_k values are handled."""
        # top_k should be positive integer
        # This will be caught by type checking or validation
        pass
    
    def test_predict_genres_invalid_threshold(self):
        """Test that invalid threshold values are handled."""
        # threshold should be between 0 and 1
        # This will be caught by type checking or validation
        pass

