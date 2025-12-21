"""Tests for API service layer."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services import PredictionService


class TestPredictionServiceTopK:
    """Tests for top-k selection in PredictionService."""
    
    def test_predict_default_top_k(self):
        """Test that predict uses default top_k=3."""
        service = PredictionService()
        
        # Mock the model and preprocessors
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [0.8, 0.6, 0.4, 0.3, 0.2]
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_mlb = Mock()
        mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama')]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
        
        service.model = mock_model
        service.vectorizer = mock_vectorizer
        service.mlb = mock_mlb
        service.normalizer = mock_normalizer
        service.feature_selector = mock_feature_selector
        service._is_loaded = True
        
        result = service.predict(['A test description'])
        
        assert isinstance(result, list)
        assert len(result) == 1
        # Verify decision_function was called (for top-k selection)
        mock_model.decision_function.assert_called_once()
    
    def test_predict_custom_top_k(self):
        """Test that predict respects custom top_k parameter."""
        service = PredictionService()
        
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [0.9, 0.8, 0.7, 0.6, 0.5]
        ])
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_mlb = Mock()
        mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama', 'Comedy', 'Romance')]
        
        mock_normalizer = Mock()
        mock_normalizer.transform.return_value = np.array([[1, 2, 3]])
        
        mock_feature_selector = Mock()
        mock_feature_selector.transform.return_value = np.array([[1, 2, 3]])
        
        service.model = mock_model
        service.vectorizer = mock_vectorizer
        service.mlb = mock_mlb
        service.normalizer = mock_normalizer
        service.feature_selector = mock_feature_selector
        service._is_loaded = True
        
        result = service.predict(['A test description'], top_k=5)
        
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_predict_top_k_with_threshold(self):
        """Test that top_k selection respects threshold."""
        service = PredictionService()
        
        # Create probabilities that will be filtered by threshold
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([
            [1.4, 1.0, -0.4, -0.8]  # sigmoid: ~0.8, 0.73, 0.4, 0.31
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
        
        service.model = mock_model
        service.vectorizer = mock_vectorizer
        service.mlb = mock_mlb
        service.normalizer = mock_normalizer
        service.feature_selector = mock_feature_selector
        service._is_loaded = True
        
        result = service.predict(
            ['A test description'],
            top_k=3,
            threshold=0.5
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_predict_multiple_descriptions(self):
        """Test that predict handles multiple descriptions."""
        service = PredictionService()
        
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
        
        service.model = mock_model
        service.vectorizer = mock_vectorizer
        service.mlb = mock_mlb
        service.normalizer = mock_normalizer
        service.feature_selector = mock_feature_selector
        service._is_loaded = True
        
        result = service.predict(
            ['First description', 'Second description'],
            top_k=2
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_predict_model_not_loaded_raises_error(self):
        """Test that predict raises error when model is not loaded."""
        service = PredictionService()
        service._is_loaded = False
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict(['A test description'])
    
    def test_is_ready(self):
        """Test that is_ready returns correct status."""
        service = PredictionService()
        
        # Initially not ready
        assert service.is_ready() == False
        
        # After loading
        service._is_loaded = True
        assert service.is_ready() == True
        
        service._is_loaded = False
        assert service.is_ready() == False

