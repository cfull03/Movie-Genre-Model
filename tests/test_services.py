"""Tests for API service layer."""
from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock
from scipy.sparse import csr_matrix

from app.services import PredictionService


@pytest.fixture
def mock_service_components():
    """Create mocked service components for testing."""
    mock_model = Mock()
    mock_model.decision_function.return_value = np.array([[0.8, 0.6, 0.4, 0.3, 0.2]])
    
    mock_vectorizer = Mock()
    mock_vectorizer.transform.return_value = csr_matrix([[1, 2, 3]])
    
    mock_mlb = Mock()
    mock_mlb.inverse_transform.return_value = [('Action', 'Thriller', 'Drama')]
    mock_mlb.classes_ = ['Action', 'Thriller', 'Drama', 'Comedy', 'Romance']
    
    mock_normalizer = Mock()
    mock_normalizer.transform.return_value = csr_matrix([[1, 2, 3]])
    
    mock_feature_selector = Mock()
    mock_feature_selector.transform.return_value = csr_matrix([[1, 2, 3]])
    
    return {
        'model': mock_model,
        'vectorizer': mock_vectorizer,
        'mlb': mock_mlb,
        'normalizer': mock_normalizer,
        'feature_selector': mock_feature_selector,
    }


@pytest.fixture
def configured_service(mock_service_components):
    """Service with mocked components configured."""
    service = PredictionService()
    service.model = mock_service_components['model']
    service.vectorizer = mock_service_components['vectorizer']
    service.mlb = mock_service_components['mlb']
    service.normalizer = mock_service_components['normalizer']
    service.feature_selector = mock_service_components['feature_selector']
    service._is_loaded = True
    return service


class TestPredictionServiceTopK:
    """Tests for top-k selection in PredictionService."""
    
    def test_default_top_k(self, configured_service):
        """Predict uses default top_k=3."""
        result = configured_service.predict(['A test description'])
        assert isinstance(result, list)
        assert len(result) == 1
        configured_service.model.decision_function.assert_called_once()
    
    def test_custom_top_k(self, mock_service_components):
        """Predict respects custom top_k parameter."""
        service = PredictionService()
        for key, value in mock_service_components.items():
            setattr(service, key, value)
        service._is_loaded = True
        
        result = service.predict(['A test description'], top_k=5)
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_top_k_with_threshold(self, mock_service_components):
        """Top_k selection respects threshold."""
        mock_service_components['model'].decision_function.return_value = np.array([
            [1.4, 1.0, -0.4, -0.8]  # sigmoid: ~0.8, 0.73, 0.4, 0.31
        ])
        mock_service_components['mlb'].inverse_transform.return_value = [('Action', 'Thriller')]
        
        service = PredictionService()
        for key, value in mock_service_components.items():
            setattr(service, key, value)
        service._is_loaded = True
        
        result = service.predict(['A test description'], top_k=3, threshold=0.5)
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_multiple_descriptions(self, mock_service_components):
        """Predict handles multiple descriptions."""
        mock_service_components['model'].decision_function.return_value = np.array([
            [0.8, 0.6, 0.4],
            [0.7, 0.5, 0.3]
        ])
        mock_service_components['vectorizer'].transform.return_value = csr_matrix([[1, 2], [3, 4]])
        mock_service_components['normalizer'].transform.return_value = csr_matrix([[1, 2], [3, 4]])
        mock_service_components['feature_selector'].transform.return_value = csr_matrix([[1, 2], [3, 4]])
        mock_service_components['mlb'].inverse_transform.return_value = [
            ('Action', 'Thriller'),
            ('Comedy', 'Romance')
        ]
        
        service = PredictionService()
        for key, value in mock_service_components.items():
            setattr(service, key, value)
        service._is_loaded = True
        
        result = service.predict(['First description', 'Second description'], top_k=2)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_model_not_loaded_raises_error(self, tmp_path, monkeypatch):
        """Predict raises error when model is not loaded."""
        import descriptions.modeling.model as model_module
        import descriptions.modeling.preprocess as preprocess_module
        
        original_models_dir = model_module.MODELS_DIR
        original_preprocess_models_dir = preprocess_module.MODELS_DIR
        
        test_models_dir = tmp_path / 'models'
        test_models_dir.mkdir(parents=True, exist_ok=True)
        
        model_module.MODELS_DIR = test_models_dir
        preprocess_module.MODELS_DIR = test_models_dir
        
        try:
            service = PredictionService()
            service._is_loaded = False
            monkeypatch.setattr('app.services.MODELS_DIR', test_models_dir)
            
            with pytest.raises((RuntimeError, FileNotFoundError)):
                service.predict(['A test description'])
        finally:
            model_module.MODELS_DIR = original_models_dir
            preprocess_module.MODELS_DIR = original_preprocess_models_dir
    
    def test_is_ready(self):
        """Is_ready returns correct status."""
        service = PredictionService()
        assert service.is_ready() is False
        
        service._is_loaded = True
        assert service.is_ready() is True
        
        service._is_loaded = False
        assert service.is_ready() is False
