"""Tests for model building and utilities."""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from descriptions.modeling.model import (
    build_model,
    get_params,
    get_model_name,
    save_model,
    load_model,
)


class TestBuildModel:
    """Tests for build_model function."""
    
    def test_returns_ovr_classifier(self):
        """Build_model returns OneVsRestClassifier."""
        model = build_model()
        assert isinstance(model, OneVsRestClassifier)
    
    def test_default_parameters(self):
        """Build_model uses correct default parameters."""
        model = build_model()
        base_estimator = model.estimator
        assert isinstance(base_estimator, LinearSVC)
        assert base_estimator.C == 0.1
        assert base_estimator.penalty == 'l2'
        assert base_estimator.loss == 'squared_hinge'
        assert base_estimator.max_iter == 1000
        assert base_estimator.class_weight == 'balanced'
        assert base_estimator.random_state == 42
    
    def test_custom_parameters(self):
        """Build_model accepts custom parameters."""
        model = build_model(C=0.5, penalty='l2', loss='hinge', max_iter=500)
        base_estimator = model.estimator
        assert base_estimator.C == 0.5
        assert base_estimator.penalty == 'l2'
        assert base_estimator.loss == 'hinge'
        assert base_estimator.max_iter == 500
    
    def test_can_fit_and_predict(self):
        """Built model can be trained and used for prediction."""
        model = build_model()
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, size=(10, 3))
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == y.shape
        assert np.all((predictions == 0) | (predictions == 1))


class TestGetParams:
    """Tests for get_params function."""
    
    def test_extracts_correctly(self):
        """Get_params extracts model parameters."""
        model = build_model(C=0.5, penalty='l2', loss='squared_hinge', max_iter=1500)
        params = get_params(model)
        
        assert all(key in params for key in ['C', 'penalty', 'loss', 'max_iter'])
        assert params['C'] == 0.5
        assert params['penalty'] == 'l2'
        assert params['loss'] == 'squared_hinge'
        assert params['max_iter'] == 1500
    
    def test_with_pipeline(self):
        """Get_params works with Pipeline objects."""
        from descriptions.modeling.preprocess import build_preprocessor
        
        vectorizer, _, _, _ = build_preprocessor()
        model = build_model()
        pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])
        
        params = get_params(pipeline)
        assert all(key in params for key in ['C', 'penalty'])


class TestGetModelName:
    """Tests for get_model_name function."""
    
    def test_returns_lowercase(self):
        """Get_model_name returns lowercase string."""
        model = build_model()
        name = get_model_name(model)
        assert isinstance(name, str)
        assert name.islower()
        assert 'linearsvc' in name
    
    def test_with_pipeline(self):
        """Get_model_name works with Pipeline objects."""
        from descriptions.modeling.preprocess import build_preprocessor
        
        vectorizer, _, _, _ = build_preprocessor()
        model = build_model()
        pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])
        
        name = get_model_name(pipeline)
        assert isinstance(name, str)
        assert 'linearsvc' in name


class TestSaveLoadModel:
    """Tests for save_model and load_model functions."""
    
    def test_save_creates_file(self, sample_model, tmp_path, monkeypatch):
        """Save_model creates a model file."""
        monkeypatch.setattr('descriptions.modeling.model.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        save_model(sample_model, 'test_model')
        assert (tmp_path / 'models' / 'test_model.joblib').exists()
    
    def test_save_with_path_object(self, sample_model, tmp_path, monkeypatch):
        """Save_model works with Path objects."""
        monkeypatch.setattr('descriptions.modeling.model.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        model_path = tmp_path / 'models' / 'custom_model.joblib'
        save_model(sample_model, model_path)
        assert model_path.exists()
    
    def test_load_returns_correct_type(self, sample_model, tmp_path, monkeypatch):
        """Load_model returns the correct model type."""
        monkeypatch.setattr('descriptions.modeling.model.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        save_model(sample_model, 'test_model')
        loaded_model = load_model('test_model')
        
        assert isinstance(loaded_model, OneVsRestClassifier)
        assert isinstance(loaded_model.estimator, LinearSVC)
    
    def test_load_file_not_found(self, tmp_path, monkeypatch):
        """Load_model raises FileNotFoundError for missing file."""
        monkeypatch.setattr('descriptions.modeling.model.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        with pytest.raises(FileNotFoundError):
            load_model('nonexistent_model')
    
    @pytest.mark.parametrize("invalid_name", ['', '   '])
    def test_load_invalid_name_raises_error(self, invalid_name):
        """Load_model raises ValueError for invalid name."""
        with pytest.raises(ValueError):
            load_model(invalid_name)
    
    def test_save_and_load_preserves_parameters(self, tmp_path, monkeypatch):
        """Saved and loaded model preserves parameters."""
        monkeypatch.setattr('descriptions.modeling.model.MODELS_DIR', tmp_path / 'models')
        (tmp_path / 'models').mkdir()
        
        original_model = build_model(C=0.5, penalty='l2', loss='hinge')
        save_model(original_model, 'test_model')
        loaded_model = load_model('test_model')
        
        assert loaded_model.estimator.C == 0.5
        assert loaded_model.estimator.penalty == 'l2'
        assert loaded_model.estimator.loss == 'hinge'
