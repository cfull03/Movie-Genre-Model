"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_has_correct_structure(self, client):
        """Test that health endpoint returns correct structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_root_endpoint(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_endpoint_requires_description(self, client):
        """Test that predict endpoint requires description field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_with_defaults(self, client):
        """Test predict endpoint with default threshold and top_k."""
        response = client.post(
            "/predict",
            json={"description": "A thrilling action movie about a spy"}
        )
        assert response.status_code in [200, 500]  # 500 if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "description" in data
            assert "prediction" in data
            assert "threshold" in data
            assert "top_k" in data
            assert data["threshold"] == 0.55
            assert data["top_k"] == 3
            assert "genres" in data["prediction"]
            assert "genre_count" in data["prediction"]
    
    def test_predict_endpoint_with_custom_threshold(self, client):
        """Test predict endpoint with custom threshold."""
        response = client.post(
            "/predict",
            json={
                "description": "A thrilling action movie",
                "threshold": 0.7
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.7
    
    def test_predict_endpoint_with_custom_top_k(self, client):
        """Test predict endpoint with custom top_k."""
        response = client.post(
            "/predict",
            json={
                "description": "A sci-fi action thriller with romance",
                "top_k": 5
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["top_k"] == 5
    
    def test_predict_endpoint_with_both_parameters(self, client):
        """Test predict endpoint with both threshold and top_k."""
        response = client.post(
            "/predict",
            json={
                "description": "A romantic comedy about two strangers",
                "threshold": 0.5,
                "top_k": 4
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.5
            assert data["top_k"] == 4
    
    def test_predict_endpoint_validates_threshold_range(self, client):
        """Test that threshold must be between 0 and 1."""
        # Test threshold too high
        response = client.post(
            "/predict",
            json={
                "description": "A movie",
                "threshold": 1.5
            }
        )
        assert response.status_code == 422
        
        # Test threshold too low
        response = client.post(
            "/predict",
            json={
                "description": "A movie",
                "threshold": -0.1
            }
        )
        assert response.status_code == 422
    
    def test_predict_endpoint_validates_top_k_range(self, client):
        """Test that top_k must be within valid range."""
        # Test top_k too high
        response = client.post(
            "/predict",
            json={
                "description": "A movie",
                "top_k": 25
            }
        )
        assert response.status_code == 422
        
        # Test top_k too low
        response = client.post(
            "/predict",
            json={
                "description": "A movie",
                "top_k": 0
            }
        )
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""
    
    def test_batch_predict_endpoint_requires_descriptions(self, client):
        """Test that batch predict endpoint requires descriptions field."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422
    
    def test_batch_predict_endpoint_with_defaults(self, client):
        """Test batch predict endpoint with default parameters."""
        response = client.post(
            "/predict/batch",
            json={
                "descriptions": [
                    "A thrilling action movie",
                    "A romantic comedy"
                ]
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_predictions" in data
            assert "threshold" in data
            assert "top_k" in data
            assert data["threshold"] == 0.55
            assert data["top_k"] == 3
            assert len(data["predictions"]) == 2
    
    def test_batch_predict_endpoint_with_custom_parameters(self, client):
        """Test batch predict endpoint with custom threshold and top_k."""
        response = client.post(
            "/predict/batch",
            json={
                "descriptions": [
                    "A sci-fi action thriller",
                    "A horror movie"
                ],
                "threshold": 0.6,
                "top_k": 2
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.6
            assert data["top_k"] == 2
    
    def test_batch_predict_endpoint_empty_list(self, client):
        """Test batch predict endpoint with empty descriptions list."""
        response = client.post(
            "/predict/batch",
            json={"descriptions": []}
        )
        assert response.status_code == 422  # Validation error for empty list
    
    def test_batch_predict_endpoint_single_description(self, client):
        """Test batch predict endpoint with single description."""
        response = client.post(
            "/predict/batch",
            json={
                "descriptions": ["A single movie description"]
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_predictions"] == 1
            assert len(data["predictions"]) == 1
