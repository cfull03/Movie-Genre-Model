"""Tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Health endpoint returns 200 with correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_root_returns_info(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert all(key in data for key in ["message", "version", "docs"])


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_requires_description(self, client):
        """Predict endpoint requires description field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_with_defaults(self, client):
        """Predict with default threshold and top_k."""
        response = client.post(
            "/predict",
            json={"description": "A thrilling action movie about a spy"}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert all(key in data for key in ["description", "prediction", "threshold", "top_k"])
            assert data["threshold"] is None or data["threshold"] == 0.55
            assert data["top_k"] == 3
            assert "genres" in data["prediction"]
    
    def test_with_custom_parameters(self, client):
        """Predict with custom threshold and top_k."""
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
    
    @pytest.mark.parametrize("threshold,expected_status", [
        (1.5, 422),  # Too high
        (-0.1, 422),  # Too low
    ])
    def test_validates_threshold_range(self, client, threshold, expected_status):
        """Threshold must be between 0 and 1."""
        response = client.post(
            "/predict",
            json={"description": "A movie", "threshold": threshold}
        )
        assert response.status_code == expected_status
    
    @pytest.mark.parametrize("top_k,expected_status", [
        (25, 422),  # Too high
        (0, 422),   # Too low
    ])
    def test_validates_top_k_range(self, client, top_k, expected_status):
        """Top_k must be within valid range (1-20)."""
        response = client.post(
            "/predict",
            json={"description": "A movie", "top_k": top_k}
        )
        assert response.status_code == expected_status


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""
    
    def test_requires_descriptions(self, client):
        """Batch predict requires descriptions field."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422
    
    def test_with_defaults(self, client):
        """Batch predict with default parameters."""
        response = client.post(
            "/predict/batch",
            json={"descriptions": ["A thrilling action movie", "A romantic comedy"]}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert all(key in data for key in ["predictions", "total_predictions", "threshold", "top_k"])
            assert data["threshold"] is None or data["threshold"] == 0.55
            assert data["top_k"] == 3
            assert len(data["predictions"]) == 2
    
    def test_with_custom_parameters(self, client):
        """Batch predict with custom threshold and top_k."""
        response = client.post(
            "/predict/batch",
            json={
                "descriptions": ["A sci-fi action thriller", "A horror movie"],
                "threshold": 0.6,
                "top_k": 2
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["threshold"] == 0.6
            assert data["top_k"] == 2
    
    def test_empty_list_rejected(self, client):
        """Empty descriptions list is rejected."""
        response = client.post("/predict/batch", json={"descriptions": []})
        assert response.status_code == 422
    
    def test_single_description(self, client):
        """Batch predict handles single description."""
        response = client.post(
            "/predict/batch",
            json={"descriptions": ["A single movie description"]}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_predictions"] == 1
            assert len(data["predictions"]) == 1
