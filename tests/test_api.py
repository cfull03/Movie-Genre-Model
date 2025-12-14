from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0", "model_loaded": True}

def test_predict_endpoint(client):
    response = client.post("/predict", json={"description": "A thrilling action movie"})
    assert response.status_code == 200
    assert response.json() == {"genres": ["Action", "Thriller"], "genre_count": 2}

def test_batch_predict_endpoint(client):
    response = client.post("/batch-predict", json={"descriptions": ["A thrilling action movie", "A romantic comedy about two strangers"]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [{"description": "A thrilling action movie", "genres": ["Action", "Thriller"], "genre_count": 2}, {"description": "A romantic comedy about two strangers", "genres": ["Comedy", "Romance"], "genre_count": 2}], "total_predictions": 2, "threshold": 0.55}