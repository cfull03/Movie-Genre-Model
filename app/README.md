# Movie Genre Classification API

Flask REST API for movie genre classification.

## Quick Start

### Development Mode

```bash
# Start the API server
make api

# Or directly
python app/run.py
```

The API will be available at `http://localhost:5000`

### Production Mode

```bash
# Using gunicorn
make api-prod

# Or directly
gunicorn -w 4 -b 0.0.0.0:5000 "app.app:create_app()"
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model/info
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
    "description": "A thrilling action movie about a spy who saves the world",
    "threshold": 0.55
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
    "descriptions": [
        "A thrilling action movie about a spy",
        "A romantic comedy about two people"
    ],
    "threshold": 0.55
}
```

### Metrics
```bash
GET /metrics
```

## Configuration

Set environment variables or create a `.env` file:

```bash
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False
MODEL_PATH=  # Optional: path to model file (auto-detects if not set)
DEFAULT_THRESHOLD=0.55
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

## Example Usage

### Using curl

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "A thrilling action movie"}'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"descriptions": ["Action movie", "Romantic comedy"]}'
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:5000/predict",
    json={"description": "A thrilling action movie", "threshold": 0.55}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:5000/predict/batch",
    json={
        "descriptions": ["Action movie", "Romantic comedy"],
        "threshold": 0.55
    }
)
print(response.json())
```

## Testing

```bash
# Run API tests (to be implemented)
pytest tests/test_api.py
```
