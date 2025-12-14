# Movie Genre Prediction API

FastAPI application for predicting movie genres from text descriptions.

## Features

- **RESTful API** with automatic OpenAPI documentation
- **Single prediction** endpoint for individual descriptions
- **Batch prediction** endpoint for multiple descriptions
- **Health check** endpoint for monitoring
- **Type validation** using Pydantic schemas
- **Async support** for better performance

## Quick Start

### Development Server

```bash
make api
# or
python app/run.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Production Server

```bash
make api-prod
# or
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint. Returns API status and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

### `POST /predict`
Predict genres for a single movie description.

**Request:**
```json
{
  "description": "A thrilling action movie about a secret agent",
  "threshold": 0.55,
  "model_path": null
}
```

**Response:**
```json
{
  "description": "A thrilling action movie about a secret agent",
  "prediction": {
    "genres": ["Action", "Thriller"],
    "genre_count": 2
  },
  "threshold": 0.55
}
```

### `POST /predict/batch`
Predict genres for multiple movie descriptions.

**Request:**
```json
{
  "descriptions": [
    "A thrilling action movie",
    "A romantic comedy about two strangers"
  ],
  "threshold": 0.55,
  "model_path": null
}
```

**Response:**
```json
{
  "predictions": [
    {
      "description": "A thrilling action movie",
      "prediction": {
        "genres": ["Action", "Thriller"],
        "genre_count": 2
      },
      "threshold": 0.55
    },
    {
      "description": "A romantic comedy about two strangers",
      "prediction": {
        "genres": ["Romance", "Comedy"],
        "genre_count": 2
      },
      "threshold": 0.55
    }
  ],
  "total_predictions": 2,
  "threshold": 0.55
}
```

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"description": "A thrilling action movie"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"descriptions": ["A thrilling action movie", "A romantic comedy"]}'
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"description": "A thrilling action movie"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "descriptions": [
            "A thrilling action movie",
            "A romantic comedy"
        ]
    }
)
print(response.json())
```

## Configuration

Configuration is managed through `app/config.py` and can be overridden with environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Debug mode (default: `False`)
- `DEFAULT_THRESHOLD`: Default prediction threshold (default: `0.55`)

## Project Structure

```
app/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application and routes
├── config.py            # Configuration settings
├── schemas.py           # Pydantic request/response models
├── services.py          # Business logic and prediction service
├── run.py               # Run script for development
└── README.md            # This file
```

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation
- `pydantic-settings`: Settings management
