"""FastAPI application main file."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.config import settings
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    GenrePrediction,
    GenreWithConfidence,
    DescriptionValidation,
    ModelInfoResponse,
    DescriptionLengthStats,
)
from app.services import prediction_service


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up Movie Genre Prediction API...")
    try:
        prediction_service.load_model()
        logger.success("✓ API startup complete - model loaded")
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.warning("API will start but predictions will not be available until models are loaded")
        logger.info("To load models: Place .joblib files in the models/ directory or mount as volume")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.warning("API will start but model may not be loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Movie Genre Prediction API...")


@app.get(
    "/",
    tags=["Root"],
    summary="Root endpoint",
    description="Returns API information or serves UI"
)
async def root(request: Request):
    """Root endpoint - serves the UI or returns API info as JSON."""
    # Check Accept header - if JSON is preferred, return JSON
    accept = request.headers.get("accept", "")
    if "application/json" in accept or "text/html" not in accept:
        return {
            "message": "Movie Genre Prediction API",
            "version": settings.API_VERSION,
            "docs": "/docs",
            "health": "/health"
        }
    
    # Otherwise, serve HTML if available
    html_file = static_dir / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    
    return {
        "message": "Movie Genre Prediction API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check API health and model status"
)
async def health_check():
    """Health check endpoint."""
    is_ready = prediction_service.is_ready()
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        version=settings.API_VERSION,
        model_loaded=is_ready
    )


@app.get(
    "/info",
    response_model=ModelInfoResponse,
    tags=["Info"],
    summary="Get model information and recommendations",
    description="Returns comprehensive model metadata, metrics, and description length recommendations"
)
async def get_model_info():
    """
    Get model information and recommendations.
    
    Returns:
        - Model metadata (name, path, status)
        - Model performance metrics
        - Description length statistics and recommendations
        - Threshold configuration information
    
    This endpoint is useful for:
    - Understanding optimal description length for best predictions
    - Checking model performance metrics
    - Getting model configuration details
    """
    try:
        if not prediction_service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure the model is trained and available."
            )
        
        info = prediction_service.get_model_info()
        
        # Build response with proper schema structure
        description_stats = None
        if info.get("description_stats"):
            stats = info["description_stats"]
            description_stats = DescriptionLengthStats(**stats)
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_path=info["model_path"],
            model_loaded=info["model_loaded"],
            n_classes=info["n_classes"],
            n_features=info["n_features"],
            metrics=info["metrics"],
            description_stats=description_stats,
            threshold_type=info["threshold_type"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict genres for a single description",
    description="Predict movie genres from a single description text with confidence scores and validation"
)
async def predict(request: PredictionRequest):
    """
    Predict genres for a single movie description.
    
    Includes:
    - Genre predictions with confidence scores
    - Description length validation and recommendations
    - Optimal description length guidance
    
    Args:
        request: Prediction request with description and optional parameters
    
    Returns:
        Prediction response with predicted genres, confidence scores, and validation info
    """
    try:
        # Get description length stats for validation
        stats = prediction_service.get_description_length_stats()
        optimal_min = stats["optimal_min"] if stats else None
        optimal_max = stats["optimal_max"] if stats else None
        
        # Validate description length
        validation_info = prediction_service.validate_description_length(
            request.description,
            optimal_min=optimal_min,
            optimal_max=optimal_max
        )
        
        # Make prediction with confidence scores
        predicted_genres, confidence_scores = prediction_service.predict_with_confidence(
            descriptions=[request.description],
            threshold=request.threshold,
            top_k=request.top_k,
            model_path=request.model_path,
        )
        
        genres = predicted_genres[0] if predicted_genres else []
        conf_scores = confidence_scores[0] if confidence_scores else {}
        
        # Build genres with confidence
        genres_with_conf = [
            GenreWithConfidence(genre=genre, confidence=conf_scores.get(genre, 0.0))
            for genre in genres
        ]
        # Sort by confidence descending
        genres_with_conf.sort(key=lambda x: x.confidence, reverse=True)
        
        return PredictionResponse(
            description=request.description[:100] + "..." if len(request.description) > 100 else request.description,
            prediction=GenrePrediction(
                genres=list(genres),
                genre_count=len(genres),
                genres_with_confidence=genres_with_conf
            ),
            threshold=request.threshold,
            top_k=request.top_k,
            validation=DescriptionValidation(**validation_info) if validation_info else None
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict genres for multiple descriptions",
    description="Predict movie genres for a batch of descriptions with confidence scores and validation"
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict genres for multiple movie descriptions.
    
    Includes:
    - Genre predictions with confidence scores for each description
    - Description length validation and recommendations for each description
    
    Args:
        request: Batch prediction request with list of descriptions
    
    Returns:
        Batch prediction response with all predictions, confidence scores, and validation info
    """
    try:
        # Get description length stats for validation
        stats = prediction_service.get_description_length_stats()
        optimal_min = stats["optimal_min"] if stats else None
        optimal_max = stats["optimal_max"] if stats else None
        
        # Make predictions with confidence scores
        predicted_genres, confidence_scores = prediction_service.predict_with_confidence(
            descriptions=request.descriptions,
            threshold=request.threshold,
            top_k=request.top_k,
            model_path=request.model_path,
        )
        
        # Format responses with validation
        predictions = []
        for desc, genres, conf_scores in zip(request.descriptions, predicted_genres, confidence_scores):
            # Validate description length
            validation_info = prediction_service.validate_description_length(
                desc,
                optimal_min=optimal_min,
                optimal_max=optimal_max
            )
            
            # Build genres with confidence
            genres_with_conf = [
                GenreWithConfidence(genre=genre, confidence=conf_scores.get(genre, 0.0))
                for genre in genres
            ]
            genres_with_conf.sort(key=lambda x: x.confidence, reverse=True)
            
            predictions.append(
                PredictionResponse(
                    description=desc[:100] + "..." if len(desc) > 100 else desc,
                    prediction=GenrePrediction(
                        genres=list(genres),
                        genre_count=len(genres),
                        genres_with_confidence=genres_with_conf
                    ),
                    threshold=request.threshold,
                    top_k=request.top_k,
                    validation=DescriptionValidation(**validation_info) if validation_info else None
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            threshold=request.threshold,
            top_k=request.top_k
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
