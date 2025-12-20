"""FastAPI application main file."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
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


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up Movie Genre Prediction API...")
    try:
        prediction_service.load_model()
        logger.success("✓ API startup complete - model loaded")
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
    description="Returns API information"
)
async def root():
    """Root endpoint."""
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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict genres for a single description",
    description="Predict movie genres from a single description text"
)
async def predict(request: PredictionRequest):
    """
    Predict genres for a single movie description.
    
    Args:
        request: Prediction request with description and optional parameters
    
    Returns:
        Prediction response with predicted genres
    """
    try:
        # Make prediction
        predicted_genres = prediction_service.predict(
            descriptions=[request.description],
            threshold=request.threshold,
            top_k=request.top_k,
            model_path=request.model_path,
        )
        
        genres = predicted_genres[0] if predicted_genres else []
        
        return PredictionResponse(
            description=request.description[:100] + "..." if len(request.description) > 100 else request.description,
            prediction=GenrePrediction(
                genres=list(genres),
                genre_count=len(genres)
            ),
            threshold=request.threshold,
            top_k=request.top_k
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
    description="Predict movie genres for a batch of descriptions"
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict genres for multiple movie descriptions.
    
    Args:
        request: Batch prediction request with list of descriptions
    
    Returns:
        Batch prediction response with all predictions
    """
    try:
        # Make predictions
        predicted_genres = prediction_service.predict(
            descriptions=request.descriptions,
            threshold=request.threshold,
            top_k=request.top_k,
            model_path=request.model_path,
        )
        
        # Format responses
        predictions = []
        for desc, genres in zip(request.descriptions, predicted_genres):
            predictions.append(
                PredictionResponse(
                    description=desc[:100] + "..." if len(desc) > 100 else desc,
                    prediction=GenrePrediction(
                        genres=list(genres),
                        genre_count=len(genres)
                    ),
                    threshold=request.threshold,
                    top_k=request.top_k
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
