"""Pydantic schemas for request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    
    description: str = Field(
        ...,
        description="Movie description text",
        min_length=1,
        max_length=10000,
        example="A thrilling action movie about a secret agent saving the world"
    )
    threshold: Optional[float] = Field(
        default=0.55,
        description="Probability threshold for predictions (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to model file (optional, uses default if not provided)"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    descriptions: List[str] = Field(
        ...,
        description="List of movie descriptions",
        min_items=1,
        max_items=1000,
        example=["A thrilling action movie", "A romantic comedy about two strangers"]
    )
    threshold: Optional[float] = Field(
        default=0.55,
        description="Probability threshold for predictions (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to model file (optional, uses default if not provided)"
    )


class GenrePrediction(BaseModel):
    """Schema for a single genre prediction."""
    
    genres: List[str] = Field(
        ...,
        description="List of predicted genres",
        example=["Action", "Thriller"]
    )
    genre_count: int = Field(
        ...,
        description="Number of predicted genres",
        ge=0
    )


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    description: str = Field(
        ...,
        description="Input description (truncated if long)"
    )
    prediction: GenrePrediction = Field(
        ...,
        description="Genre prediction result"
    )
    threshold: float = Field(
        ...,
        description="Threshold used for prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions"
    )
    threshold: float = Field(
        ...,
        description="Threshold used for predictions"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details"
    )
