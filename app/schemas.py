"""Pydantic schemas for request/response validation."""

from typing import Dict, List, Optional
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
        default=None,
        description="Probability threshold for predictions (0.0-1.0). If None, uses per-label thresholds if available, else 0.55. Only genres above their threshold will be included.",
        ge=0.0,
        le=1.0
    )
    top_k: Optional[int] = Field(
        default=3,
        description="Maximum number of top genres to select (default: 3). The top k genres by probability will be selected, but only those above the threshold will be returned.",
        ge=1,
        le=20
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
        default=None,
        description="Probability threshold for predictions (0.0-1.0). If None, uses per-label thresholds if available, else 0.55. Only genres above their threshold will be included.",
        ge=0.0,
        le=1.0
    )
    top_k: Optional[int] = Field(
        default=3,
        description="Maximum number of top genres to select (default: 3). The top k genres by probability will be selected, but only those above the threshold will be returned.",
        ge=1,
        le=20
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to model file (optional, uses default if not provided)"
    )


class GenreWithConfidence(BaseModel):
    """Schema for a genre with its confidence score."""
    
    genre: str = Field(
        ...,
        description="Genre name",
        example="Action"
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence/probability (0.0-1.0)",
        ge=0.0,
        le=1.0
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
    genres_with_confidence: Optional[List[GenreWithConfidence]] = Field(
        default=None,
        description="List of predicted genres with confidence scores"
    )


class DescriptionValidation(BaseModel):
    """Schema for description validation and recommendations."""
    
    length: int = Field(
        ...,
        description="Description length in characters",
        ge=0
    )
    is_optimal: bool = Field(
        ...,
        description="Whether the description length is within the recommended range"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommendation message if description length is not optimal"
    )
    optimal_min: Optional[int] = Field(
        default=None,
        description="Recommended minimum description length (characters)"
    )
    optimal_max: Optional[int] = Field(
        default=None,
        description="Recommended maximum description length (characters)"
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
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold used for prediction. None indicates per-label thresholds were used."
    )
    top_k: int = Field(
        ...,
        description="Top-k value used for prediction"
    )
    validation: Optional[DescriptionValidation] = Field(
        default=None,
        description="Description validation and recommendations"
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
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold used for predictions. None indicates per-label thresholds were used."
    )
    top_k: int = Field(
        ...,
        description="Top-k value used for predictions"
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


class DescriptionLengthStats(BaseModel):
    """Schema for description length statistics."""
    
    min: int = Field(..., description="Minimum description length in characters")
    max: int = Field(..., description="Maximum description length in characters")
    mean: float = Field(..., description="Mean description length in characters")
    median: float = Field(..., description="Median description length in characters")
    q25: float = Field(..., description="25th percentile (Q1) description length")
    q75: float = Field(..., description="75th percentile (Q3) description length")
    optimal_min: int = Field(
        ..., 
        description="Recommended minimum description length (Q1, 25th percentile)"
    )
    optimal_max: int = Field(
        ..., 
        description="Recommended maximum description length (Q3, 75th percentile)"
    )


class ModelInfoResponse(BaseModel):
    """Schema for model information response."""
    
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model file")
    model_loaded: bool = Field(..., description="Whether model is currently loaded")
    n_classes: int = Field(..., description="Number of genre classes")
    n_features: int = Field(..., description="Number of features")
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Model evaluation metrics (if available)"
    )
    description_stats: Optional[DescriptionLengthStats] = Field(
        default=None,
        description="Description length statistics from training data"
    )
    threshold_type: str = Field(
        ..., 
        description="Threshold type: 'global', 'per-label', or 'default'"
    )
