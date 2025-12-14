"""Configuration settings for the FastAPI application."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Movie Genre Prediction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for predicting movie genres from text descriptions"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model Settings
    DEFAULT_MODEL_PATH: str = "linearsvc.joblib"
    DEFAULT_THRESHOLD: float = 0.55
    
    # Paths
    MODELS_DIR: Path = Path("models")
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Legacy Flask environment variables (ignored but allowed)
    FLASK_ENV: str | None = None
    API_HOST: str | None = None
    API_PORT: int | None = None
    API_DEBUG: bool | None = None
    LOG_LEVEL: str | None = None
    CORS_ORIGINS: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables
    )


settings = Settings()
