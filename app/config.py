"""Configuration for Flask API."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration."""

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "5000"))
    API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", None)  # None = auto-detect
    DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.55"))

    # Paths
    PROJ_ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = PROJ_ROOT / "models"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")


class DevelopmentConfig(Config):
    """Development configuration."""

    API_DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""

    API_DEBUG = False
    LOG_LEVEL = "INFO"


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
