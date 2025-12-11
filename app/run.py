"""Run Flask API server."""

import os
from pathlib import Path

from loguru import logger

from app.app import create_app

if __name__ == "__main__":
    # Determine config from environment
    config_name = os.getenv("FLASK_ENV", "development")

    # Create app
    app = create_app(config_name=config_name)

    # Get host and port from config
    host = app.config["API_HOST"]
    port = app.config["API_PORT"]
    debug = app.config["API_DEBUG"]

    logger.info("=" * 70)
    logger.info("Starting Flask API Server")
    logger.info("=" * 70)
    logger.info(f"Environment: {config_name}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    logger.info("=" * 70)

    # Run Flask development server
    app.run(host=host, port=port, debug=debug)
