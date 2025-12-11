"""Flask application for movie genre classification API."""

import time
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, request
from loguru import logger

from app.config import Config, config
from app.model_loader import model_loader
from descriptions.modeling.predict import predict_genres


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Create and configure Flask application.

    Args:
        config_name: Configuration name ('development', 'production', or None for default)

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)

    # Load configuration
    config_name = config_name or "default"
    app.config.from_object(config[config_name])

    # Initialize model loader
    try:
        model_path_str = app.config.get("MODEL_PATH")
        model_path = Path(model_path_str) if model_path_str else None
        model_loader.load_model(model_path)
        logger.success("✓ Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
        logger.info("Model will be loaded on first request")

    # Request logging middleware
    @app.before_request
    def log_request():
        """Log incoming requests."""
        request.start_time = time.time()
        logger.debug(f"Request: {request.method} {request.path}")

    @app.after_request
    def log_response(response):
        """Log outgoing responses."""
        duration = time.time() - request.start_time
        logger.debug(
            f"Response: {response.status_code} | Duration: {duration:.3f}s | "
            f"Path: {request.path}"
        )
        return response

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({"error": "Not found", "message": str(error)}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        model_status = model_loader.is_loaded()
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": model_status,
            }
        ), 200

    # Model info endpoint
    @app.route("/model/info", methods=["GET"])
    def model_info():
        """Get model information."""
        try:
            info = model_loader.get_model_info()
            return jsonify(info), 200
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return jsonify({"error": str(e)}), 500

    # Prediction endpoint
    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Predict genres for a single movie description.

        Request body:
        {
            "description": "A thrilling action movie...",
            "threshold": 0.55  # optional
        }

        Response:
        {
            "genres": ["Action", "Thriller"],
            "description": "A thrilling action movie...",
            "threshold": 0.55
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            description = data.get("description")
            if not description:
                return jsonify({"error": "Missing 'description' field"}), 400

            threshold = data.get("threshold", app.config["DEFAULT_THRESHOLD"])

            # Ensure model is loaded
            if not model_loader.is_loaded():
                model_path_str = app.config.get("MODEL_PATH")
                model_path = Path(model_path_str) if model_path_str else None
                model_loader.load_model(model_path)

            # Make prediction
            predictions = predict_genres(
                descriptions=[description], threshold=threshold
            )

            return jsonify(
                {
                    "genres": predictions[0] if predictions else [],
                    "description": description,
                    "threshold": threshold,
                }
            ), 200

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500

    # Batch prediction endpoint
    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """
        Predict genres for multiple movie descriptions.

        Request body:
        {
            "descriptions": ["Description 1", "Description 2", ...],
            "threshold": 0.55  # optional
        }

        Response:
        {
            "predictions": [
                {"description": "Description 1", "genres": ["Action"]},
                {"description": "Description 2", "genres": ["Drama", "Romance"]}
            ],
            "threshold": 0.55
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            descriptions = data.get("descriptions")
            if not descriptions:
                return jsonify({"error": "Missing 'descriptions' field"}), 400

            if not isinstance(descriptions, list):
                return jsonify({"error": "'descriptions' must be a list"}), 400

            threshold = data.get("threshold", app.config["DEFAULT_THRESHOLD"])

            # Ensure model is loaded
            if not model_loader.is_loaded():
                model_path_str = app.config.get("MODEL_PATH")
                model_path = Path(model_path_str) if model_path_str else None
                model_loader.load_model(model_path)

            # Make predictions
            predictions = predict_genres(descriptions=descriptions, threshold=threshold)

            # Format response
            results = [
                {"description": desc, "genres": pred}
                for desc, pred in zip(descriptions, predictions)
            ]

            return jsonify(
                {
                    "predictions": results,
                    "threshold": threshold,
                    "count": len(results),
                }
            ), 200

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return jsonify({"error": str(e)}), 500

    # Metrics endpoint (placeholder)
    @app.route("/metrics", methods=["GET"])
    def metrics():
        """Get model metrics."""
        try:
            # TODO: Load metrics from models/metrics_linearsvc.json
            return jsonify({"message": "Metrics endpoint - to be implemented"}), 200
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({"error": str(e)}), 500

    return app
