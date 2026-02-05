# Movie Genre Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A production-ready machine learning pipeline for predicting movie genres from textual descriptions using TF-IDF vectorization and multi-label classification with LinearSVC (chosen for better generalization and reduced overfitting).

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for multi-label genre classification. Given a movie description, the model predicts one or more genres (e.g., Action, Drama, Comedy) that best describe the movie. The system uses scikit-learn's TF-IDF vectorization combined with OneVsRestClassifier strategy for handling multiple labels per movie.

### Key Features

- **Multi-label Classification**: Predicts multiple genres per movie (e.g., "Action, Adventure, Thriller")
- **Production-Ready Pipeline**: Complete workflow from raw data to trained model
- **REST API**: FastAPI-based API with `/predict` and `/predict/batch` endpoints
- **Docker Support**: Containerized deployment for easy production deployment
- **Top-K Genre Selection**: Selects top k genres by probability, filtered by threshold (default: top 3)
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Generalization-Focused**: LinearSVC with strong regularization (C=0.1) prevents overfitting
- **Strong Performance**: Achieves **69.65% F1-score**, **78.93% precision**, and **62.32% recall**
- **Comprehensive Logging**: Detailed progress tracking and error handling

## 📊 Model Performance

The trained LinearSVC model achieves strong performance with excellent generalization properties:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **F1 Score** | 69.65% | Strong overall precision-recall balance |
| **Precision** | 78.93% | High accuracy when predicting genres |
| **Recall** | 62.32% | Good - captures most true genres |
| **Hamming Loss** | 9.95% | Low error rate, excellent for multi-label classification |
| **Jaccard Score** | 53.43% | Good overlap between predicted and true genres |

**Model Selection**: LinearSVC was chosen over LogisticRegression specifically to address overfitting concerns. The stronger regularization (C=0.1) ensures better generalization to unseen data, making it more suitable for production deployment while maintaining strong performance metrics.

See the [Model Evaluation Report](reports/model_evaluation_report.md) for detailed analysis.

## 🚀 Quick Start

### Prerequisites

**Option 1: Local Development (Conda)**
- Python 3.12
- Conda (for environment management)

**Option 2: Docker Deployment**
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd movie_genre_model
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate movie_genre_model
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Running the Pipeline

The project uses a Makefile for convenient command execution. Run `make help` to see all available commands.

#### Complete Pipeline (Recommended)

Run the entire pipeline from data processing to model evaluation:

```bash
make train    # Processes data, trains model, and evaluates
```

#### Step-by-Step Execution

1. **Process raw data**:
   ```bash
   make data
   ```
   This loads raw movie data and performs basic cleaning.

2. **Train the model** (Recommended):
   ```bash
   make train
   ```
   This will:
   - Load interim data
   - Split into train/test sets **before** preprocessing (prevents data leakage)
   - Fit TF-IDF and label encoding on **training data only**
   - Transform both train and test sets
   - Train the multi-label classifier and save it to `models/`

3. **Preprocess data** (Optional - for exploration/evaluation only):
   ```bash
   make preprocess
   ```
   ⚠️ **Warning**: This fits preprocessors on the entire dataset, causing data leakage if used for training.
   Only use this for exploratory analysis or evaluation on pre-processed datasets.

4. **Evaluate the model**:
   ```bash
   make evaluate
   ```
   Evaluates the trained model on interim data (transforms using saved preprocessors from training).
   The evaluation pipeline automatically uses the preprocessors saved during training, ensuring
   consistent feature representation.

#### Making Predictions

**Command Line Interface:**

Predict genres for a single description (default: top 3 genres above threshold 0.55):
```bash
python -m descriptions.modeling.predict \
    --description "A thrilling action movie about a spy who saves the world"
```

Predict with custom top-k and threshold:
```bash
python -m descriptions.modeling.predict \
    --description "A sci-fi action thriller with romance elements" \
    --top-k 5 \
    --threshold 0.5
```

Predict genres from a CSV file:
```bash
python -m descriptions.modeling.predict \
    --input-file data/test_movies.csv \
    --output-file predictions.csv \
    --top-k 3 \
    --threshold 0.55
```

**REST API:**

Start the API server:
```bash
make api
# or for production
make api-prod
```

The API will be available at `http://localhost:8000` with interactive documentation at `/docs`.

Single prediction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A thrilling action movie about a secret agent saving the world",
    "threshold": 0.55,
    "top_k": 3
  }'
```

Batch prediction:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "descriptions": [
      "A thrilling action movie about a secret agent",
      "A romantic comedy about two strangers who fall in love"
    ],
    "threshold": 0.55,
    "top_k": 3
  }'
```

**Top-K Selection**: The model uses a top-k selection strategy where it:
1. Selects the top k genres by probability (default: k=3)
2. Only includes genres above the probability threshold (default: 0.55)
3. Returns up to k genres per prediction, ensuring quality predictions

## 📁 Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Convenience commands for common tasks
├── README.md          <- This file
├── Dockerfile         <- Docker configuration for containerized deployment
├── .dockerignore      <- Docker build exclusions
├── environment.yml    <- Conda environment specification
├── pyproject.toml     <- Project configuration
│
├── data/
│   ├── external/      <- Data from third party sources
│   ├── interim/      <- Intermediate data (cleaned)
│   ├── processed/    <- Final canonical datasets
│   └── raw/          <- Original immutable data dump
│
├── app/              <- FastAPI REST API application
│   ├── main.py      <- API endpoints and routes
│   ├── services.py  <- Prediction service layer
│   ├── schemas.py   <- Pydantic request/response models
│   └── config.py    <- API configuration
│
├── descriptions/      <- Source code package
│   ├── __init__.py
│   ├── config.py     <- Configuration and paths
│   ├── dataset.py    <- Data loading and processing
│   ├── modeling/
│   │   ├── preprocess.py  <- TF-IDF and label encoding
│   │   ├── train.py      <- Model training
│   │   ├── evaluate.py    <- Model evaluation
│   │   ├── predict.py    <- Inference/prediction
│   │   └── model.py      <- Model building utilities
│   └── plots.py      <- Visualization utilities
│
├── models/           <- Trained models
│   ├── linearsvc.joblib      <- Full pipeline (tfidf → normalizer → feature_selector → classifier)
│   ├── genre_binarizer.joblib <- Label encoding/decoding
│   ├── metrics_linearsvc.json
│   └── linearsvc_parameters.json
│
├── mlruns/           <- MLflow experiment tracking
│
├── notebooks/        <- Jupyter notebooks for exploration
│
├── reports/          <- Generated analysis and reports
│   ├── figures/     <- Visualization outputs
│   └── model_evaluation_report.md
│
├── tests/            <- Unit tests
│
└── docs/             <- Documentation (MkDocs)
```

## 🔧 Configuration

### Model Hyperparameters

The model uses the following hyperparameters optimized for generalization:

- **C**: 0.1 (regularization strength - lower value provides stronger regularization)
- **Penalty**: L2 (Ridge regularization)
- **Loss**: squared_hinge (more robust to outliers than logistic loss)
- **Max Iterations**: 1000
- **Class Weight**: balanced (handles class imbalance)
- **Dual**: False (uses primal formulation for efficiency)

**Note**: The lower C value (0.1 vs 50.0) was chosen specifically to prevent overfitting observed with LogisticRegression, prioritizing generalization over peak performance metrics.

### Preprocessing Parameters

- **Max Features**: 20,000 TF-IDF features
- **N-gram Range**: (1, 2) - unigrams and bigrams
- **Stop Words**: English stop words removed
- **Sublinear TF**: True (log scaling for term frequencies)

### Customization

You can customize hyperparameters when training:

```bash
python -m descriptions.modeling.train \
    --C 0.1 \
    --penalty l2 \
    --loss squared_hinge \
    --max-iter 1000
```

**Note**: When tuning C, consider values between 0.05-1.0. Lower values provide stronger regularization and better generalization, while higher values may improve training metrics but risk overfitting.

## 🌐 REST API

The project includes a production-ready FastAPI REST API for making predictions.

### Starting the API Server

**Development mode:**
```bash
make api
```

**Production mode (with multiple workers):**
```bash
make api-prod
```

The API will be available at `http://localhost:8000` with:
- Interactive API documentation: `http://localhost:8000/docs` (Swagger UI)
- Alternative documentation: `http://localhost:8000/redoc` (ReDoc)
- Health check endpoint: `http://localhost:8000/health`

### API Endpoints

**POST `/predict`** - Predict genres for a single description
- Request body: `{ "description": "...", "threshold": 0.55, "top_k": 3 }`
- Returns: Predicted genres with confidence

**POST `/predict/batch`** - Predict genres for multiple descriptions
- Request body: `{ "descriptions": ["...", "..."], "threshold": 0.55, "top_k": 3 }`
- Returns: List of predictions for each description

**GET `/health`** - Health check and model status
- Returns: API status and whether model is loaded

### API Features

- **Top-K Selection**: Selects top k genres by probability (default: 3)
- **Threshold Filtering**: Only includes genres above probability threshold (default: 0.55)
- **Batch Processing**: Process multiple descriptions in a single request
- **Automatic Model Loading**: Model loads automatically on API startup
- **CORS Enabled**: Ready for frontend integration
- **Request Validation**: Pydantic schemas ensure data validation
- **Error Handling**: Comprehensive error responses

See the interactive documentation at `/docs` for detailed request/response schemas and examples.

## 🐳 Docker Deployment

The project includes a Dockerfile for containerized deployment. This is the recommended approach for production deployments.

### Prerequisites

- Docker installed on your system
- Trained models in the `models/` directory (or mount them as a volume)

### Building the Docker Image

Build the Docker image:

```bash
docker build -t movie-genre-api .
```

### Running the Container

**Basic run:**
```bash
docker run -p 8000:8000 movie-genre-api
```

**With volume mount for models (recommended for production):**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  movie-genre-api
```

**With environment variables:**
```bash
docker run -p 8000:8000 \
  -e PORT=8000 \
  -v $(pwd)/models:/app/models \
  movie-genre-api
```

The API will be available at `http://localhost:8000`.

### Docker Features

- **Health Checks**: Built-in health check endpoint monitoring
- **Optimized Image**: Uses Python 3.12 slim image for smaller size
- **Production Ready**: Includes all necessary dependencies
- **Volume Support**: Models can be mounted as volumes for easy updates

### Important Notes

1. **Models Required**: The Docker container needs access to trained model files. Either:
   - Copy model files into the image during build, or
   - Mount the `models/` directory as a volume at runtime (recommended)

2. **Model Location**: Ensure your trained models are in the `models/` directory before building or mounting.

3. **Health Check**: The container includes a health check that monitors the `/health` endpoint.

### Testing the Docker Container

Once the container is running, test it:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A thrilling action movie about a secret agent",
    "threshold": null,
    "top_k": 3
  }'
```

### Docker Compose (Optional)

For easier local development with Docker Compose, create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

Then run:
```bash
docker-compose up
```

## 📈 MLflow Integration

The project uses MLflow for experiment tracking. All training runs are automatically logged with:

- Model hyperparameters
- Preprocessing parameters
- Training metrics
- Model artifacts

View experiments:
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

## 🧪 Testing

Run the test suite:

```bash
make test
```

Or run specific tests:
```bash
pytest tests/
```

## 📝 Usage Examples

### Example 1: Complete Pipeline

```bash
# Complete end-to-end pipeline (recommended)
make evaluate
# This runs: data → train → evaluate
# All steps are connected and handle data leakage prevention automatically
```

Or step-by-step:
```bash
# Process raw data
make data

# Train model (splits before preprocessing, saves model + preprocessors)
make train

# Evaluate model (uses saved preprocessors to transform interim data)
make evaluate
```

### Example 2: Custom Training

```bash
python -m descriptions.modeling.train \
    --C 0.1 \
    --penalty l2 \
    --loss squared_hinge \
    --max-iter 1000 \
    --test-size 0.2 \
    --random-state 42
```

**Note**: The training pipeline automatically handles train/test splitting **before** preprocessing to prevent data leakage. TF-IDF and label encoding are fitted only on training data.

### Example 3: Prediction with Custom Threshold and Top-K

```bash
# Use lower threshold to get more genres
python -m descriptions.modeling.predict \
    --description "A romantic comedy about two people who fall in love" \
    --threshold 0.4 \
    --top-k 5

# Use higher threshold for more conservative predictions
python -m descriptions.modeling.predict \
    --description "A sci-fi action thriller" \
    --threshold 0.7 \
    --top-k 2
```

**Understanding Top-K and Threshold**:
- **Top-K**: Maximum number of genres to select (default: 3). The model selects the top k genres by probability.
- **Threshold**: Minimum probability required for a genre to be included (default: 0.55). Only genres above this threshold are returned.
- Lower thresholds predict more genres, higher thresholds are more conservative.
- The top-k selection ensures you get the most confident predictions up to k genres.

## 🛠️ Development

### Code Formatting

Format code:
```bash
make format
```

Check formatting:
```bash
make lint
```

### Adding New Features

The codebase follows a modular structure:
- `dataset.py`: Data loading and cleaning
- `preprocess.py`: Feature engineering (TF-IDF, label encoding) - ⚠️ For exploration/evaluation only
- `train.py`: Model training logic (handles train/test split before preprocessing to prevent data leakage)
- `evaluate.py`: Model evaluation and metrics
- `predict.py`: Inference pipeline with top-k selection
- `app/main.py`: FastAPI REST API endpoints
- `app/services.py`: Prediction service layer
- `app/schemas.py`: API request/response models

**Important**: The training pipeline (`train.py`) properly splits data before fitting preprocessors to avoid data leakage. The standalone `preprocess.py` script fits on the entire dataset and should only be used for exploratory analysis.

## 📊 Model Architecture

### Approach

The model uses a **OneVsRestClassifier** strategy:
- One binary classifier per genre
- Each classifier learns to distinguish one genre from all others
- Allows independent prediction of multiple genres per movie

### Base Classifier

- **Algorithm**: LinearSVC (Linear Support Vector Classifier)
- **Regularization**: L2 (Ridge) with C=0.1 (strong regularization)
- **Loss Function**: Squared hinge (more robust than logistic loss)
- **Optimization**: Primal formulation (dual=False)
- **Class Weighting**: Balanced (handles imbalanced genres)

**Why LinearSVC?** The model was switched from LogisticRegression to LinearSVC to address overfitting concerns. The stronger regularization (C=0.1) ensures better generalization to unseen data, making it more suitable for production deployment despite lower training metrics.

### Feature Engineering

- **Text Vectorization**: TF-IDF with 20,000 features
- **N-grams**: Unigrams and bigrams
- **Sublinear TF**: Log scaling for better feature representation
- **Stop Words**: Removed common English words

## 📚 Documentation

- [Model Evaluation Report](reports/model_evaluation_report.md) - Detailed performance analysis
- [Getting Started Guide](docs/docs/getting-started.md) - Setup instructions
- Code documentation available via docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built using the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project template
- Uses scikit-learn for machine learning
- MLflow for experiment tracking

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: December 2025  
**Model Version**: 2.0 (LinearSVC)  
**Performance**: 69.65% F1-score, 9.95% Hamming Loss, 78.93% Precision  
**Model Selection**: LinearSVC chosen over LogisticRegression to address overfitting concerns  
**Latest Features**: Top-k genre selection, per-label thresholds, REST API with FastAPI, Docker support
