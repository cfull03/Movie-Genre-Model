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
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Generalization-Focused**: LinearSVC with strong regularization (C=0.1) prevents overfitting
- **Solid Performance**: Achieves **60.59% F1-score**, **54.22% precision**, and **68.67% recall**
- **Comprehensive Logging**: Detailed progress tracking and error handling

## 📊 Model Performance

The trained LinearSVC model achieves solid performance with strong generalization properties:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **F1 Score** | 60.59% | Solid overall precision-recall balance |
| **Precision** | 54.22% | Moderate accuracy when predicting genres |
| **Recall** | 68.67% | Good - captures most true genres |
| **Hamming Loss** | 16.53% | Moderate error rate, acceptable for multi-label classification |
| **Jaccard Score** | 43.46% | Moderate overlap between predicted and true genres |

**Model Selection**: LinearSVC was chosen over LogisticRegression specifically to address overfitting concerns. While metrics are lower than a less regularized model, the stronger regularization (C=0.1) ensures better generalization to unseen data, making it more suitable for production deployment.

See the [Model Evaluation Report](reports/model_evaluation_report.md) for detailed analysis.

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- Conda (for environment management)

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

Predict genres for a single description:
```bash
python -m descriptions.modeling.predict \
    --description "A thrilling action movie about a spy who saves the world"
```

Predict genres from a CSV file:
```bash
python -m descriptions.modeling.predict \
    --input-file data/test_movies.csv \
    --output-file predictions.csv
```

## 📁 Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Convenience commands for common tasks
├── README.md          <- This file
├── environment.yml    <- Conda environment specification
├── pyproject.toml     <- Project configuration
│
├── data/
│   ├── external/      <- Data from third party sources
│   ├── interim/      <- Intermediate data (cleaned)
│   ├── processed/    <- Final canonical datasets
│   └── raw/          <- Original immutable data dump
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
├── models/           <- Trained models and preprocessors
│   ├── linearsvc.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── genre_binarizer.joblib
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

### Example 3: Prediction with Custom Threshold

```bash
python -m descriptions.modeling.predict \
    --description "A romantic comedy about two people who fall in love" \
    --threshold 0.4
```

Lower thresholds predict more genres, higher thresholds are more conservative.

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
- `predict.py`: Inference pipeline

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

**Last Updated**: November 2025  
**Model Version**: 2.0 (LinearSVC)  
**Performance**: 60.59% F1-score, 16.53% Hamming Loss  
**Model Selection**: LinearSVC chosen over LogisticRegression to address overfitting concerns
