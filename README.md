🎬 Movie Genre Model

📖 Project Overview

This project implements a multi-label movie genre classification model that predicts the genres of a movie based on its description. The model is capable of handling complex descriptions and assigns multiple genres simultaneously using a transformer-based deep learning architecture.

🛠️ Model Architecture
The model uses a custom Transformer Encoder with the following components:

Embedding Layer: Maps token IDs to dense vectors.
Positional Encoding: Provides token position information.
Transformer Encoder:
4 stacked encoder layers
Multi-Head Self-Attention (8 heads)
Feedforward layers with residual connections and layer normalization
Self-Attention Pooling: Learns to focus on key parts of the description.
Fully Connected Layers: Three dense layers to map pooled features to genre probabilities.
Monte Carlo Dropout: Enables uncertainty estimation during training.
Optional GNN Integration: (Branch-in-progress) Adds label correlation modeling through a Graph Neural Network.

🗃️ Dataset
Input: Movie descriptions
Labels: Multi-label genres (e.g., Action, Comedy, Drama)
Total Samples: ~7,500
Data Source: Custom CSV containing movie descriptions and their corresponding genres.
Tokenization: BERT-style tokenizer (WordPiece) with a vocabulary size of 30,522.

📊 Evaluation Metrics
The model is evaluated using multi-label classification metrics:

Subset Accuracy: Percentage of samples where all predicted labels exactly match the true labels (strict metric).
Micro Precision: Measures overall label accuracy across all classes.
Micro Recall: Measures overall ability to detect all true labels.
Micro F1 Score: Harmonic mean of precision and recall.
Hamming Loss: Fraction of misclassified labels (recommended for multi-label tasks).
Example Metrics:
Metric	Train	Validation
Precision	~46%	~43%
Recall	~66%	~64%
F1 Score	~55%	~52%
Subset Accuracy	~4%	~4%

🚀 Features
Supports multi-label genre prediction.
Uses attention mechanisms to focus on relevant text parts.
Incorporates Monte Carlo dropout for model uncertainty.
GNN-based label correlation (experimental branch in progress).
Threshold tuning for flexible decision boundaries at inference.

📂 Files
Text Classifier.ipynb – Main training notebook.
movie_genre_model.pth – Saved model weights.
top_movies.csv – Dataset containing movie descriptions and genres.
README.md – Project documentation.

✅ How to Use
Train the model using Text Classifier.ipynb.
Save the model using torch.save().
Load the model and run predict_genres() to infer genres from new movie descriptions.

📈 Future Work
Tune per-label thresholds for improved performance.
Fully integrate Graph Neural Networks (GNN) for label dependency modeling.
Deploy as a REST API or web app for interactive predictions.
