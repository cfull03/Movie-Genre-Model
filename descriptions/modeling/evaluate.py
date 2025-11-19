from pathlib import Path
from typing import Any, Optional, Union

import joblib
from loguru import logger

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from descriptions.config import MODELS_DIR


app = typer.Typer()

# ---- PRIVATE HELPERS ----
def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return precision_score(y_true, y_pred)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return recall_score(y_true, y_pred)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, micro: bool = False) -> float:
    return f1_score(y_true, y_pred, micro=micro)

# ---- PUBLIC API FUNCTIONS ----
def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return precision, recall, f1, accuracy

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    return classification_report(y_true, y_pred)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    return confusion_matrix(y_true, y_pred)

def roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    return roc_auc_score(y_true, y_pred)

def roc_curve(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    return roc_curve(y_true, y_pred)


@app.command()
def main(
    model_path: Path = MODELS_DIR / "model.joblib",
    data_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
):
    model = load_model(model_path)
    data = load_data(data_path)
    X, y = _split_data(data)
    evaluate_model(model, X, y)


if __name__ == "__main__":
    app()