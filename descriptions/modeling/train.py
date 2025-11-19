from pathlib import Path

from descriptions.modeling.model import build_pipeline
from loguru import logger
import typer

from descriptions.config import MODELS_DIR, PROCESSED_DATA_DIR
from descriptions.dataset import load_processed
from descriptions.modeling.model import save_model

app = typer.Typer()


def _split_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = data.drop(columns=["genres"])
    labels = data["genres"]
    return features, labels

def train_model(features: np.ndarray, labels: np.ndarray) -> None:
    model = build_pipeline()
    model.fit(features, labels)
    return model


@app.command()
def main(
    processed_path: Path = PROCESSED_DATA_DIR / "processed_movies.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
):
    logger.info("Loading processed data...")
    data = load_processed(processed_path)
    logger.success("Processed data loaded successfully.")

    logger.info("Splitting data into features and labels...")
    X, y = _split_data(data)
    logger.success("Data split successfully.")

    logger.info("Training model...")
    model = train_model(X, y)
    logger.success("Model trained successfully.")

    logger.info("Saving model...")
    save_model(model, model_path)
    logger.success("Model saved successfully.")


if __name__ == "__main__":
    app()
