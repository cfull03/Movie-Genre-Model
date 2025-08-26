from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from model import TextClassifier
import yaml

app = Flask(__name__)

# Load config from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_config = config["model"]
genre_to_index = config["genre_to_index"]
index_to_genre = {v: k for k, v in genre_to_index.items()}
threshold = config.get("threshold", 0.5)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(config["tokenizer"]["pretrained_model"])

# Load model
model = TextClassifier(**model_config)
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
model.eval()

# Preprocess input text
def preprocess(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=model_config["max_seq_length"],
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    input_ids, attention_mask = preprocess(text)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(0)

    predicted_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    predicted_genres = [index_to_genre[i] for i in predicted_indices]

    return jsonify({
        "genres": predicted_genres,
        "probabilities": probs.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
