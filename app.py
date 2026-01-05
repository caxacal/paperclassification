from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import os

# ---------- CONFIG ----------
MODEL_DIR = "./onnx_model"
API_KEY = os.environ.get("API_KEY", "03b8d02ecf8c9898e960ecf2f4dcf287")
MAX_TOKENS = 128

CLASSES = [
    "AI & Learning Systems",
    "Bioinformatics",
    "Business & Information Systems",
    "Educational Technologies",
    "Media, Interfaces & Applications",
    "Networked & Distributed Systems",
    "Security & Privacy",
    "Software & Systems Engineering"
]

# ---------- LOAD ----------
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"]
)

input_names = [i.name for i in session.get_inputs()]

# ---------- ROUTES ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/classify", methods=["POST"])
def classify():
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    text = (data.get("abstract") or data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No abstract provided"}), 400

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKENS,
        return_tensors="np"
    )

    ort_inputs = {k: inputs[k] for k in input_names}
    logits = session.run(None, ort_inputs)[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    idx = int(np.argmax(probs))
    confidence = float(probs[0][idx])

    return jsonify({
        "prediction": CLASSES[idx],
        "confidence": confidence
    })

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
