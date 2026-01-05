from flask import Flask, request, jsonify
import os
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------
API_KEY = os.environ.get("API_KEY", "03b8d02ecf8c9898e960ecf2f4dcf287")
MAX_TOKENS = 128

MODEL_DIR = "onnx_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

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

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("psite-api")

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- LOAD TOKENIZER ----------------
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
logger.info("Tokenizer loaded")

# ---------------- LOAD ONNX MODEL ----------------
logger.info("Loading ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
logger.info("ONNX model loaded")

# ---------------- UTIL ----------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def predict(text: str):
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length",
        return_tensors="np"
    )

    # ✅ Ensure token_type_ids exist
    if "token_type_ids" not in tokens:
        tokens["token_type_ids"] = np.zeros_like(tokens["input_ids"])

    inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "token_type_ids": tokens["token_type_ids"]
    }

    logits = session.run(None, inputs)[0][0]

    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    top_idx = probs.argsort()[-3:][::-1]

    return [
        {
            "category": CLASSES[i],
            "confidence": float(probs[i])
        }
        for i in top_idx
    ]


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "active",
        "service": "PSITE Abstract Classifier API",
        "engine": "ONNX Runtime",
        "categories": CLASSES
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/classify", methods=["POST"])
def classify():
    # API key check
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    text = (data.get("abstract") or data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No abstract provided"}), 400

    if len(text.split()) > 300:
        return jsonify({"error": "Abstract too long (max 300 words)"}), 400

    results = predict(text)

    return jsonify({
        "prediction": results[0]["category"],
        "confidence": results[0]["confidence"],
        "top_k": results
    })

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("⚠️  Use Gunicorn on Render")
