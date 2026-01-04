from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import logging
from functools import lru_cache

# ---------- Configuration ----------
MODEL_ID = "caxacal/article-classifier-model"
HF_TOKEN = os.environ.get("HF_TOKEN")  # REQUIRED for private repo
API_KEY = os.environ.get("API_KEY", "03b8d02ecf8c9898e960ecf2f4dcf287")
MAX_TOKENS = 128


# 🔴 HARD-CODED LABELS (ORDER MUST MATCH TRAINING)
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

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Globals ----------
device = None
tokenizer = None
model = None
model_loaded = False


# ---------- Load Model ----------
def load_model():
    global device, tokenizer, model, model_loaded

    try:
        logger.info("Loading model from Hugging Face (no pickle, no disk)...")

        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN environment variable is missing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_auth_token=HF_TOKEN
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            use_auth_token=HF_TOKEN
        )

        model.to(device)
        model.eval()

        logger.info("Model loaded successfully")
        logger.info(f"Classes: {CLASSES}")

        model_loaded = True

    except Exception as e:
        logger.exception("MODEL LOAD FAILED")
        model_loaded = False


load_model()

# ---------- Prediction Cache ----------
@lru_cache(maxsize=1000)
def cached_prediction(text):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    top_probs, top_indices = torch.topk(probs[0], k=min(3, probs.shape[-1]))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "category": CLASSES[idx.item()],
            "confidence": float(prob.item())
        })

    return results


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "active",
        "service": "PSITE Abstract Classifier API",
        "version": "1.0",
        "model_loaded": model_loaded,
        "categories": CLASSES if model_loaded else []
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    abstract = data.get("abstract", "").strip()

    if not abstract:
        return jsonify({"error": "No abstract provided"}), 400

    if len(abstract.split()) > 300:
        return jsonify({"error": "Abstract too long (max 300 words)"}), 400

    inputs = tokenizer(
        abstract,
        truncation=True,
        max_length=MAX_TOKENS,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)



# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
