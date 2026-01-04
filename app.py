from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import os
import logging
from functools import lru_cache
from huggingface_hub import hf_hub_download

# ---------- Configuration ----------
MODEL_ID = "caxacal/article-classifier-model"
HF_TOKEN = os.environ.get("HF_TOKEN")  # set in Render
API_KEY = os.environ.get("API_KEY", "03b8d02ecf8c9898e960ecf2f4dcf287")
CACHE_DIR = "/data/hf_cache"  # Render persistent disk (optional)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Globals ----------
device = None
tokenizer = None
model = None
label_encoder = None
model_loaded = False


# ---------- Load Model ----------
def load_model():
    global device, tokenizer, model, label_encoder, model_loaded

    try:
        logger.info("Loading model and tokenizer from Hugging Face...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR
        )

        model.to(device)
        model.eval()

        # ✅ Download label_encoder.pkl correctly
        label_encoder_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="label_encoder.pkl",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR
        )

        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        logger.info("Model loaded successfully")
        logger.info(f"Categories: {list(label_encoder.classes_)}")

        model_loaded = True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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
        category = label_encoder.inverse_transform([idx.item()])[0]
        results.append({
            "category": category,
            "confidence": float(prob.item())
        })

    return results


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "active",
        "service": "PSITE Abstract Classifier API",
        "model_loaded": model_loaded,
        "categories": list(label_encoder.classes_) if model_loaded else []
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })


@app.route("/classify", methods=["POST"])
def classify():
    try:
        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 503

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

        data = request.get_json()
        abstract = data.get("abstract", "").strip()

        if not abstract:
            return jsonify({"error": "No abstract provided"}), 400
        if len(abstract) < 50:
            return jsonify({"error": "Abstract too short (min 50 characters)"}), 400

        predictions = cached_prediction(abstract[:5000])

        confidence = predictions[0]["confidence"]
        confidence_level = (
            "high" if confidence > 0.8
            else "medium" if confidence > 0.6
            else "low"
        )

        return jsonify({
            "success": True,
            "primary_category": predictions[0]["category"],
            "confidence": confidence,
            "confidence_level": confidence_level,
            "all_predictions": predictions
        })

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": "Internal server error"}), 500


# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
