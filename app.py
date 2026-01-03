from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pickle
import os
import logging
from functools import lru_cache
import gdown

# ---------- Configuration ----------
MODEL_PATH = './model'
API_KEY = os.environ.get('API_KEY', '03b8d02ecf8c9898e960ecf2f4dcf287')

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

# ---------- Download model files ----------
def download_model_from_drive():
    """Download model files from Google Drive using file IDs."""
    files = {
        'model.safetensors': '1dchdkuECFfH148FQYXs3t1sIs-MsjEUw',
        'config.json': '1uVBSx0bUjusg5gmwQMZdbp4K8wJcgrjt',
        'tokenizer.json': '1BgiG333tiuAgRSOc_yQ8CcUQeW3MiCZe',
        'vocab.txt': '1ya4bnH9IYBHdOoxouOvnx4J7H6T6BXrI',
        'tokenizer_config.json': '1HYMmiACJEmEKef9d_rgxL1XNzSlOKnvT',
        'special_tokens_map.json': '1NX1Ms_DA5Rhxe2Epz1rEHU5zpmscZqZo',
        'label_encoder.pkl': '17c7jMT8sWpJNKwBuMxLCDsOiSwMvwGL_'
    }

    os.makedirs(MODEL_PATH, exist_ok=True)

    for filename, file_id in files.items():
        output_path = os.path.join(MODEL_PATH, filename)
        if not os.path.exists(output_path):
            url = f'https://drive.google.com/uc?id={file_id}'
            logger.info(f'Downloading {filename}...')
            gdown.download(url, output_path, quiet=False)
        else:
            logger.info(f'{filename} already exists. Skipping.')

# ---------- Load model ----------
def load_model():
    """Initialize model, tokenizer, and label encoder."""
    global device, tokenizer, model, label_encoder
    try:
        logger.info("Loading model and tokenizer...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)

        logger.info(f"Model loaded. Categories: {list(label_encoder.classes_)}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Run downloads and load model
download_model_from_drive()
model_loaded = load_model()

# ---------- Prediction cache ----------
@lru_cache(maxsize=1000)
def cached_prediction(abstract_text):
    """Cache predictions to speed up repeat queries."""
    if not model_loaded:
        return None

    inputs = tokenizer(
        abstract_text,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    top_probs, top_indices = torch.topk(probs[0], k=min(3, probs.shape[-1]))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        category = label_encoder.inverse_transform([idx.item()])[0]
        results.append({
            'category': category,
            'confidence': float(prob.item())
        })

    return results

# ---------- Endpoints ----------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'active',
        'service': 'PSITE Abstract Classifier API',
        'version': '1.0',
        'model_loaded': model_loaded,
        'categories': list(label_encoder.classes_) if model_loaded else []
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded})

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 503

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer ') or auth_header[7:] != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401

        data = request.get_json()
        abstract = data.get('abstract', '').strip()

        if not abstract:
            return jsonify({'error': 'No abstract provided'}), 400
        if len(abstract) < 50:
            return jsonify({'error': 'Abstract too short (min 50 characters)'}), 400

        predictions = cached_prediction(abstract[:5000])
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500

        confidence = predictions[0]['confidence']
        confidence_level = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'

        return jsonify({
            'success': True,
            'primary_category': predictions[0]['category'],
            'confidence': confidence,
            'confidence_level': confidence_level,
            'all_predictions': predictions
        })

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ---------- Run ----------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
