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

import os
import urllib.request

# At the top of your app.py, make sure MODEL_PATH is defined
MODEL_PATH = './model'  # or os.path.join(os.getcwd(), 'model')

def download_model_from_server():
    """Download model from your web server via PHP"""
    try:
        base_url = "https://www.article-classifier.xyz/download.php?file="
        files = ['model.safetensors', 'config.json', 'label_encoder.pkl', 
                 'tokenizer_config.json', 'vocab.txt']
        
        # Use MODEL_PATH consistently
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        for filename in files:
            filepath = os.path.join(MODEL_PATH, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                url = base_url + filename
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename} to {filepath}")
            else:
                print(f"{filename} already exists, skipping download")
        
        # Verify all files were downloaded
        for filename in files:
            filepath = os.path.join(MODEL_PATH, filename)
            if not os.path.exists(filepath):
                print(f"ERROR: {filename} not found after download")
                return False
                
        print("All model files downloaded successfully")
        return True
        
    except Exception as e:
        print(f"Failed to download models: {e}")
        return False

# Call this before loading model
download_model_from_server()

# ---------- Load model ----------
def load_model():
    """Initialize model, tokenizer, and label encoder."""
    global device, tokenizer, model, label_encoder
    try:
        logger.info("Loading model and tokenizer...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model path {MODEL_PATH} does not exist")
            return False
            
        # Check for model.safetensors
        safetensors_path = os.path.join(MODEL_PATH, 'model.safetensors')
        if not os.path.exists(safetensors_path):
            logger.error(f"Model file not found: {safetensors_path}")
            return False
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        
        # Load label encoder
        label_encoder_path = os.path.join(MODEL_PATH, 'label_encoder.pkl')
        if not os.path.exists(label_encoder_path):
            logger.error(f"Label encoder not found: {label_encoder_path}")
            return False
            
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
            
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Categories: {list(label_encoder.classes_)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

# Run downloads and load model
#download_model_from_drive()
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



