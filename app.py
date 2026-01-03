from flask import Flask, jsonify, request
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
# ---------- Configuration ----------
MODEL_PATH = './model'
API_KEY = os.environ.get('API_KEY', '03b8d02ecf8c9898e960ecf2f4dcf287')
